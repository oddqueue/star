import os
from functools import partial
from typing import Callable, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import Mlp
from torch.utils.checkpoint import checkpoint

from star.sampling.masks import normalize_token_grid_shape


_PRETRAINED_POSITION_PADDING = 1024
_PRETRAINED_TIMESTEP_PADDING = 100


# util function
def build_causal_mask(seq_length):
    mask = torch.empty(seq_length, seq_length)
    mask.fill_(float("-inf"))
    mask.triu_(1)  # zero out the lower diagonal
    return mask

# weight init
def init_weights(module):
    if (isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or
     isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d)):
        module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        if module.bias is not None:
            module.bias.data.zero_()
        if module.weight is not None:
            module.weight.data.fill_(1.0)

# attention layer with KV cache supported
class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.kv_cache = False
        self.k_cache = None
        self.v_cache = None

    def reset_kv_cache(self):
        self.k_cache = None
        self.v_cache = None

    def forward(self, x: torch.Tensor, attn_mask=None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.kv_cache:
            if self.k_cache is None and self.v_cache is None:
                k_cache = k
                v_cache = v
            else:
                if N not in (1, 2):
                    raise RuntimeError(
                        "KV-cache updates require one or two tokens, "
                        f"got x.shape {x.shape}"
                    )
                k_cache = torch.cat([self.k_cache, k], dim=-2)
                v_cache = torch.cat([self.v_cache, v], dim=-2)

            self.k_cache = k_cache
            self.v_cache = v_cache

            k = k_cache
            v = v_cache

        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class FinalLayer(nn.Module):
    def __init__(self, dim, norm_layer):
        super().__init__()
        self.norm_final = norm_layer(dim, elementwise_affine=False)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 2*dim)
        )
    
    def forward(self, x, c):
        scale, shift = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        return x
    

# basic transformer block
class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )


    def forward(self, x: torch.Tensor, attn_mask=None, c = None) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), attn_mask=attn_mask)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class STARGenerator(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # parse the configs
        embed_dim = config.model.generator.hidden_size
        depth = config.model.generator.num_hidden_layers
        num_heads = config.model.generator.num_attention_heads
        intermediate_size = config.model.generator.intermediate_size
        mlp_ratio = intermediate_size / embed_dim

        token_grid_shape = normalize_token_grid_shape(
            config.model.generator.token_grid_shape
        )
        image_seq_len = token_grid_shape[0] * token_grid_shape[1]
        target_codebook_size = config.model.vq_model.codebook_size
        condition_num_classes = config.model.generator.condition_num_classes
        norm_layer=partial(nn.LayerNorm, eps=1e-6)

        dropout_rate = config.model.generator.dropout
        attn_dropout_rate = config.model.generator.attn_drop
   
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                qk_norm=True,
                proj_drop=dropout_rate,
                attn_drop=attn_dropout_rate,
                norm_layer=norm_layer)
            for _ in range(depth)])

        self.embeddings = nn.Embedding(
            target_codebook_size + 1 + condition_num_classes + 1, embed_dim)

        self.pos_embed = nn.init.trunc_normal_(
            nn.Parameter(
                torch.zeros(
                    1,
                    image_seq_len + _PRETRAINED_POSITION_PADDING,
                    embed_dim,
                )
            ),
            0.0,
            0.02,
        )

        self.target_aware_pos_embed = nn.init.trunc_normal_(
            nn.Parameter(
                torch.zeros(
                    1,
                    image_seq_len + _PRETRAINED_POSITION_PADDING,
                    embed_dim,
                )
            ),
            0.0,
            0.02,
        )

        # number of steps == image_seq_len
        self.timesteps_embeddings = nn.init.trunc_normal_(
            nn.Parameter(
                torch.zeros(
                    1,
                    image_seq_len + _PRETRAINED_TIMESTEP_PADDING,
                    embed_dim,
                )
            ),
            0.0,
            0.02,
        )
        self.adaln_before_head = FinalLayer(embed_dim, norm_layer=norm_layer)
        self.lm_head = nn.Linear(embed_dim,
                                 target_codebook_size, bias=True)
        self.condition_num_classes = condition_num_classes
        self.token_grid_shape = token_grid_shape
        self.image_seq_len = image_seq_len
        self.target_codebook_size = target_codebook_size
        self.none_condition_id = self.condition_num_classes + self.target_codebook_size + 1
        
        self.apply(init_weights)

        attn_mask = build_causal_mask(self.image_seq_len + 2)
        self.register_buffer('attn_mask', attn_mask, persistent=False)

        self.use_gradient_checkpointing = config.model.generator.get(
            "use_gradient_checkpointing", False
        )

        # init for adaln-zero.

        nn.init.constant_(self.adaln_before_head.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaln_before_head.adaLN_modulation[-1].bias, 0)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def save_pretrained_weights(
        self,
        save_directory: Union[str, os.PathLike],
        save_function: Callable = None,
        state_dict: Optional[Dict[str, torch.Tensor]] = None,
    ):
        if os.path.isfile(save_directory):
            raise NotADirectoryError(
                f"Provided path ({save_directory}) should be a directory, not a file"
            )

        if save_function is None:
            save_function = torch.save

        os.makedirs(save_directory, exist_ok=True)

        if state_dict is None:
            state_dict = self.state_dict()
        weights_name = "pytorch_model.bin"

        save_function(state_dict, os.path.join(save_directory, weights_name))

        print(f"Model weights saved in {os.path.join(save_directory, weights_name)}")

    def enable_kv_cache(self):
        for block in self.blocks:
            block.attn.kv_cache = True
            block.attn.reset_kv_cache()

    def disable_kv_cache(self):
        for block in self.blocks:
            block.attn.kv_cache = False
            block.attn.reset_kv_cache()

    def shuffle(self, x, orders):
        batch_size, seq_len = x.shape[:2]
        batch_indices = (
            torch.arange(batch_size, device=x.device)
            .unsqueeze(1)
            .expand(-1, seq_len)
        )
        shuffled_x = x[batch_indices, orders]
        return shuffled_x

    def unshuffle(self, shuffled_x, orders):
        # Unshuffle the tensor based on the original orders
        batch_size, seq_len = shuffled_x.shape[:2]
        batch_indices = (
            torch.arange(batch_size, device=shuffled_x.device)
            .unsqueeze(1)
            .expand(-1, seq_len)
        )
        unshuffled_x = torch.zeros_like(shuffled_x)
        unshuffled_x[batch_indices, orders] = shuffled_x
        return unshuffled_x

    def preprocess_condition(self, class_ids, cond_drop_prob=0.0):
        # Replace dropped class IDs with the unconditional token ID.
        drop_label_mask = (
            torch.rand_like(class_ids, dtype=torch.float) < cond_drop_prob
        )
        condition_ids = class_ids + self.target_codebook_size + 1
        condition_ids[drop_label_mask] = self.none_condition_id
        return condition_ids

    def get_none_condition(self, condition_ids):
        return torch.full_like(condition_ids, self.none_condition_id)
    
    def forward(self, token_ids, condition_ids, orders):
        if orders is None:
            raise ValueError("STAR requires explicit spanning-tree BFS orders.")
        target_token_ids = self.shuffle(token_ids, orders)
        logits = self.forward_fn(token_ids, condition_ids, orders)
        return logits, target_token_ids

    def forward_fn(self, token_ids, condition_ids, orders, is_sampling=False):
        # Token space:
        #  [0, codebook_size - 1]                       : those are the learned quantized image tokens
        #  codebook_size                                : the mask token used to mask image tokens
        #  [codebook_size + 1, codebook_size + condition_num_classes]: ImageNet class tokens
        #  codebook_size + 1 + condition_num_classes: the class drop label

        # prepend condition token
        token_ids = torch.cat(
            [
                condition_ids.view(condition_ids.shape[0], -1),
                token_ids.view(token_ids.shape[0], -1),
            ],
            dim=1,
        )
        embeddings = self.embeddings(token_ids)
        condition_token = embeddings[:, 0]

        # prepare positional embeddings.
        # shuffle pos embed
        pos_embed = self.pos_embed.repeat(token_ids.shape[0], 1, 1)
        # The traversal permutation does not affect prefix tokens.
        prefix = 2
        pos_embed_prefix = pos_embed[:, :prefix]
        pos_embed_postfix = self.shuffle(pos_embed[:, prefix:prefix+self.image_seq_len], orders)

        # prepare target-aware positional embeddings.
        target_aware_pos_embed = self.target_aware_pos_embed.repeat(
            token_ids.shape[0], 1, 1
        )
        target_aware_pos_embed_postfix = self.shuffle(target_aware_pos_embed[:, prefix:prefix+self.image_seq_len], orders)

        if not is_sampling:
            # During training, shuffle the image-token embeddings into the
            # sampled traversal order. Sampling already supplies that order.
            embeddings = torch.cat([embeddings[:, :1], self.shuffle(embeddings[:, 1:], orders)], dim=1)

        x = embeddings
        # prepend the cls token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add original pos embed
        x = x + torch.cat([pos_embed_prefix, pos_embed_postfix], dim=1)[:, :x.shape[1]]

        # add target-aware pos embed
        target_aware_pos_embed = torch.cat(
            [torch.zeros_like(x[:, :prefix-1]), target_aware_pos_embed_postfix, torch.zeros_like(x[:, -1:])], dim=1
        )
        x = x + target_aware_pos_embed[:, :x.shape[1]]

        # causal attention masking
        attn_mask = self.attn_mask[:x.shape[1], :x.shape[1]]
        
        # Add a distinct timestep embedding to each condition representation.
        condition_token = condition_token.unsqueeze(1) + self.timesteps_embeddings[:, :x.shape[1]]

        if self.blocks[0].attn.kv_cache:
            if self.blocks[0].attn.k_cache is not None and self.blocks[0].attn.v_cache is not None:
                # only need to process the last token
                x = x[:, -1:]
                attn_mask = None
                # Only keep the final condition representation.
                condition_token = condition_token[:, -1:]

        for blk in self.blocks:
            if self.training and self.use_gradient_checkpointing:
                x = checkpoint(
                        blk.forward, x, attn_mask, condition_token, use_reentrant=False)
            else:
                x = blk(x, attn_mask=attn_mask, c=condition_token)

        if not self.blocks[0].attn.kv_cache:
            # remove cls token
            x = x[:, prefix - 1:]
            condition_token = condition_token[:, prefix - 1:]


        x = self.adaln_before_head(x, condition_token)
        x = self.lm_head(x)

        return x

    @torch.no_grad()
    def generate(
        self,
        class_ids,
        guidance_scale,
        temperature,
        guidance_scale_power,
        orders,
        token_ids=None,
        masks=None,
    ):
        if orders is None:
            raise ValueError("STAR requires explicit spanning-tree BFS orders.")
        condition_ids = self.preprocess_condition(class_ids, cond_drop_prob=0.0)
        device = condition_ids.device
        num_samples = condition_ids.shape[0]

        self.enable_kv_cache()
        try:
            generated_token_ids = torch.full(
                (num_samples, 0), -1, device=device
            )
            if masks is not None:
                regeneration_mask = torch.tensor(
                    [
                        [step in mask for step in range(self.image_seq_len)]
                        for mask in masks
                    ],
                    dtype=torch.bool,
                    device=device,
                )
                forced_token_ids = self.shuffle(
                    token_ids.masked_fill(regeneration_mask, -1), orders
                )
            else:
                forced_token_ids = None

            guidance_power = (
                torch.ones((1), device=device) * guidance_scale_power
            )
            for step in range(self.image_seq_len):
                scale_step = (
                    1
                    - torch.cos(
                        ((step / self.image_seq_len) ** guidance_power)
                        * torch.pi
                    )
                ) * 0.5
                cfg_scale = (guidance_scale - 1) * scale_step + 1

                if guidance_scale != 0:
                    logits = self.forward_fn(
                        torch.cat(
                            [generated_token_ids, generated_token_ids],
                            dim=0,
                        ),
                        torch.cat(
                            [
                                condition_ids,
                                self.get_none_condition(condition_ids),
                            ],
                            dim=0,
                        ),
                        torch.cat([orders, orders], dim=0),
                        is_sampling=True,
                    )
                    cond_logits = logits[:num_samples]
                    uncond_logits = logits[num_samples:]
                    logits = uncond_logits + (
                        cond_logits - uncond_logits
                    ) * cfg_scale
                else:
                    logits = self.forward_fn(
                        generated_token_ids,
                        condition_ids,
                        orders,
                        is_sampling=True,
                    )

                logits = logits[:, -1] / temperature
                probs = F.softmax(logits, dim=-1)
                sampled = torch.multinomial(probs, num_samples=1)
                if forced_token_ids is not None:
                    forced_token_ids_at_step = forced_token_ids[:, step : step + 1]
                    sampled = torch.where(
                        forced_token_ids_at_step == -1,
                        sampled,
                        forced_token_ids_at_step,
                    )
                generated_token_ids = torch.cat(
                    (generated_token_ids, sampled), dim=-1
                )

            return self.unshuffle(generated_token_ids, orders)
        finally:
            self.disable_kv_cache()
