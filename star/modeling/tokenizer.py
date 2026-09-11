import torch
import torch.nn as nn
from omegaconf import OmegaConf

from star.modeling.maskgit_vqgan import Decoder as Pixel_Decoder
from star.modeling.maskgit_vqgan import Encoder as Pixel_Encoder
from star.modeling.maskgit_vqgan import VectorQuantizer as Pixel_Quantizer
from star.utils.checkpoint_loading import load_raw_state_dict


class PretrainedTokenizer(nn.Module):
    def __init__(self, tokenizer_checkpoint):
        super().__init__()
        state_dict = load_raw_state_dict(tokenizer_checkpoint)
        try:
            embedding_weight = state_dict["quantize.embedding.weight"]
        except KeyError as exc:
            raise ValueError(
                "Tokenizer checkpoint is missing quantize.embedding.weight."
            ) from exc
        if (
            not isinstance(embedding_weight, torch.Tensor)
            or embedding_weight.ndim != 2
        ):
            raise ValueError(
                "Tokenizer quantize.embedding.weight must be a two-dimensional tensor."
            )
        codebook_size, embedding_dim = embedding_weight.shape
        if codebook_size <= 0 or embedding_dim <= 0:
            raise ValueError(
                "Tokenizer quantize.embedding.weight must have non-empty dimensions."
            )

        conf = OmegaConf.create(
            {
                "channel_mult": [1, 1, 2, 2, 4],
                "num_resolutions": 5,
                "dropout": 0.0,
                "hidden_channels": 128,
                "num_channels": 3,
                "num_res_blocks": 2,
                "z_channels": embedding_dim,
            }
        )
        self.encoder = Pixel_Encoder(conf)
        self.decoder = Pixel_Decoder(conf)
        self.quantize = Pixel_Quantizer(
            num_embeddings=codebook_size,
            embedding_dim=embedding_dim,
        )
        self.load_state_dict(state_dict, strict=True)

        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def encode(self, x):
        hidden_states = self.encoder(x)
        codebook_indices = self.quantize(hidden_states)
        return codebook_indices.detach()

    @property
    def codebook_size(self):
        return self.quantize.embedding.num_embeddings

    @torch.no_grad()
    def decode_tokens(self, token_ids):
        if token_ids.ndim != 3:
            raise ValueError(
                "token_ids must have shape [batch, grid_height, grid_width]."
            )
        quantized_states = self.quantize.get_codebook_entry(token_ids)
        rec_images = self.decoder(quantized_states)
        rec_images = torch.clamp(rec_images, 0.0, 1.0)
        return rec_images.detach()
