from __future__ import annotations

import torch

from .masks import normalize_token_grid_shape


@torch.no_grad()
def sample_fn(
    generator,
    tokenizer,
    class_ids,
    guidance_scale,
    guidance_scale_power,
    temperature,
    token_ids=None,
    masks=None,
    device="cuda",
    return_tensor=False,
    return_token_ids=False,
    orders=None,
):
    generator.eval()
    tokenizer.eval()
    if not isinstance(class_ids, torch.Tensor):
        class_ids = torch.as_tensor(class_ids, dtype=torch.long, device=device)
    if orders is None:
        raise ValueError("STAR sampling requires explicit BFS spanning-tree orders.")
    generated_token_ids = generator.generate(
        class_ids=class_ids,
        guidance_scale=guidance_scale,
        guidance_scale_power=guidance_scale_power,
        temperature=temperature,
        token_ids=token_ids,
        masks=masks,
        orders=orders,
    )

    height, width = normalize_token_grid_shape(generator.token_grid_shape)
    expected_shape = (generated_token_ids.shape[0], height * width)
    if generated_token_ids.shape != expected_shape:
        raise RuntimeError(
            "The generator returned token IDs with shape "
            f"{tuple(generated_token_ids.shape)}; expected {expected_shape}."
        )
    generated_images = tokenizer.decode_tokens(
        generated_token_ids.reshape(generated_token_ids.shape[0], height, width)
    )
    if return_tensor:
        return generated_images

    generated_images = torch.clamp(generated_images, 0.0, 1.0)
    generated_images = (
        (generated_images * 255.0)
        .permute(0, 2, 3, 1)
        .to("cpu", dtype=torch.uint8)
        .numpy()
    )

    if return_token_ids:
        return generated_images, generated_token_ids
    return generated_images
