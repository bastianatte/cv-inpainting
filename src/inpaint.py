from diffusers import StableDiffusionInpaintPipeline
import torch
from PIL import Image

def load_pipeline(model_id="runwayml/stable-diffusion-inpainting", device=None, dtype=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = dtype or (torch.float16 if device == "cuda" else torch.float32)
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None
    )
    pipe = pipe.to(device)
    if device == "cuda":
        pipe.enable_xformers_memory_efficient_attention = getattr(pipe, "enable_xformers_memory_efficient_attention", lambda: None)
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
    return pipe

@torch.inference_mode()
def inpaint(pipe, image: Image.Image, mask: Image.Image, prompt: str = "", negative_prompt: str = "",
            num_inference_steps: int = 30, guidance_scale: float = 7.5, seed: int | None = None):
    """
    image: RGB
    mask:  bianca sulle aree da rimuovere, nera sul resto (L o RGB)
    """
    if mask.mode != "L":
        mask = mask.convert("L")
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)
    else:
        generator = None

    result = pipe(
        prompt=prompt or "",
        image=image,
        mask_image=mask,
        negative_prompt=negative_prompt or None,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator
    ).images[0]
    return result
