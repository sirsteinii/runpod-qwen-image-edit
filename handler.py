import base64
import io

import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline
import runpod

# ---------------------------------------------------------
# Modell einmal global laden (Cold Start ist langsam, danach schneller)
# ---------------------------------------------------------
print("Loading Qwen/Qwen-Image-Edit-2509 ...")
pipe = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2509",
    torch_dtype=torch.float16,  # bfloat16 geht nur auf manchen GPUs
)
pipe.to("cuda")
pipe.set_progress_bar_config(disable=True)
print("Model loaded on CUDA.")


def decode_image_from_base64(b64_str: str) -> Image.Image:
    """Base64 (mit oder ohne 'data:image/...;base64,') -> PIL.Image."""
    if "," in b64_str:
        _, b64 = b64_str.split(",", 1)
    else:
        b64 = b64_str
    img_bytes = base64.b64decode(b64)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def encode_image_to_base64(img: Image.Image) -> str:
    """PIL.Image -> Base64 (ohne data:-Prefix)."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()
    return base64.b64encode(img_bytes).decode("utf-8")


# ---------------------------------------------------------
# RunPod-Handler
# ---------------------------------------------------------
def handler(event):
    """
    Erwartetes Input-JSON:

    {
      "input": {
        "prompt": "text",
        "image_base64": "...."   // PNG/JPEG als Base64
      }
    }

    Optional (kannst du später erweitern):
      true_cfg_scale, negative_prompt, num_inference_steps, guidance_scale, seed
    """
    inp = event.get("input", {}) or {}

    prompt = inp.get("prompt", "")
    image_b64 = inp.get("image_base64")

    if not image_b64:
        # Falls kein Bild geschickt wurde -> Dummy weißes Bild (nur für Tests)
        image1 = Image.new("RGB", (512, 512), (255, 255, 255))
    else:
        image1 = decode_image_from_base64(image_b64)

    # QwenImageEditPlus erwartet zwei Bilder. Wir duplizieren das eine.
    image2 = image1

    negative_prompt = inp.get("negative_prompt", " ")
    true_cfg_scale = float(inp.get("true_cfg_scale", 4.0))
    num_inference_steps = int(inp.get("num_inference_steps", 40))
    guidance_scale = float(inp.get("guidance_scale", 1.0))
    seed = int(inp.get("seed", 0))

    generator = torch.Generator(device="cuda").manual_seed(seed)

    with torch.inference_mode():
        out = pipe(
            image=[image1, image2],
            prompt=prompt,
            negative_prompt=negative_prompt,
            generator=generator,
            true_cfg_scale=true_cfg_scale,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=1,
        )

    out_image = out.images[0]
    image_base64 = encode_image_to_base64(out_image)
    data_url = f"data:image/png;base64,{image_base64}"

    return {
        "image_base64": image_base64,
        "image_data_url": data_url,
    }


# RunPod-Serverless starten
runpod.serverless.start({"handler": handler})
