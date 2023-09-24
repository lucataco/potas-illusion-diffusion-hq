from potassium import Potassium, Request, Response
import io
import gzip
import torch
import qrcode
import base64
from PIL import Image
from io import BytesIO
from typing import List
from PIL.Image import LANCZOS
from diffusers import (
    AutoencoderKL,
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler
)

BASE_MODEL = "SG161222/Realistic_Vision_V5.1_noVAE"
BASE_CACHE = "model-cache"
CONTROL_CACHE = "control-cache"
VAE_CACHE = "vae-cache"
IMG_CACHE = "img-cache"

SAMPLER_MAP = {
    "DPM++ Karras SDE": lambda config: DPMSolverMultistepScheduler.from_config(config, use_karras=True, algorithm_type="sde-dpmsolver++"),
    "Euler": lambda config: EulerDiscreteScheduler.from_config(config),
}

app = Potassium("my_app")

# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    device = 0 if torch.cuda.is_available() else -1
    vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse",
            torch_dtype=torch.float16,
            cache_dir=VAE_CACHE,
        )
    controlnet = ControlNetModel.from_pretrained(
        "monster-labs/control_v1p_sd15_qrcode_monster",
        torch_dtype=torch.float16,
        cache_dir=CONTROL_CACHE,
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        BASE_MODEL,
        controlnet=controlnet,
        vae=vae,
        safety_checker=None,
        torch_dtype=torch.float16,
        cache_dir=BASE_CACHE,
    ).to("cuda")
   
    context = {
        "pipe": pipe
    }
    return context

def resize_for_condition_image(input_image, width, height):
    input_image = input_image.convert("RGB")
    W, H = input_image.size
    k = float(min(width, height)) / min(H, W)
    H *= k
    W *= k
    H = int(round(H / 64.0)) * 64
    W = int(round(W / 64.0)) * 64
    img = input_image.resize((W, H), resample=LANCZOS)
    return img

def generate_qrcode(qr_code_content, background, border, width, height):
        print("Generating QR Code from content")
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_H,
            box_size=10,
            border=border,
        )
        qr.add_data(qr_code_content)
        qr.make(fit=True)

        qrcode_image = qr.make_image(fill_color="black", back_color=background)
        qrcode_image = resize_for_condition_image(qrcode_image, width, height)
        return qrcode_image

# @app.handler runs for every call
@app.handler("/")
def handler(context: dict, request: Request) -> Response:
    # Parameters
    prompt = request.json.get("prompt")
    negative_prompt = request.json.get("negative_prompt")
    controlnet_conditioning_scale = 1.0
    image = request.json.get("image")
    seed = int(request.json.get("seed"))
    guidance_scale = 7.5
    num_inference_steps = 40
    width = 768
    height = 768
    num_outputs = 1
    qr_code_content = "https://catacolabs.com"
    # Model
    pipe = context.get("pipe")
    # Seed
    if seed == None:
        seed = torch.randint(0, 2**32, (1,)).item() 
    print(f"Seed: {seed}")

    image = base64.b64decode(image.encode("utf-8"))
    image_io = io.BytesIO(image)

    # Controlism img
    if image is None:
        if qrcode_background == "gray":
            qrcode_background = "#808080"
        image = generate_qrcode(
            qr_code_content, background=qrcode_background, border=1, width=width, height=height,
        )
    else:
        image = Image.open(image_io)

    # Run pipeline
    output = pipe(
        prompt=[prompt] * num_outputs,
        negative_prompt=[negative_prompt] * num_outputs,
        image=[image] * num_outputs,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        generator=torch.Generator().manual_seed(seed),
        num_inference_steps=num_inference_steps,
    )

    img_out = output.images[0]
    # Make return image smaller to fit under 1MB
    img_out = img_out.resize((576, 576), Image.ANTIALIAS)

    fname = f"out.png"
    buffered = BytesIO()
    img_out.save(buffered, format="png", optimize=True, quality=80)
    img_data = buffered.getvalue()
    compressed_img_data = gzip.compress(img_data)
    compressed_img_str = base64.b64encode(compressed_img_data).decode('utf-8')

    return Response(
        json = {"outputs": compressed_img_str}, 
        status=200
    )

if __name__ == "__main__":
    app.serve()