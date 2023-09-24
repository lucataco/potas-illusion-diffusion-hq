# This file runs during container build time to get model weights built into the container

# In this example: A Huggingface BERT model
import torch
from transformers import pipeline
from diffusers import AutoencoderKL, ControlNetModel, StableDiffusionControlNetPipeline

BASE_MODEL = "SG161222/Realistic_Vision_V5.1_noVAE"
BASE_CACHE = "model-cache"
CONTROL_CACHE = "control-cache"
VAE_CACHE = "vae-cache"
IMG_CACHE = "img-cache"

def download_model():
    # do a dry run of loading the huggingface model, which will download weights
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
    )

if __name__ == "__main__":
    download_model()