#!/usr/bin/env python3

from datetime import datetime
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium")

prompt = "a photo of a man in a restaurant eating a giant pizza"
image = pipe(prompt).images[0]

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"pizzaman_{timestamp}.png"

image.save(filename)

