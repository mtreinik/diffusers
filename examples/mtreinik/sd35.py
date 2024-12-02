#!/usr/bin/env python3

from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium")

prompt = "a photo of a man"
image = pipe(prompt).images[0]
image.save('man1.png')

