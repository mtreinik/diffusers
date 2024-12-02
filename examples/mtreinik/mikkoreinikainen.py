#!/usr/bin/env python3

from datetime import datetime
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("mtreinik/sd35-mikkoreinikainen")

prompt = "a photo of a M@R man"
image = pipe(prompt).images[0]

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"mikko_{timestamp}.png"

image.save(filename)

