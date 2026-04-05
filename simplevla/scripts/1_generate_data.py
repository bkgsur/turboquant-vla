# Script 1: Generate Synthetic Data

#PIL A 2-tuple (width, height) in pixels

# for i in range(1000):
#     image = blank white image (480x640)
#     x, y = random integers (0-640, 0-480)
#     color = random RGB
#     draw circle at (x, y)
#     save image as "image_0001.png"
#     save label as (x, y) in CSV

from  PIL import Image, ImageDraw
import numpy as np
import os
from openpyxl import load_workbook

rng = np.random.default_rng()
max = 1000
width = 480
height = 640
radius = 5
x_ints = rng.integers(low=0, high=height,size=max)
y_ints = rng.integers(low =0, high = width, size=max)
folder_path = "image"
os.makedirs(folder_path, exist_ok=True)

colors = ['black', 'red', 'green', 'blue', 'cyan', 'magenta', 'yellow']


for i in range(max):
    img = Image.new ('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    x = x_ints[i]
    y = y_ints[i]
    random_color = np.random.choice(colors)
    draw.circle((x,y), radius=radius, fill= random_color, width=2)
    image_filename = f"{i}.png"
    image_path = os.path.join(folder_path,image_filename)
    img.save(image_path)


