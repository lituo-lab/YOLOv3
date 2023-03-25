import numpy as np
from PIL import Image, ImageDraw
import os

f = open('data.txt', 'r')
datas = f.readlines()
for data in datas:
    data = data.strip().split()
    img_path = os.path.join('images', data[0])
    img = Image.open(img_path)
    w, h = img.size
    _boxes = np.array([float(x) for x in data[1:]])
    boxes = np.split(_boxes, len(_boxes)//5)
    draw = ImageDraw.Draw(img)
    for box in boxes:
        cls, cx, cy, w, h = box
        x1, y1, x2, y2 = cx-0.5*w, cy-0.5*h, cx+0.5*w, cy+0.5*h
        draw.rectangle((x1, y1, x2, y2), outline='red', width=2)

    img.show()
