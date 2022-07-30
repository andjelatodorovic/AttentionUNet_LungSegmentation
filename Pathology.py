import random
import cv2
import numpy as np
import math, random
from typing import List, Tuple
from PIL import Image, ImageDraw

def intersect(img, thresh):
    tab = np.array(thresh)
    image = np.array(img)
    return np.where(image==tab, image, 0)
    
def add_noise(img, medval, img_rows=256, img_cols=256):
    variation = np.random.randint(-15,15, size=(img_rows,img_cols))
    tab = np.array(img)
    noise = np.zeros((img_rows,img_cols))
    a = np.where(tab != 255, noise, variation + medval)
    a = np.where(a < 255, a, 255)
    a = np.where(a > 0, a, 0)
    return a

def imagefinal(img,lung):
    current = np.array(img)
    if len(lung.shape) >= 3 :
        original=np.array(lung[:, :, 0])
    else:
        original=np.array(lung)
    return np.where(current<original, original, current)

def createCircle(img,nb):
    for i in range(nb):
        posX= random.randint(100,412)
        posY = random.randint(100, 412)
        rad = random.randint(20,70)
        circle = cv2.circle(img,(posX,posY),radius=rad,color=255,thickness=-1)
        
def clip(value, lower, upper):
    return min(upper, max(value, lower))

def random_angle_steps(steps: int, irregularity: float) -> List[float]:
    angles = []
    lower = (2 * math.pi / steps) - irregularity
    upper = (2 * math.pi / steps) + irregularity
    cumsum = 0
    for i in range(steps):
        angle = random.uniform(lower, upper)
        angles.append(angle)
        cumsum += angle

    cumsum /= (2 * math.pi)
    for i in range(steps):
        angles[i] /= cumsum
    return angles

def generate_polygon(center: Tuple[float, float], avg_radius: float,irregularity: float, spikiness: float,num_vertices: int) -> List[Tuple[float, float]]:
   
    if irregularity < 0 or irregularity > 1:
        raise ValueError("Irregularity must be between 0 and 1.")
    if spikiness < 0 or spikiness > 1:
        raise ValueError("Spikiness must be between 0 and 1.")

    irregularity *= 2 * math.pi / num_vertices
    spikiness *= avg_radius
    angle_steps = random_angle_steps(num_vertices, irregularity)
    points = []
    angle = random.uniform(0, 2 * math.pi)
    for i in range(num_vertices):
        radius = clip(random.gauss(avg_radius, spikiness), 0, 2 * avg_radius)
        point = (center[0] + radius * math.cos(angle),
                 center[1] + radius * math.sin(angle))
        points.append(point)
        angle += angle_steps[i]

    return points

def createFakeImage(lung,mask,minval,maxval, img_rows=128, img_cols=128):
	img = np.zeros((img_rows, img_cols))
	createCircle(img, 20)
	thresh = np.zeros((img_rows, img_cols))
	ret, thresh = cv2.threshold(mask, 5, 255, cv2.THRESH_BINARY)
	image = intersect(img,thresh)
	medval = random.randint(minval, maxval)
	img = add_noise(image,medval, img_rows, img_cols)
	lung = imagefinal(img, lung)
	image = np.array(lung)
	return image

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def createPolygonFakeImage(lung,mask,minval,maxval, img_rows=128, img_cols=128, center=(60, 60), avg_radius=15,irregularity=0.35,spikiness=0.2,num_vertices=32):
    img = Image.new('RGB', (img_rows, img_cols), (0,0,0))
    vertices = generate_polygon(center,avg_radius,irregularity,spikiness,num_vertices)
    im_px_access = img.load()
    draw = ImageDraw.Draw(img)
    draw.polygon(vertices, fill = (255, 255, 255))
    draw.line(vertices + [vertices[0]], width=2, fill=(0,0,0))
    img = img.convert("L")
    img = np.array(img)
    img.reshape(128,128)
    thresh = np.zeros((img_rows, img_cols))
    ret, thresh = cv2.threshold(mask, 5, 255, cv2.THRESH_BINARY)
    image = intersect(img,thresh)
    medval = random.randint(minval, maxval)
    img = add_noise(image,medval, img_rows, img_cols)
    lung = imagefinal(img, lung)
    image = np.array(lung)
    return image

def main ():
    lung= cv2.imread('lung.pgm')
    mask= cv2.imread('msk.pgm')
    #mask = mask[:,:,0]
    minval=70
    maxval=230
    
    createFakeImage(lung,mask,minval,maxval)
    return

if __name__ == '__main__':
    main()



