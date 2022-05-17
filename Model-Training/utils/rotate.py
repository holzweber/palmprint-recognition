import PIL
from PIL import Image
import numpy as np
import math
import os

def rotateCoordinates(thetaOld, angle):
    thetaNew = thetaOld.reshape(2, 9)
    thetaNew = np.stack((thetaNew[0], thetaNew[1]), axis=-1)
    lst = []
    for idx, coord in enumerate(thetaNew):
        oldX = coord[0]
        oldY = coord[1]
        thetaNew[idx][0] = oldX * math.cos(angle) - oldY * math.sin(angle)
        thetaNew[idx][1] = oldX * math.sin(angle) + oldY * math.cos(angle)
        lst.append(thetaNew[idx][0])
    for idx, coord in enumerate(thetaNew):
        lst.append(thetaNew[idx][1])
    theta_new = np.array(lst)
    return np.array(theta_new)

p90Degr = math.pi/2 #rotate right
m90Degr = -math.pi/2 #rotate left
leng = len(os.listdir('imgMask'))

for count, filename in enumerate(os.listdir('imgMask')):
    currentleftRotate = leng+count
    currentrightRotate = 2*leng+count
    purename = filename.split('.')[-2]
    img = Image.open(f"imgMask/{filename}")
    imgm90 = img.rotate(90, PIL.Image.NEAREST, expand=1)  # rotate left
    imgp90 = img.rotate(-90, PIL.Image.NEAREST, expand=1)  # rotate right
    imgm90.save(f"imgMask/{currentleftRotate}.jpg")
    imgp90.save(f"imgMask/{currentrightRotate}.jpg")
    with open(f"csvLandmark/{purename}.csv") as file:
        theta = np.loadtxt(file, delimiter=";")
        thetam90 = rotateCoordinates(theta.copy(), m90Degr)
        thetap90 = rotateCoordinates(theta.copy(), p90Degr)
        np.savetxt(f"csvLandmark/{currentleftRotate}.csv",[thetam90], fmt='%1.17f', delimiter=';')
        np.savetxt(f"csvLandmark/{currentrightRotate}.csv",[thetap90], fmt='%1.17f', delimiter=';')
