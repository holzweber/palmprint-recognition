import cv2
import numpy as np
import matplotlib.pyplot as plt
image = cv2.imread("imgMask/100.jpg")
with open(f"csvLandmark/100.csv") as file:
    theta = np.loadtxt(file, delimiter=";")

thetaNew = theta.reshape(2, 9)
theta = np.stack((thetaNew[0], thetaNew[1]), axis=-1)
print(image.shape)
for idx, coord in enumerate(theta):

    currX = coord[0]
    currY = coord[1]
    x = int((600 - 1) / (1 + 1) * (currX - 1) + 600)
    y = int((600 - 1) / (1 + 1) * (currY - 1) + 600)
    image = cv2.circle(image,(x,y),3,(200,0,0),2)
    image = cv2.putText(
        image,  # numpy array on which text is written
        str(idx),  # text
        (x, y),  # position at which writing has to start
        cv2.FONT_HERSHEY_SIMPLEX,  # font family
        1,  # font size
        (209, 80, 0, 255),  # font color
        3)
image = image[...,::-1]
plt.imshow(image)
plt.show()