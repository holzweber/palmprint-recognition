# importing the module
import cv2
import os
import numpy as np
clicked = 0
coordinates = []

width = 600 # set width of output image
height = 600 # set height of output image
dir ="TODO" # set input directory, where all hand images are placed

# function to display the coordinates of
# of the points clicked on the image
def click_event(event, x, y, flags, params):
    global clicked, coordinates, height, width
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked += 1 # inc clicked, so we can check if there is a right amount of landmarks afterwards
        # normalize
        currX = ((x-width)*2)/(width-1)+1
        currY = ((y-height)*2)/(height-1)+1
        coordinates.append([currX, currY])
        # displaying the coordinates on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(clicked), (x, y), font, 1, (255, 0, 0), 2)
        cv2.imshow('image', img)

# driver function
if __name__ == "__main__":
    cnt = 0
    for currentIndex, filename in enumerate(os.listdir(dir)):
        print(filename)
        # reading the image
        img = cv2.imread(f'{dir}/{filename}', 1)
        img = cv2.resize(img, (width, height))
        print()
        # displaying the image
        cv2.imshow('image', img)

        # setting mouse handler for the image
        # and calling the click_event() function
        cv2.setMouseCallback('image', click_event)

        # wait for a key to be pressed to exit
        cv2.waitKey(0)

        # close the window
        cv2.destroyAllWindows()
        clicked = 0
        lst = []
        img = cv2.imread(f'{dir}/{filename}', 1)
        img = cv2.resize(img, (width, height))
        for coord in coordinates:
            lst.append(coord[0])
        for coord in coordinates:
            lst.append(coord[1])
        #coordinates = []
        arr = np.array(lst)
        if len(arr) == 18:
            np.savetxt(f"csvLandmark/{cnt}.csv", [arr], fmt='%1.17f', delimiter=';')
            cv2.imwrite(f"imgMask/{cnt}.jpg", img)
            cnt += 1
        else:
            print(f'Error with image: {filename}')
        #print(arr)
        """
        for idx, coord in enumerate(coordinates):
            currX = coord[0]
            currY = coord[1]
            x = int((width - 1) / (1 + 1) * (currX - 1) + width)
            y = int((height - 1) / (1 + 1) * (currY - 1) + height)
            image = cv2.circle(img, (x, y), 3, (200, 0, 0), 2)
            image = cv2.putText(
                image,  # numpy array on which text is written
                str(idx),  # text
                (x, y),  # position at which writing has to start
                cv2.FONT_HERSHEY_SIMPLEX,  # font family
                1,  # font size
                (209, 80, 0, 255),  # font color
                3)
        image = image[..., ::-1]
        cv2.imshow('image', image)
        cv2.waitKey(0)

        # close the window
        cv2.destroyAllWindows()
        """
        coordinates = []
