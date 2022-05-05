# Python 3 code to rename multiple
# files in a directory or folder

# importing os module
import os
import cv2

# Function to rename multiple files
def main():
    folder = "imgMask"
    for filename in os.listdir(folder):

        src = f"{folder}/{filename}"  # foldername/filename, if .py file is outside folder
        img = cv2.imread(src)
        img = cv2.resize(img,(227,227))
        cv2.imwrite(src,img)


# Driver Code
if __name__ == '__main__':
    # Calling main() function
    main()