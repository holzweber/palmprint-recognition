# Python 3 code to rename multiple
# files in a directory or folder

# importing os module
import os


# Function to rename multiple files
def main():
    folder = "output"
    folderCSV = "csvLandmark"
    offset = 1381
    for count, filename in enumerate(os.listdir(folder)):
        dst = f"{str(count+offset)}.jpg"
        src = f"{folder}/{filename}"  # foldername/filename, if .py file is outside folder
        dst = f"{folder}/{dst}"

        dstCSV = f"{str(count+offset)}.csv"
        csvSrc = f"{folderCSV}/{filename.split('.')[0]}.csv"  # prune jpg ending
        csvDst = f"{folderCSV}/{dstCSV}"
        print(f"{src} -> {dst}")
        print(f"{csvSrc} -> {csvDst}")
        # rename() function will
        # rename all the files
        os.rename(src, dst)
        #os.rename(csvSrc, csvDst)


# Driver Code
if __name__ == '__main__':
    # Calling main() function
    main()