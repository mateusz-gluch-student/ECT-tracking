import cv2

from ect.helpers import IdSequenceImageGenerator

SEQ = "fernando"

def main():
    path = "/home/mateusz/Desktop/Magisterka/vot-workspace/sequences/wiper/color/{id:08d}.jpg"


    gen = IdSequenceImageGenerator(path, 1000)

    for image in gen.images():

        cv2.imshow("source_image", image)
        cv2.waitKey(30)


if __name__ == "__main__":
    main()