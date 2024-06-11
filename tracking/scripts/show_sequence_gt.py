from dataclasses import dataclass, field
import cv2

from utils import GroundTruth

from ect.helpers import IdSequenceImageGenerator
from typing import Any, Iterable

from icecream import ic

SEQ = "bag"

def main():
    pathroot = f"/home/mateusz/Desktop/Magisterka/vot-workspace/sequences/{SEQ}/"
    path = pathroot + "color/{id:08d}.jpg"
    gen = IdSequenceImageGenerator(path, 1000)

    gtpath = pathroot + "groundtruth.txt"
    gt = GroundTruth(gtpath)

    i = 0
    for image in gen.images():
        bbox = gt.gettruth(i)
        ic(i, bbox)
        i += 1
        pts = [bbox.ll, bbox.lr, bbox.ul, bbox.ur]
        cx = sum(x[0] for x in pts)//4
        cy = sum(x[1] for x in pts)//4
        image = cv2.merge([image, image, image])
        # image = cv2.rectangle(image, bbox.lowleft, bbox.upright, color=(255,0,0))
        image = cv2.drawMarker(image, (cx, cy), (255, 0, 0))
        cv2.imshow("source_image", image)
        cv2.waitKey(30)


if __name__ == "__main__":
    main()