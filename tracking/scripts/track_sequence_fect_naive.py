from dataclasses import dataclass, field
import cv2

from utils import GroundTruth

from ect import Config

from ect.helpers import IdSequenceImageGenerator
from typing import Any, Iterable

from icecream import ic

from ect.tracking import (
    Tracker,
    FECTCorrTransformer,
    NaiveMatcher,
    AbsoluteCartesianTracer as PositionTracer
)



SEQ = "bag"
# SEQ = "wiper"
# SEQ = "fernando"

def main():
    pathroot = f"/home/mateusz/Desktop/Magisterka/vot-workspace/sequences/{SEQ}/"
    path = pathroot + "color/{id:08d}.jpg"
    gen = IdSequenceImageGenerator(path, 1000)

    gtpath = pathroot + "groundtruth.txt"
    gt = GroundTruth(gtpath)
    bbox = gt.gettruth(0)
    pts = [bbox.ll, bbox.lr, bbox.ul, bbox.ur]
    cx = sum(x[0] for x in pts)//4
    cy = sum(x[1] for x in pts)//4
    tx = int(.5*(bbox.ll[0]-bbox.lr[0])+.5*(bbox.ul[0]-bbox.ur[0]))
    ty = int(.5*(bbox.ul[1]-bbox.ll[1])+.5*(bbox.ur[1]-bbox.lr[1]))
    tx += 1 if tx%2 else 0
    ty += 1 if ty%2 else 0
    ic(cx, cy)
    ic(tx, ty)

    return

    t = FECTCorrTransformer(Config(), dsize=(360, 150))

    p = PositionTracer(t)

    m = NaiveMatcher(gt=(cx, cy), template_shape=(int(ty), int(tx)), transformer=t, thresh=1e-3, logpolar=True)
    tr = Tracker(gen, m, p.callback)

    for idx, images in enumerate(tr.track()):
        bbox = gt.gettruth(idx)
        ic(idx, bbox)

        image, out, tpl = images 

        rx, ry = m.gt

        pts = [bbox.ll, bbox.lr, bbox.ul, bbox.ur]
        cx = sum(x[0] for x in pts)//4
        cy = sum(x[1] for x in pts)//4
        image = cv2.merge([image, image, image])
        # image = cv2.rectangle(image, bbox.lowleft, bbox.upright, color=(255,0,0))
        image = cv2.drawMarker(image, (cx, cy), (255, 0, 0))
        image = cv2.drawMarker(image, (rx, ry), (0, 0, 255))
        cv2.imshow("source_image", image)
        cv2.waitKey(30)


if __name__ == "__main__":
    main()