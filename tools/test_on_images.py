import argparse
import cv2
from detectron2.utils.logger import setup_logger
# setup_logger()
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
from pathlib import Path
from detectron2.structures import Instances
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--weights", required=True, type=str)
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--conf-threshold", default=0.6, type=float)
    args = parser.parse_args()
    
    return args

def main(args):
    # Get the configuration ready
    logger = setup_logger()
    cfg = get_cfg()
    cfg.merge_from_file(args.config)
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.conf_threshold
    dicts = list(DatasetCatalog.get("lisa_bulb_coco_train"))
    metadata = MetadataCatalog.get("lisa_bulb_coco_train")

    predictor = DefaultPredictor(cfg)
    inputs = Path(args.input)
    if inputs.is_file():
        im = cv2.imread(str(inputs))
        outputs = predictor(im)
        v = Visualizer(im[:,:,::-1], metadata)
        v = v.draw_instance_predictions(outputs['instances'].to('cpu'))
        img = v.get_image()[:, :, ::-1]
        cv2.imshow('image', img)
        cv2.waitKey(0)
    else:
        img_lists = list(inputs.glob('*.jpg'))
        for img in img_lists:
            im = cv2.imread(str(img))
            outputs = predictor(im)
            v = Visualizer(im[:,:,::-1], metadata)
            v = v.draw_instance_predictions(outputs['instances'].to('cpu'))
            img = v.get_image()[:, :, ::-1]
            cv2.imshow('image', img)
            cv2.waitKey(0)

if __name__ == "__main__":
    args = parse_args()
    main(args)