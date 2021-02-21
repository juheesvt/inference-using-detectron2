import argparse

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger


# import some common libraries
import numpy as np
import os, json, cv2, random
from glob import glob
from pathlib import Path

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


def parse_args(args) -> argparse:

    parser = argparse.ArgumentParser(description='Inference images using Detectron2')

    parser.add_argument('--model-help', help="https://github.com/facebookresearch/detectron2/tree/master/configs", type=str)
    parser.add_argument('--mode', help='single-image or directory', default="directory", type=str)
    parser.add_argument('--single-img', help='single image path', type=str)
    parser.add_argument('--dir', help='directory path', type=str)
    parser.add_argument('--save', help='svae directory path', default="./save", type=str)
    parser.add_argument('--model', help='', default="COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml", type=str)

    print(vars(parser.parse_args(args)))
    return parser.parse_args(args)


def config(model: str) -> DefaultPredictor:
    setup_logger()
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)
    return DefaultPredictor(cfg), cfg


def inference(image_path: str, save_path: str, predictor: DefaultPredictor, cfg: get_cfg) -> None:
    images = glob(image_path + '/*')

    for image in images:
        read_img = cv2.imread(image)
        img_name = image.split("/")[-1]

        # inference
        outputs = predictor(read_img)

        # Visualization
        v = Visualizer(read_img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        # save image
        save_path = os.path.join(save_path + "/" + img_name)
        cv2.imwrite(save_path, out.get_image()[:, :, ::-1])


def main(args=None):

    args = parse_args(args)

    # variable definition
    model = args.model
    mode = args.mode
    save_path = args.save
    image_path = args.dir if mode == "directory" else args.single_img

    Path(save_path).mkdir(parents=True, exist_ok=True)

    # inference start
    predictor, cfg = config(model)
    inference(image_path, save_path, predictor, cfg)


if __name__ == '__main__':
    main()