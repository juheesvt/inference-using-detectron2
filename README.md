# inference-using-detectron2

#### Start !
```shell
python inference.py --mode directory --dir {dir_name} --save {save_dir_name} \
        --model COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml
```
  
    
    
#### Single image inference
```shell
python inference.py --mode single-img --single-imge ./test.png --save {save_dir_name} \
        --model COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml
```

###### reference  
https://github.com/facebookresearch/detectron2