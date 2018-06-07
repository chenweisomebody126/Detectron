import argparse
import json
import os
from os import path as osp
import sys


def parse_args():
    """Use argparse to get command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('label_dir', help='path to the label dir')
    parser.add_argument('det_path', help='path to output detection file')
    args = parser.parse_args()

    return args


def label2det(label):
    boxes = list()
    for frame in label['frames']:
        for obj in frame['objects']:
            if 'box2d' not in obj:
                continue
            xy = obj['box2d']
            if xy['x1'] >= xy['x2'] and xy['y1'] >= xy['y2']:
                continue
            box = {'name': label['name'],
                   'timestamp': frame['timestamp'],
                   'category': obj['category'],
                   'bbox': [xy['x1'], xy['y1'], xy['x2'], xy['y2']],
                   'score': 1}
            boxes.append(box)
    return boxes


def change_dir(label_dir, det_path):
    if not osp.exists(label_dir):
        print('Can not find', label_dir)
        return
    print('Processing', label_dir)
    input_names = [n for n in os.listdir(label_dir)
                   if osp.splitext(n)[1] == '.json']
    boxes = []
    count = 0
    for name in input_names:
        in_path = osp.join(label_dir, name)
        out = label2det(json.load(open(in_path, 'r')))
        boxes.extend(out)
        count += 1
        if count % 1000 == 0:
            print('Finished', count)
    with open(det_path, 'w') as fp:
        json.dump(boxes, fp, indent=4, separators=(',', ': '))


def main():
    args = parse_args()
    change_dir(args.label_dir, args.det_path)


if __name__ == '__main__':
    main()


gcloud compute ssh --zone=us-west1-b cs231

'bdd100k_test': {
    _IM_DIR:
        _DATA_DIR + '/bdd100k/images/100k/test',
    _ANN_FN:
        _DATA_DIR + '/bdd100k/images/100k/annotations/origin_test.json'
},

python2 tools/train_net.py \
    --multi-gpu-testing \
    --cfg configs/getting_started/tutorial_4gpu_e2e_faster_rcnn_R-50-FPN.yaml \
    OUTPUT_DIR /detectron/detectron/datasets/data/detectron-output USE_NCCL True


python2 tools/train_net.py \
    --multi-gpu-testing \
    --cfg configs/getting_started/tutorial_8gpu_e2e_faster_rcnn_R-50-FPN.yaml \
    OUTPUT_DIR /detectron/detectron/datasets/data/detectron-output USE_NCCL True

sudo nvidia-docker cp /home/shared/Detectron/detectron/datasets/dataset_catalog.py bdd100k:/detectron/detectron/datasets/dataset_catalog.py
sudo nvidia-docker cp /home/shared/Detectron/configs/getting_started/tutorial_4gpu_e2e_faster_rcnn_R-50-FPN.yaml bdd100k:/detectron/configs/getting_started/tutorial_4gpu_e2e_faster_rcnn_R-50-FPN.yaml

sudo nvidia-docker cp /home/shared/Detectron/configs/getting_started/tutorial_8gpu_e2e_faster_rcnn_R-50-FPN.yaml bdd100k:/detectron/configs/getting_started/tutorial_8gpu_e2e_faster_rcnn_R-50-FPN.yaml


sudo nvidia-docker cp /home/shared/Detectron/detectron/datasets/json_dataset_evaluator.py  bdd100k:/detectron/detectron/datasets/json_dataset_evaluator.py
sudo nvidia-docker cp /home/shared/Detectron/detectron/utils/train.py bdd100k:/detectron/detectron/utils/train.py

INFO json_dataset_evaluator.py: 232: classbike
INFO json_dataset_evaluator.py: 233: this is all precisions
INFO json_dataset_evaluator.py: 234: [[0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 ...
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]]
INFO json_dataset_evaluator.py: 236: 0.0
INFO json_dataset_evaluator.py: 232: classbus
INFO json_dataset_evaluator.py: 233: this is all precisions
INFO json_dataset_evaluator.py: 234: [[-100. -100. -100. ... -100. -100. -100.]
 [-100. -100. -100. ... -100. -100. -100.]
 [-100. -100. -100. ... -100. -100. -100.]
 ...
 [-100. -100. -100. ... -100. -100. -100.]
 [-100. -100. -100. ... -100. -100. -100.]
 [-100. -100. -100. ... -100. -100. -100.]]
INFO json_dataset_evaluator.py: 236: nan
INFO json_dataset_evaluator.py: 232: classcar
INFO json_dataset_evaluator.py: 233: this is all precisions
INFO json_dataset_evaluator.py: 234: [[0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 ...
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]]
INFO json_dataset_evaluator.py: 236: 0.0


INFO json_dataset_evaluator.py: 232: ~~~~ Summary metrics ~~~~
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.000
INFO json_dataset_evaluator.py: 199: Wrote json eval results to: /detectron/detectron/datasets/data/detectron-output/test/bdd100k_val/generalized_rcnn/detection_results.pkl
INFO task_evaluation.py:  62: Evaluating bounding boxes is done!
INFO task_evaluation.py: 181: copypaste: Dataset: bdd100k_val
INFO task_evaluation.py: 183: copypaste: Task: box
INFO task_evaluation.py: 186: copypaste: AP,AP50,AP75,APs,APm,APl
INFO task_evaluation.py: 187: copypaste: 0.0000,0.0000,0.0000,0.0000,0.0000,0.0000



Accumulating evaluation results...
DONE (t=3.40s).
INFO json_dataset_evaluator.py: 222: ~~~~ Mean and per-category AP @ IoU=[0.50,0.95] ~~~~
INFO json_dataset_evaluator.py: 223: 0.0
INFO json_dataset_evaluator.py: 231: 0.0
INFO json_dataset_evaluator.py: 231: 0.0
INFO json_dataset_evaluator.py: 231: 0.0
/usr/local/lib/python2.7/dist-packages/numpy/core/fromnumeric.py:2957: RuntimeWarning: Mean of empty slice.
  out=out, **kwargs)
/usr/local/lib/python2.7/dist-packages/numpy/core/_methods.py:80: RuntimeWarning: invalid value encountered in double_scalars
  ret = ret.dtype.type(ret / rcount)
INFO json_dataset_evaluator.py: 231: nan
INFO json_dataset_evaluator.py: 231: 0.0
INFO json_dataset_evaluator.py: 231: nan
INFO json_dataset_evaluator.py: 231: 0.0
INFO json_dataset_evaluator.py: 231: 0.0
INFO json_dataset_evaluator.py: 231: nan
INFO json_dataset_evaluator.py: 231: 0.0
INFO json_dataset_evaluator.py: 232: ~~~~ Summary metrics ~~~~
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.000

 mv /home/shared/Detectron/detectron/datasets/data/bdd100k/images/100k/val
 /home/shared/Detectron/detectron/datasets/data/bdd100k/images/100k/test

NFO json_dataset_evaluator.py: 225: ~~~~ Mean and per-category AP @ IoU=[0.50,0.95] ~~~~
INFO json_dataset_evaluator.py: 226: 0.0
INFO json_dataset_evaluator.py: 233: classtraffic light
INFO json_dataset_evaluator.py: 238: 0.0
INFO json_dataset_evaluator.py: 233: classtraffic sign
INFO json_dataset_evaluator.py: 238: 0.0
INFO json_dataset_evaluator.py: 233: classperson
INFO json_dataset_evaluator.py: 238: 0.0
INFO json_dataset_evaluator.py: 233: classrider
INFO json_dataset_evaluator.py: 238: 0.0
INFO json_dataset_evaluator.py: 233: classbike
INFO json_dataset_evaluator.py: 238: 0.0
INFO json_dataset_evaluator.py: 233: classbus
INFO json_dataset_evaluator.py: 238: 0.0
INFO json_dataset_evaluator.py: 233: classcar
INFO json_dataset_evaluator.py: 238: 0.0
INFO json_dataset_evaluator.py: 233: classmotor
INFO json_dataset_evaluator.py: 238: 0.0
INFO json_dataset_evaluator.py: 233: classtrain
INFO json_dataset_evaluator.py: 238: 0.0
INFO json_dataset_evaluator.py: 233: classtruck
INFO json_dataset_evaluator.py: 238: 0.0
INFO json_dataset_evaluator.py: 239: ~~~~ Summary metrics ~~~~
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.000
INFO json_dataset_evaluator.py: 202: Wrote json eval results to: /detectron/detectron/datasets/data/detectron-output/test/bdd100k_val/generalized_rcnn/detection_results.pkl
INFO task_evaluation.py:  62: Evaluating bounding boxes is done!

gcloud compute scp --recurse /Users/weichen/Downloads/annotations-id/origin_train.json  macbook@cs231n:/home/shared/Detectron/detectron/datasets/data/bdd100k/images/100k/annotations/origin_train.json


sudo nvidia-docker run -it --name bdd100k -v /home/shared/Detectron/:/Detectron detectron:c2-cuda9-cudnn7 bash
sudo nvidia-docker exec -it 7ec7161e421f bash


sudo nvidia-docker run -it --name bdd100k -v /home/shared/Detectron/configs:/detectron/configs -v /home/shared/Detectron/detectron/datasets/data:/detectron/detectron/datasets/data detectron:c2-cuda9-cudnn7 bash


sudo nvidia-docker exec -v /home/shared/Detectron/configs:/detectron/configs -v /home/shared/Detectron/detectron/datasets/data:/detectron/detectron/datasets/data -it 7ec7161e421f detectron:c2-cuda9-cudnn7  bash


python2 tools/test_net.py --cfg configs/getting_started/ml349_2gpu_e2e_faster_rcnn
Inception_ResNetv2.yaml --multi-gpu-testing TEST.WEIGHTS /tmp/detectron-output/train/coco_2014_train/generalized
rcnn/model_final.pkl NUM_GPUS 2

python2 tools/train_net.py \
    --multi-gpu-testing \
    --cfg configs/12_2017_baselines/fast_rcnn_R-50-FPN_1x.yaml \
    OUTPUT_DIR /detectron/detectron/datasets/data/detectron-output USE_NCCL True

sudo nvidia-docker cp  /home/shared/Detectron/configs/12_2017_baselines/fast_rcnn_R-50-FPN_1x.yaml bdd100k:/detectron/configs/12_2017_baselines/fast_rcnn_R-50-FPN_1x.yaml
