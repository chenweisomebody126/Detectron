
0./home/shared/setup.sh && source ~/.bashrc

1. Install docker. (https://docs.docker.com/install/)
2. Install nvidia-docker (https://github.com/NVIDIA/nvidia-docker)
3. Follow the steps of (https://github.com/facebookresearch/Detectron/blob/master/INSTALL.md)
3.1 Clone repo
3.2 Install docker image & run the test image
4. After that, start the container associated with the image: sudo nvidia-docker container run --rm -it detectron:c2-cuda9-cudnn7 bash
5. Now, inside the container you can run the demo: 
python2 tools/infer_simple.py 
--cfg configs/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml 
--output-dir /tmp/detectron-visualizations 
--image-ext jpg 
--wts https://s3-us-west-2.amazonaws.com/detectron/35861858/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml.02_32_51.SgT4y1cO/output/train/coco_2014_train:coco_2014_valminusminival/generalized_rcnn/model_final.pkl 
demo

6.
