## Contrastive Predictive Learning with Transformer for Video Representation Learning

This repository contains the implementation of Contrastive Predictive Learning with Transformer (CPCTR). 


### CPCTR Results

Performance result from this implementation:

| Pretrain Dataset| Resolution | Backbone | Finetune Acc@1 (UCF101) | Finetune Acc@1 (HMDB51) |
|----|----|----|----|----|
|UCF101|128x128|2d-R18|99.3|82.4|


### Installation

The implementation should work with python >= 3.6, pytorch >= 0.4, torchvision >= 0.2.2. 

The repo also requires cv2, tensorboardX >= 1.7, joblib, tqdm, ipdb.

### Prepare data

Please download HMDB51 and UCF101 dataset along with their three splits, then use /ProcessData to extract frames from video.

### Self-supervised training (CPCTR)

Change directory `cd CPCTrans/CPCTrans/`

* example: train CPCTR using 1 GPUs, with ResNet18 backbone, on UCF101 dataset with 128x128 resolution, for 300 epochs
  ```
  python main.py --net resnet18 --dataset ucf101 --batch_size 16 --img_dim 128 --epochs 300
  ```

### Evaluation: supervised action classification

Change directory `cd CPCTrans/Evaluate/`

* example: finetune pretrained CPCTR weights (replace `{model.pth.tar}` with pretrained CPCTR model)
  ```
  python test.py --net resnet18 --dataset ucf101 --batch_size 16 --img_dim 128 --pretrain {model.pth.tar} --train_what ft --epochs 300
  ```

* example (continued): test the finetuned model (replace `{finetune_model.pth.tar}` with finetuned classifier model)
  ```
  python test.py --net resnet18 --dataset ucf101 --batch_size 16 --img_dim 128 --test {finetune_model.pth.tar}
  ```


### Contact

For any questions, feel free to contact Yue Liu at yueliu1229@qq.com.



