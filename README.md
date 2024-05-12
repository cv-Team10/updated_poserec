# updated_poserec

model based on 
https://github.com/microsoft/human-pose-estimation.pytorch

The code is designed to operate based on the pretrained model of the link

This code is designed for the task of generating the appropriate pose based on the background
We built an appropriate parallelel dataset for 30000 background and good poses.
Based on Dataset, finetuning was conducted for the above model.



Download mpii and coco pretrained models from above link
Please download under
${ROOT}/pretrained/pose_resnet_50_256x192.pth.tar
