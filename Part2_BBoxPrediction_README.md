# SAI-EVA7-Capstone

## Part - 1
**1. We take the encoded image (dxH/32xW/32) and send it to Multi-Head Attention (FROM WHERE DO WE TAKE THIS ENCODED IMAGE?)**

This is coming from the DETR object detection encoder output. It is reduced to the size of [the model embeddings(default 256), height/32 and width/32] dimension.

**2. We do something here to generate NxMxH/32xW/32 maps. (WHAT DO WE DO HERE?)**

we use multi-head attention module to check the cosine similarity between each object embeddings and each pixel in the image to genrate the rough low resolution mask. This low resolution mask is further upsampled by the following CNN layers to get a clean object level mask.

**3. Then we concatenate these maps with Res5 Block (WHERE IS THIS COMING FROM?)**

Res5 block is coming from the input projection(conv2d layer with kernal size of 1, outdim equal to model dimension) of the feature map that is obtained from encoder backbone convolution model (ResNet50/101). 

![Panoptic Head](https://github.com/sdev2030/SAI-EVA7-Capstone/blob/main/resource_imgs/arch-1.png)

**4. Then we perform the above steps (EXPLAIN THESE STEPS)**

After applying a high threshold of 0.85, we select  the top bboxes that would form as queries to the MultiHead attention block. This multi-head attention block also takes the object detection encoder output feature map. In the multi-head attention module we check the cosine similarity between each object embeddings and each pixel in the image to genrate the rough low resolution mask. This low resolution mask is further upsampled by the following CNN layers (with image feature map from encoder backbone) along with mask logits to get a clean object level mask. Then we perform pixel-wise argmax to get the final panoptic segmentation output.

## High-Level solution for Capstone Project:

The given concrete defect detection [dataset](https://github.com/ZHANGKEON/DIS-YOLO) has original images, defect labels with segmentation masks.

First we have to convert segmentation masks to the Bounding box annotations using the code link specified in class notes. Then we need to add the classes for stuff (other  than the concrete defects)  as it is required for DETR panoptic segmentation task. Then we need to prepare the images in coco dataset format with labelled masks annoations json for train and validation datasets (80/20 split for train and val).

Once we have the images, labels and the bounding box data in coco annotation format, we have to first fine tune the backbone of the DETR object detection model for 20/30 epochs with low learning rate for backbone (1e-5) while the rest of the network will train for another 100/200 epochs at a slight higher learning rate of 1e-3. Then we need to check the output of the trained model to confirm that the object detection with BBox prediction is working fine. if not adjust/check the data setup, model definitions, and training code to make sure that the object detection part of the DETR model works.

Then we need to add the panoptic head to the model and freeze the object detection model. We continue train just the panoptic segmentation head for 25 epochs to check that the panoptic segmentation outputs are predicted correctly.

Finally we need to generate a grid output of 100 test images that shows the original image, ground truth image and the predicted panoptic segmentation image.
