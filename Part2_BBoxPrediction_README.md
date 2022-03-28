# SAI-EVA7-Capstone - Part 2 - BBox Prediction Model

## Train / Test Data Split

We have to first download the concrete defect detection [dataset](https://github.com/ZHANGKEON/DIS-YOLO). It has original image files and segmentation mask files that identifies the defect as part of the mask file name. Since the requirement is to split the data in 80:20 ratio for train and test, we have to make sure that proportionate number of defects appear in both datasets.

There are 3 defect types, Crack, rebar and Spall, We use the [notebook](add the git link for notebook) to figure out the number of files of each defect type that needs to be in train and test dataset and put them under their separate directories. We use just the train dataset for training the BBox model and then gather metrics on Test dataset using the trained model. 

## Coco formatted Annotations

We need to prepare a coco formatted annotation file for train and test dataset. The annotation has to be generated using the image mask files provided in the dataset. Our code is based the [repo](https://github.com/chrise96/image-to-coco-json-converter) that is used to take mask image as input and generate coco json formatted annotation file. 

As we observe some noise in mask file, we treat pixel value 1 as zero(full black) and 254 as 255(full white). Also we drop Bboxes that don't have atleast 3pixels height and 3pixels height as they seem to be more noise rather then real object bounding boxes.

We use [notebook](add git link for notebook) for generating coco formatted annotation files for train and test dataset.

Directory structure needs to be modified as follows

```
.
|-- annotations
|   |-- custom_train.json
|   `-- custom_val.json
|-- train2017  (this holds original train images)
`-- val2017    (this holds original val images)
```

We also need coco formatted panoptic annotations (for fine tuning the mask head). This segmentation annotation format is at the image file level instead of the object instance level.

We use [notebook](add git link for notebook) for generating coco formatted panoptic annotation files for train and test dataset.

Directory structure needs to be modified as follows for panoptic data.

```
.
|-- annotations
|   |-- panoptic_train2017.json 
|   `-- panoptic_val2017.json
|-- panoptic_train2017  (this holds mask train images)
`-- panoptic_val2017    (this holds mask val images)

```

## Fine tune pretrained DETR model for our defect BBox prediction 

In order to fine-tune the pretrained model, we clone the [repo](https://github.com/woctezuma/detr.git) and its branch 'finetune'. Following commands will do that.

```
git clone https://github.com/woctezuma/detr.git
cd ./detr/
git checkout finetune 
```

Once the repo is downloaded, ensure --

- dataset is prepared as in the previous step
- pretrained DETR model is downloaded 
	- the model can be downloaded form this [repo](https://github.com/facebookresearch/detr#model-zoo)
	- once the model is downloaded, remove the class weights and save it. This will help us to utilize our list of custom defect classes instead of coco default classes.
- fine tune the model for 25 epochs with low leaning rate of 1e-5 using the following command

```
!python main.py \
  --dataset_file "custom" \
  --coco_path "/home/sn/EVA7/capstone/data/custom/" \
  --output_dir "outputs" \
  --resume "detr-r50_no-class-head.pth" \
  --num_classes $num_classes \
  --epochs 25
```

## Evaluation Results

Following are the loss graphs, mAP and test image results.

![loss and mAP]()
![loss graphs]()
![error graphs]()

![test rebar]()
![test spall]()


