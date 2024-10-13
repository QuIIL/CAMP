# CAMP: Continuous and Adaptive Learning Model in Pathology

## Overview

Implementation of the paper (under review): 

> [**CAMP: Classify Anything Model in Pathology**]([/guides/content/editing-an-existing-page#modifying-front-matter](https://arxiv.org/abs/2407.09030)) \
> Anh Tien Nguyen, Keunho Byeon, Kyungeun Kim, Boram Song, Seoung Wan Chae, and Jin Tae Kwak


#### Abstract
> There exist numerous diagnostic tasks in pathology. Conventional computational pathology formulates and tackles them as independent and individual image classification problems, thereby resulting in computational inefficiency and high costs. To address the challenges, we propose a generic, unified, and universal framework, called a continuous and adaptive learning model in pathology (CAMP), for pathology image classification. CAMP is a generative, efficient, and adaptive classification model that can continuously adapt to any classification task by leveraging pathology-specific prior knowledge and learning task-specific knowledge with minimal computational cost and without forgetting the knowledge from the existing tasks. We evaluated CAMP on 22 datasets, including 1,171,526  patches and 11,811 pathology slides, across 17 classification tasks. CAMP achieves state-of-the-art classification performance on a wide range of datasets and tasks at both patch- and slide-levels and reduces up to 94\% of computation time and 85\% of storage memory in comparison to the conventional classification models. Our results demonstrate that CAMP can offer a fundamental transformation in pathology image classification, paving the way for the fully digitized and computerized pathology practice.


#### Architecture


## Environment set up
```
git clone https://github.com/QuIIL/CAMP
cd CAMP
conda create --name CAMP --file requirements.txt
conda activate CAMP
pip install -r requirements.txt
```


## Datasets
<ul>
  <li>Colon-1 and Colon-2: <a href="https://github.com/QuIIL/KBSMC_colon_cancer_grading_dataset">link</a> </li>
  <li>UHU: <a href="https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OCYCMP">link</a></li>
  <li>UBC: <a href="https://gleason2019.grand-challenge.org/">link</a></li>
  <li>AGGC: <a href="https://aggc22.grand-challenge.org/">link</a></li>
  <li>Gastric: <a href="https://github.com/QuIIL/KBSMC_gastric_cancer_grading_dataset">link</a></li>
  <li>K19 and K16: <a href="https://zenodo.org/record/53169">link</a></li>
  <li>PANDA: <a href="https://gigavision.cn/data/news/?nav=DataSet%20Panda&type=nav](https://panda.grand-challenge.org/data/">link</a></li>
  <li>WSSS4LUAD: <a href="https://wsss4luad.grand-challenge.org/WSSS4LUAD/">link</a></li>
  <li>Kidney: <a href="https://github.com/shyamfec/RCCGNet">link</a></li>
  <li>Liver: <a href="https://link.springer.com/article/10.1007/s11042-023-15176-5">link</a></li>
  <li>Bladder: <a href="https://figshare.com/articles/dataset/Bladder_Whole_Slide_Dataset/8116043">link</a></li>
  <li>BACH: <a href="https://zenodo.org/records/3632035">link</a></li>
  <li>PCam: <a href="https://github.com/basveeling/pcam">link</a></li>
  <li>HunCRC_P: <a href="https://github.com/basveeling/pcam](https://doi.org/10.6084/m9.figshare.c.5927795.v1">link</a></li>
  <li>HunCRC_W: <a href="https://github.com/basveeling/pcam](https://doi.org/10.7937/tcia.9cjf-0127">link</a></li>
  <li>BRACS: <a href="https://www.bracs.icar.cnr.it/">link</a></li>
  <li>DHMC: <a href="https://bmirds.github.io/KidneyCancer/">link</a></li>
  <li>UniToPatho: <a href="https://bmirds.github.io/KidneyCancer/">link</a></li>
  <li>CAMELYON16: <a href="https://camelyon16.grand-challenge.org/">link</a></li>
</ul>


## Models
<ul>
  <li>ConvNeXt-B: <a href="https://pytorch.org/vision/stable/models/generated/torchvision.models.convnext_base.html#torchvision.models.ConvNeXt_Base_Weights">link</a> </li>
  <li>RegNet: <a href="https://pytorch.org/vision/stable/models/generated/torchvision.models.regnet_x_16gf.html#torchvision.models.RegNet_X_16GF_Weights">link</a></li>
  <li>ResNeXt50: <a href="https://pytorch.org/vision/stable/models/generated/torchvision.models.resnext50_32x4d.html#torchvision.models.ResNeXt50_32X4D_Weights">link</a></li>
  <li>SwinV2-B: <a href="https://pytorch.org/vision/stable/models/generated/torchvision.models.swin_v2_b.html#torchvision.models.Swin_V2_B_Weights">link</a></li>
  <li>ViT-B: <a href="https://pytorch.org/vision/stable/models/generated/torchvision.models.vit_b_16.html#torchvision.models.ViT_B_16_Weights">link</a></li>
  <li>PLIP: <a href="https://huggingface.co/vinid/plip">link</a></li>
  <li>CTransPath: <a href="https://github.com/Xiyue-Wang/TransPath">link</a></li>
  <li>UNI: <a href="https://huggingface.co/MahmoodLab/UNI">link</a></li>
  <li>Phikon: <a href="https://huggingface.co/owkin/phikon">link</a></li>
  <li>GPC: <a href="https://github.com/QuIIL/GPC">link</a></li>
  <li>GIT-B: <a href="https://huggingface.co/docs/transformers/en/model_doc/git">link</a></li>


## Step by Step Instruction

### Step 1: Training
The code for training is mainly based on the file `train.py`. 
The arguments are important for the training setting includes `dataset` (dataset to train), `lora_r` (rank of LoRA), `lora_alpha` (alpha of LoRA), and `out_dir` to save the training results. Please refer to the file `train.py` for the default arguments of other arguments.

Sample command for training with colon-1.
```
python train.py \
    --dataset colon-1
    --device 0
    --lora_r 6
    --lora_alpha 12
    --out_dir <train_result_saving_dir>
```


### Step 2: Testing
The code for training is mainly based on the file `test.py`. 
The arguments are important for the training setting includes `dataset` (dataset to test), `model_pth` (path of a test model), and `out_dir` to save the testing results. Please refer to the file `test.py` for the default arguments of other arguments.

Sample command for testing with colon-1.
```
python test.py \
    --dataset colon-1
    --device 0
    --model_pth <ckpt_path>
    --out_dir <test_result_saving_dir>
```
