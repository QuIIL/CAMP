# GPC: Generative and general pathology image classifier

## Overview

Implementation of the paper: 

> **CAMP: Classify Anything Model in Pathology** \
> Anh Tien Nguyen and Jin Tae Kwak 

#### Abstract
> There exist numerous diagnostic tasks in pathology. The conventional computational pathology formulates and tackles them as independent and individual image classification problems, thereby resulting in computational inefficiency and high costs. To address the challenges, we propose a generic, unified, and universal framework, named as classify anything model in pathology (CAMP), for pathology image classification. CAMP is a generative, efficient, and adaptive classification model that can continuously adapt to any classification tasks by leveraging the pathology-specific prior knowledge and learning task-specific knowledge with minimal computational cost and without forgetting the knowledge from the existing tasks. We evaluated CAMP on a collection of 15 datasets, including 742,780 pathology images, across 10 distinct classification tasks. CAMP achieves state-of-the-art classification performance regardless of the type of datasets and tasks, substantially outperforming the conventional image classification models in computational pathology. CAMP is also able to reduce up to 94% of computation time and 85% of storage memory in comparison to the conventional models. Our results demonstrate that CAMP is a fundamental transformation in the field of computational pathology at unprecedented fashion, paving the way for the fully digitized and computerized practice of pathology.


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
  <li>PANDA: <a href="https://gigavision.cn/data/news/?nav=DataSet%20Panda&type=nav">link</a></li>
  <li>WSSS4LUAD: <a href="https://wsss4luad.grand-challenge.org/WSSS4LUAD/">link</a></li>
  <li>Kidney: <a href="https://github.com/shyamfec/RCCGNet">link</a></li>
  <li>Liver: <a href="https://link.springer.com/article/10.1007/s11042-023-15176-5">link</a></li>
  <li>Bladder: <a href="https://figshare.com/articles/dataset/Bladder_Whole_Slide_Dataset/8116043">link</a></li>
  <li>BACH: <a href="https://zenodo.org/records/3632035">link</a></li>
  <li>PCam: <a href="https://github.com/basveeling/pcam">link</a></li>
</ul>


## Training
Sample command for training with colon-1 (please check the default arguments)
```
python train.py \
    --dataset colon-1
    --device 0
    --lora_r 6
    --lora_alpha 12
    --out_dir <train_result_saving_dir>
```


### Testing
Sample command for testing with colon-1 (please check the default arguments)
```
python test.py \
    --dataset colon-1
    --device 0
    --model_pth <ckpt_path>
    --out_dir <test_result_saving_dir>
```
