#  HVC (Dynamic in Static: Hybrid Visual Correspondence for Self-Supervised Video Object Segmentation)
>This repository is the official PyTorch implementation of the paper "**[Dynamic in Static: Hybrid Visual Correspondence for Self-Supervised Video Object Segmentation](https://arxiv.org/abs/2404.13505)**"

>[Gensheng Pei](https://scholar.google.com/citations?user=ihU_QpsAAAAJ&hl),
>[Yazhou Yao](https://scholar.google.com/citations?user=_3Ucwv4AAAAJ&hl),
>[Jianbo Jiao](https://scholar.google.com/citations?user=HkEiMMwAAAAJ&hl),
>[Wenguan Wang](https://scholar.google.com/citations?user=CqAQQkgAAAAJ&hl),
>[Liqiang Nie](https://scholar.google.com/citations?user=yywVMhUAAAAJ&hl),
>[Jinhui Tang](https://scholar.google.com/citations?user=ByBLlEwAAAAJ&hl)

## Abstract
>Conventional video object segmentation (VOS) methods usually necessitate a substantial volume of pixel-level annotated video data for fully supervised learning. In this paper, we present HVC, a **h**ybrid static-dynamic **v**isual **c**orrespondence framework for self-supervised VOS. HVC extracts pseudo-dynamic signals from static images, enabling an efficient and scalable VOS model. Our approach utilizes a minimalist fully-convolutional architecture to capture static-dynamic visual correspondence in image-cropped views. To achieve this objective, we present a unified self-supervised approach to learn visual representations of static-dynamic feature similarity. Firstly, we establish static correspondence by utilizing a priori coordinate information between cropped views to guide the formation of consistent static feature representations. Subsequently, we devise a concise convolutional layer to capture the forward / backward pseudo-dynamic signals between two views, serving as cues for dynamic representations. Finally, we propose a hybrid visual correspondence loss to learn joint static and dynamic consistency representations. Our approach, without bells and whistles, necessitates only one training session using static image data, significantly reducing memory consumption (**16GB**) and training time (**2h**). Moreover, HVC achieves state-of-the-art performance in several self-supervised VOS benchmarks and additional video label propagation tasks.
![HVC](assets/pipeline.png)
><center>The architeture of HVC.</center>

## Highlights
- **Performance:** HVC achieves SOTA self-supervised results on video object segmentation, part propagation, and pose tracking.  
**video object segmentation** (J&F Mean):  
DAVIS16 val-set: **80.1**  
DAVIS17 val-set: **73.1**  
DAVIS17 dev-set: **61.7**  
YouTube-VOS 2018 val-set: **71.9**  
YouTube-VOS 2019 val-set: **71.6**  
VOST val-set: **26.3** J Mean; **15.3** J Last_Mean  
**part propagation** (mIoU):  
VIP val-set: **44.6**  
**pose tracking** (PCK):  
JHMDB val-set: **61.7** PCK@0.1; **82.8** PCK@0.2
- **Efficiency:** HVC necessitates only one training session from static image data (COCO), minimizing memory consumption (∼**16GB**) and training time (∼**2h**).
- **Robustness**: HVC enables the same self-supervised VOS performance with static image datasets ([COCO](https://cocodataset.org/): **73.1**, [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/): **72.0**, and [MSRA10k](https://mmcheng.net/msra10k/): **71.1**) as with the video dataset (e.g. [YouTube-VOS](https://youtube-vos.org/dataset/vos/): **73.1**).

## Requirements
- python 3.9
- torch==1.12.1 
- torchvision==0.13.1
- CUDA 11.3

Create a conda envvironment:
```bash
conda create -n hvc python=3.9 -y
conda activate hvc
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```

## Model Zoo and Results
| Model                           | J&F Mean ↑   |  Download                |
| :-----------------------------: | :----------: | :----------------------: |
| HVC pre-trained on COCO         |  DAVIS17 val-set: 73.1  |  [model](https://github.com/PGSmall/HVC/releases/download/v0.1/releases_models.zip) / [results](https://github.com/PGSmall/HVC/releases/download/v0.1/hvc_coco.zip) |
| HVC pre-trained on PASCAL VOC   |  DAVIS17 val-set: 72.0  |  [model](https://github.com/PGSmall/HVC/releases/download/v0.1/releases_models.zip) / [results](https://github.com/PGSmall/HVC/releases/download/v0.1/hvc_voc.zip) |
| HVC pre-trained on MSRA10k      |  DAVIS17 val-set: 71.1  |  [model](https://github.com/PGSmall/HVC/releases/download/v0.1/releases_models.zip) / [results](https://github.com/PGSmall/HVC/releases/download/v0.1/hvc_msra.zip) |
| HVC pre-trained on YouTube-VOS  |  DAVIS17 val-set: 73.1  |  [model](https://github.com/PGSmall/HVC/releases/download/v0.1/releases_models.zip) / [results](https://github.com/PGSmall/HVC/releases/download/v0.1/hvc_ytb.zip) |

``Note:`` HVC requires only one training session to infer all test datasets for VOS. 

## Dataset Preparation
For the static image dataset:
1. Donwload the COCO 2017 train-set from the [COCO website](https://cocodataset.org/).
2. Please ensure the datasets are organized as following format.
```
data
  |--filelists
  |--Static
    |--JPEGImages
      |--COCO
      |--MSRA10K
      |--PASCAL
```
For the video dataset:
1. Download the DAVIS 2017 val-set from the [DAVIS website](https://davischallenge.org/), the direct download [link](https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip).
2. Download the full YouTube-VOS dataset (version 2018) from the [YouTube-VOS website](https://youtube-vos.org/dataset/vos/), the direct download [link](https://drive.google.com/drive/folders/1bI5J1H3mxsIGo7Kp-pPZU8i6rnykOw7f?usp=sharing). Please move ``ytvos.csv`` from ``data/`` to the path ``data/YTB/2018/train_all_frames``.
3. Please ensure the datasets are organized as following format.
```
data
  |--filelists
  |--DAVIS
      |--Annotations
      |--ImageSets
      |--JPEGImages
  |--YTB
      |--2018
           |--train_all_frames
                   |--ytvos.csv
           |--valid_all_frames
```
``Note:`` Please prepare the following datasets if you want to test the DAVIS dev-set and YouTube-VOS val-set (2019 version).

Download link: [DAVIS dev-set](https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-test-dev-480p.zip), [YTB 2019 val-set](https://drive.google.com/drive/folders/1BWzrCWyPEmBEKm0lOHe5KLuBuQxUSwqz?usp=sharing).

## Training
```bash
# pre-train on COCO
bash ./scripts/run_train_img.sh
```
Or
```bash
# pre-train on YouTube-VOS
bash ./scripts/run_train_vid.sh
```
## Testing
- Download [MoCo V1](https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v1_200ep/moco_v1_200ep_pretrain.pth.tar), and put it in the folder ``checkpoints/``.
- Download [HVC](https://github.com/PGSmall/HVC/releases/download/v0.1/releases_models.zip) and unzip them into the folder ``checkpoints/``.
```bash
# DAVIS 17 val-set
bash ./scripts/run_test.sh hvc davis17
bash ./scripts/run_metrics hvc davis17
```
```bash
# YouTube-VOS val-set
bash ./scripts/run_test.sh hvc ytvos
# Please use the official YouTube-VOS server to calculate scores.
```
```bash
# DAVIS 17 dev-set
bash ./scripts/run_test.sh hvc davis17dev
# Please use the official DAVIS server to calculate scores.
```
``Note:`` YouTube-VOS servers ([2018 server](https://codalab.lisn.upsaclay.fr/competitions/7685) and [2019 server](https://codalab.lisn.upsaclay.fr/competitions/6066)); DAVIS server ([2017 dev-set](https://codalab.lisn.upsaclay.fr/competitions/6812)).

## Qualitative Results

### Sequence Results:
>![HVC](assets/heat_map.png)
><center>Learned representation visualization from HVC without any supervision.</center>

>![HVC](assets/VOS.png)
><center>Qualitative results for video object segmentation.</center>

>![HVC](assets/MP.png)
><center>Qualitative results of the proposed method for (a) body part propagation and (b) human pose tracking .</center>

### Video Results:
<center>DAVIS 2017 val-set</center>

![HVC](assets/bmx-trees.gif) | ![HVC](assets/india.gif)
---|---

<center>YouTube-VOS val-set</center>

![HVC](assets/06a5dfb511.gif) | ![HVC](assets/f1ccd08a3d.gif)
---|---

<center>DAVIS 2017 dev-set</center>

![HVC](assets/girl-dog.gif) | ![HVC](assets/tandem.gif)
---|---


## Acknowledgements
- We thank [PyTorch](https://pytorch.org/), [YouTube-VOS](https://youtube-vos.org/), and [DAVIS](https://davischallenge.org/) contributors.

- Thanks to [videowalk](https://github.com/ajabri/videowalk) for the label propagation codebases.

## Citing HVC
```
@article{pei2024dynamic,
      title={Dynamic in Static: Hybrid Visual Correspondence for Self-Supervised Video Object Segmentation}, 
      author={Pei, Gensheng and Yao, Yazhou and Jiao, Jianbo and Wang, Wenguan and Nie, Liqiang and Tang, Jinhui},
      journal={arXiv preprint arXiv:2404.13505},
      year={2024}
}
```
