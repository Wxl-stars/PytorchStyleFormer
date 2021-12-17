# StyleFormer
Official PyTorch implementation for the paper:

[StyleFormer:Real-Time Arbitrary Style Transfer via Parametric Style Composition](https://openaccess.thecvf.com/content/ICCV2021/papers/Wu_StyleFormer_Real-Time_Arbitrary_Style_Transfer_via_Parametric_Style_Composition_ICCV_2021_paper.pdf)

## Overview

This is our overall framework.
![image](https://user-images.githubusercontent.com/53161080/146502829-e6cbfd3d-47f1-48ad-9de1-1ce54a9ccbbc.png)

## Examples

![image](https://user-images.githubusercontent.com/53161080/146366097-1c314181-1d6e-4eb7-af5a-d6b17eece7a8.png)

## Introduction

This is a release of the code of our [paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Wu_StyleFormer_Real-Time_Arbitrary_Style_Transfer_via_Parametric_Style_Composition_ICCV_2021_paper.pdf) ***StyleFormer:Real-Time Arbitrary Style Transfer via Parametric Style Composition***, ICCV 2021

**Authors**: Xiaolei Wu, Zhihao Hu, Lu Sheng\*, Dong Xu (\*corresponding author)

## Update
* 2021.12.17: Upload PyTorch implementation of [StyleFormer](https://openaccess.thecvf.com/content/ICCV2021/papers/Wu_StyleFormer_Real-Time_Arbitrary_Style_Transfer_via_Parametric_Style_Composition_ICCV_2021_paper.pdf).


## Dependencies:

* CUDA 10.1
* python 3.7.7
* pytorch 1.3.1

### Datasets

### MS-COCO

Please download the [MS-COCO](http://msvocds.blob.core.windows.net/coco2014/train2014.zip) dataset.

### WikiArt

Please download the WikiArt dataset from [Kaggle](https://www.kaggle.com/c/painter-by-numbers).

## Download Trained Models

We provide the trained models of StyleFormer and VGG networks.

- StyleFormer
     - [google drive](https://drive.google.com/drive/folders/1l53CJxbMiaU7c17laAT9d8Q_a4arxI28).
     - [BaiduNetdisk](https://pan.baidu.com/s/1gGHYyIwrtoRxZWQLNWHD1w) （Extraction Code: kc44)
- VGG
     - [google drive](https://drive.google.com/drive/folders/1l53CJxbMiaU7c17laAT9d8Q_a4arxI28).
     - [BaiduNetdisk](https://pan.baidu.com/s/1jIxAlTK9LfgPhgd-rGcuew) (Extraction Code: n47y)
        
## Training
```
cd ./scripts
sh train.sh {GPU_ID}
```
## Test
```
git clone https://github.com/Wxl-stars/PytorchStyleFormer.git
cd PytorchStyleFormer

CUDA_VISIBLE_DEVICES={GPU_ID} python test.py \
     --trained_network={PRE-TRAINED_STYLEFORMER_MODEL} \
     --path={VGG_PATH} \
     --input_path={CONTENT_PATH} \
     --style_path={STYLE_PATH} \
     --results_path={RESULTS_PATH} \
```

## Citation
If you find our work useful in your research, please consider citing:
```
@inproceedings{wu2021styleformer,
  title={StyleFormer: Real-Time Arbitrary Style Transfer via Parametric Style Composition},
  author={Wu, Xiaolei and Hu, Zhihao and Sheng, Lu and Xu, Dong},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={14618--14627},
  year={2021}
}
```

## Contact
If you have any questions or suggestions about this paper, feel free to contact:
```
Xiaolei Wu: wuxiaolei@buaa.edu.cn
Zhihao Hu: huzhihao@buaa.edu.cn
```
