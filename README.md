# StyleFormer
Official PyTorch implementation for the paper [StyleFormer:Real-Time_Arbitrary_Style_Transfer_via_Parametric_Style_Composition](https://openaccess.thecvf.com/content/ICCV2021/papers/Wu_StyleFormer_Real-Time_Arbitrary_Style_Transfer_via_Parametric_Style_Composition_ICCV_2021_paper.pdf)

## Examples
![image](https://user-images.githubusercontent.com/53161080/146366097-1c314181-1d6e-4eb7-af5a-d6b17eece7a8.png)

## Update
* 2021.12.17: Upload PyTorch implementation of [StyleFormer](https://openaccess.thecvf.com/content/ICCV2021/papers/Wu_StyleFormer_Real-Time_Arbitrary_Style_Transfer_via_Parametric_Style_Composition_ICCV_2021_paper.pdf).


## Dependencies:
python 3.7.7

pytorch 1.3.1

## Usage
### Test
* Download pre-trained models from [google drive](https://drive.google.com/drive/folders/1l53CJxbMiaU7c17laAT9d8Q_a4arxI28).
* Download the pre-trained [vgg](https://drive.google.com/drive/folders/19F3dti6Oo_vVxgpgLiDGK4DFbckXfOD-) networks.
```
git clone https://github.com/Wxl-stars/PytorchStyleFormer.git
cd PytorchStyleFormer

CUDA_VISIBLE_DEVICES=$1 python test.py \
     --trained_network={PRE-TRAINED_STYLEFORMER_MODEL} \
     --path={VGG_PATH} \
     --input_path={CONTENT_PATH} \
     --style_path={STYLE_PATH} \
     --results_path={RESULTS_PATH} \
     --selection={'Ax+b'|'Ax'|'x+b'|'b'|'aAx+b'} \
     --inter_selection={'A1x+b2'|'A2x+b1'|'(a1A1+a2A2)x+b1'|'(a1A1+a2A2)x+b2'| '(a1A1+a2A2)x+a1*b1+a2*b2'} 
```
### Train from scratch
#### Prepare the dataset
- Download the [MS-COCO](http://msvocds.blob.core.windows.net/coco2014/train2014.zip) dataset.
- Download the WikiArt dataset from [Kaggle](https://www.kaggle.com/c/painter-by-numbers).

```
CUDA_VISIBLE_DEVICES=$1 python train.py 
```

## Citation
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
