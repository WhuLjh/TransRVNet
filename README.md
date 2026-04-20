<div align="center"> 

## TransRVNet

</div>

A transferable open-source data-driven framework for cross-domain multimodal extraction and analysis of RV.
The final large-scale mapping dataset for key Southeast Asian cities can be **downloaded from [**BaiduNetdisk**](https://pan.baidu.com/s/19Z2aYG1-cfr4ezvLGt_Sog?pwd=5pku)**. 

### Data folder structure

The `data` folder is structured as:
```text
data
├── source-domain_rv
│   ├── building_height
│   │   ├── test
│   │   │   ├── 0.png
│   │   │   ├── 1.png
│   │   │   ├── ...
│   │   ├── train
│   │   └── val
│   ├── img
│   ├── ndvi
│   ├── osm_building
│   └── semantic
└── traget-domain_rv
```
The `traget-domain_rv` folder has the same structure as the `source-domain_rv` folder.

**Note:** `traget-domain_rv/semantic/train` stores preliminary segmentation labels of traget-domain data predicted by the source-domain model, which serve as initial pseudo labels.

## Environment

Reference to [DELIVER](https://github.com/InSAI-Lab/DELIVER) and [segment-anything](https://github.com/facebookresearch/segment-anything).

## Data preparation

**Train and validation dataset can be downloaded from [**BaiduNetdisk**](https://pan.baidu.com/s/1Hh1EFp4v4d9UuRENfFilaA)** and will be made available after publication of this article.

## Training

Before training, please download [pre-trained SegFormer](https://drive.google.com/drive/folders/10XgSW8f7ghRs9fJ0dE-EV8G2E_guVsT5?usp=sharing) and [pre-trained SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) to checkpoints/pretrained, for example:
```text
checkpoints/pretrained/sam
├── sam_vit_b_01ec64.pth
checkpoints/pretrained/segformer
├── mit_b2.pth
```

To train TransRVNet model, please use change yaml file for `--cfg`. An example using a NVIDIA GeForce RTX 4070 Ti GPU are:  

```bash
python tools/train_mm.py --cfg ./configs/rv_source_domain.yaml

python tools/adapt_mm.py --cfg ./configs/rv_UDASS.yaml
```

## Prediction
**Model weights can be downloaded from [**BaiduNetdisk**](https://pan.baidu.com/s/1J1YK2Rnbn8rJQkq8munVaQ)** and will be made available after publication of this article.

Modify `--cfg` to respective config file, and run:
```bash
python tools/infer_mm_large_scale.py --cfg ./configs/rv_source_domain.yaml

python tools/infer_mm_large_scale.py --cfg ./configs/rv_UDASS.yaml
```

## Acknowledgements
Thanks for the public repositories:

- [DELIVER](https://github.com/InSAI-Lab/DELIVER)
- [segment-anything](https://github.com/facebookresearch/segment-anything)

