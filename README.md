# DM-Count

Official Pytorch implementation of the paper [Distribution Matching for Crowd Counting](https://arxiv.org/pdf/2009.13077.pdf) (NeurIPS, spotlight).

We propose to use Distribution Matching for crowd COUNTing (DM-Count). In DM-Count, we use Optimal Transport (OT) to measure the similarity between the normalized predicted density map and the normalized ground truth density map. To stabilize OT computation, we include a Total Variation loss in our model. We show that the generalization error bound of DM-Count is tighter than that of the Gaussian smoothed methods. Empirically, our method outperforms the state-of-the-art methods by a large margin on four challenging crowd counting datasets: UCF-QNRF, NWPU, ShanghaiTech, and UCF-CC50.

## Prerequisites

Python 3.x

Pytorch >= 1.2

For other libraries, check requirements.txt.

## Getting Started
1. Dataset download

+ Preprocessed VisDrone dataset can be downloaded [here](https://drive.google.com/file/d/1WbQnL8l8ZjGkzBonvu8vCp1JhdQOILma/view?usp=sharing). Remove the "images" and "ground-truth" folder first. Before training the fine crowd counter, rename the folder "cropped-images" and "cropped-ground truth" to "images" and "ground-truth" respectively. Before training the coarse crowd counter, rename the folder "downsampled-cropped-images" and "downsampled-cropped-ground truth" to "images" and "ground-truth" respectively. 
+ Here we use 640x640px and 160x160px images to train the fine and coarse level counters. The coarse images represented with 160x160px is the downsampled version of the high resolution images represented with 640x640 px. 

    
2. Training

```
python train.py --dataset sha --data-dir <path to dataset> --device <gpu device id>.
```
Remember to modify the train.py file "   elif args.dataset.lower() == 'sha': args.crop_size = 640" the crop_size shall be changed to 160 if it's for the coarse counter training.

3. Save the mAR values into a matrix during inference for the training of [RL model](https://github.com/swsamleo/Crowd_Counting_RL)

```
python output_inference.py (need to rewrite model path and the name of the folder (from os.mkdir('/content/content/content/VisDrone2020-CC/test_data/base_dir_metric_cd') to os.mkdir('/content/content/content/VisDrone2020-CC/test_data/base_dir_metric_fd')) which stores the MAE number from inference) if training on fine counter
```

## Pretrained models

Pretrained models on UCF-QNRF, NWPU, Shanghaitech part A and B can be found in pretrained_models folder or [Google Drive](https://drive.google.com/drive/folders/10U7F4iW_aPICM5-qJq21SXLLkzlum9tX?usp=sharing)

## References
If you find this work or code useful, please cite:

```
@inproceedings{wang2020DMCount,
  title={Distribution Matching for Crowd Counting},
  author={Boyu Wang and Huidong Liu and Dimitris Samaras and Minh Hoai},
  booktitle={Advances in Neural Information Processing Systems},
  year={2020},
}
```
