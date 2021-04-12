# DM-Count

Official Pytorch implementation of the paper [Distribution Matching for Crowd Counting](https://arxiv.org/pdf/2009.13077.pdf) (NeurIPS, spotlight).

We propose to use Distribution Matching for crowd COUNTing (DM-Count). In DM-Count, we use Optimal Transport (OT) to measure the similarity between the normalized predicted density map and the normalized ground truth density map. To stabilize OT computation, we include a Total Variation loss in our model. We show that the generalization error bound of DM-Count is tighter than that of the Gaussian smoothed methods. Empirically, our method outperforms the state-of-the-art methods by a large margin on four challenging crowd counting datasets: UCF-QNRF, NWPU, ShanghaiTech, and UCF-CC50.

## Prerequisites

Python 3.x

Pytorch >= 1.2

For other libraries, check requirements.txt.

## Getting Started
1. Dataset download

+ Preprocessed VisDrone dataset can be downloaded [here](https://drive.google.com/file/d/1s1yZwCsxWfD-2omN3J6Xs06LSI_hZZn-/view?usp=sharing).
    
2. Training

```
python train.py --data-dir <path to dataset> --counter-type <cc for coarse counter or fc for fine counter>
```


3. During training, the mae values will be saved automatically with the best model into a matrix for the training of [RL model](https://github.com/swsamleo/Crowd_Counting_RL)

The mae data will be saved into data_dir/{}/'base_dir_metric_{}'.format(args.counter_type,args.counter_type).
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
