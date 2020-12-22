import argparse
import os
import torch
import shutil
from train_helper import Trainer


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--data-dir', default='data/UCF-Train-Val-Test', help='data path')
    parser.add_argument('--dataset', default='sha', help='dataset name: qnrf, nwpu, sha, shb')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='the initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='the weight decay')
    parser.add_argument('--resume', default='', type=str,
                        help='the path of resume training model')
    parser.add_argument('--max-epoch', type=int, default=75,
                        help='max training epoch')
    parser.add_argument('--val-epoch', type=int, default=5,
                        help='the num of steps to log training information')
    parser.add_argument('--val-start', type=int, default=50,
                        help='the epoch start to val')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='train batch size')
    parser.add_argument('--device', default='0', help='assign device')
    parser.add_argument('--num-workers', type=int, default=3,
                        help='the num of training process')
    parser.add_argument('--crop-size', type=int, default=512,
                        help='the crop size of the train image')
    parser.add_argument('--wot', type=float, default=0.1, help='weight on OT loss')
    parser.add_argument('--wtv', type=float, default=0.01, help='weight on TV loss')
    parser.add_argument('--reg', type=float, default=10.0,
                        help='entropy regularization in sinkhorn')
    parser.add_argument('--num-of-iter-in-ot', type=int, default=100,
                        help='sinkhorn iterations')
    parser.add_argument('--norm-cood', type=int, default=0, help='whether to norm cood when computing distance')
    parser.add_argument('--counter-type', default='cc', help='cc for coarse counter or fc for fine counter')
    

    args = parser.parse_args()
    ori_dataset_dir=args.data_dir
    if not os.path.exists(os.path.join(ori_dataset_dir,args.counter_type)):
             os.mkdir(os.path.join(ori_dataset_dir,args.counter_type))
    os.mkdir(os.path.join(ori_dataset_dir,args.counter_type,'train_data'))
    os.mkdir(os.path.join(ori_dataset_dir,args.counter_type,'train_data','images'))
    os.mkdir(os.path.join(ori_dataset_dir,args.counter_type,'train_data','ground-truth'))
    os.mkdir(os.path.join(ori_dataset_dir,args.counter_type,'test_data'))
    os.mkdir(os.path.join(ori_dataset_dir,args.counter_type,'test_data','images'))
    os.mkdir(os.path.join(ori_dataset_dir,args.counter_type,'test_data','ground-truth'))
    if args.counter_type == 'cc':
       shutil.copytree(os.path.join(ori_dataset_dir,train_data,downsampled-cropped-images),os.path.join(ori_dataset_dir,args.counter_type,train_data,images))
       shutil.copytree(os.path.join(ori_dataset_dir,train_data,downsampled-cropped-ground-truth),os.path.join(ori_dataset_dir,args.counter_type,train_data,ground-truth))
       shutil.copytree(os.path.join(ori_dataset_dir,test_data,downsampled-cropped-ground-truth),os.path.join(ori_dataset_dir,args.counter_type,test_data,ground-truth))
       shutil.copytree(os.path.join(ori_dataset_dir,test_data,downsampled-cropped-images),os.path.join(ori_dataset_dir,args.counter_type,test_data,images))
       args.crop_size = 160
       args.dataset_dir= os.path.join(ori_dataset_dir,args.counter_type)
    elif args.counter_type == 'fc':
       shutil.copytree(os.path.join(ori_dataset_dir,train_data,cropped-images),os.path.join(ori_dataset_dir,args.counter_type,train_data,images))
       shutil.copytree(os.path.join(ori_dataset_dir,train_data,cropped-ground-truth),os.path.join(ori_dataset_dir,args.counter_type,train_data,ground-truth))
       shutil.copytree(os.path.join(ori_dataset_dir,test_data,cropped-ground-truth),os.path.join(ori_dataset_dir,args.counter_type,test_data,ground-truth))
       shutil.copytree(os.path.join(ori_dataset_dir,test_data,cropped-images),os.path.join(ori_dataset_dir,args.counter_type,test_data,images))
       args.crop_size = 640
       args.dataset_dir= os.path.join(ori_dataset_dir,args.counter_type)    
    else:
       raise NotImplementedError
    return args

    if args.dataset.lower() == 'qnrf':
        args.crop_size = 512
    elif args.dataset.lower() == 'nwpu':
        args.crop_size = 384
        args.val_epoch = 50
    elif args.dataset.lower() == 'sha':
        args.wot = 0.1
    elif args.dataset.lower() == 'shb':
        args.crop_size = 512
    else:
        raise NotImplementedError
    return args


if __name__ == '__main__':
    args = parse_args()
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu
    trainer = Trainer(args)
    trainer.setup()
    trainer.train()
