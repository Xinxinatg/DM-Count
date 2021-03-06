import os
import torch
import torch.utils.data as torchdata
import torch.nn as nn
import PIL.Image as Image
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import torch.backends.cudnn as cudnn
import cv2
import glob
import numpy
import matplotlib.pyplot as plt
import argparse
import torch
import os
import scipy.io as io
import numpy as np
import datasets.crowd as crowd
from torchvision import transforms
from models import vgg19
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda')
root = '/content/content/content/VisDrone2020-CC'
part_B_train = os.path.join(root,'train_data','images')
part_B_test = os.path.join(root,'test_data','downsampled-padded-images')
model_path = '/content/DMcouting_coarse_best_model_0.pth'
model = vgg19()
model.to(device)
model.load_state_dict(torch.load(model_path, device))
transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ])
path_sets_B = [part_B_test]
img_paths_B = []
for path in path_sets_B:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths_B.append(img_path)
number=0
image_errs_temp=[]
os.mkdir('/content/content/content/VisDrone2020-CC/test_data/base_dir_metric_cd')
for img_path in tqdm(img_paths_B):
#for k in xrange(len(img_paths_B)):
    for i in range (0,3):
       for j in range (0,3):
          image_path=img_path.replace('downsampled-padded-images','images').replace('.jpg','_{}_{}.jpg'.format(i,j))
            
          mat_path=image_path.replace('.jpg', '.mat').replace('images', 'ground-truth').replace('IMG_', 'GT_IMG_')
          mat = io.loadmat(mat_path)
#          dataloader = torch.utils.data.DataLoader('sha', 1, shuffle=False,num_workers=1, pin_memory=True)
          image_errs = []
          img = transform(Image.open(image_path).convert('RGB')).cuda()
          inputs = img.unsqueeze(0)
          
          #assert inputs.size(0) == 1, 'the batch size should equal to 1'
          with torch.set_grad_enabled(False):
              outputs, _ = model(inputs)
          img_err = abs(mat["image_info"][0,0][0,0][1] - torch.sum(outputs).item())
          img_err=np.squeeze(img_err)
          print(image_path, img_err)
          image_errs_temp.append(img_err)
     
    image_errs = np.reshape(image_errs_temp,(3,3))
    
    with open(img_path.replace('downsampled-padded-images','base_dir_metric_cd').replace('.jpg','.npy'), 'wb') as f:
     np.save(f, image_errs)
    image_errs_temp.clear()
