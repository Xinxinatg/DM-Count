import os
import torch
import torch.utils.data as torchdata
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
import torch.optim as optim
import torch.backends.cudnn as cudnn
import cv2
import glob
import numpy
import matplotlib.pyplot as plt
import argparse
import torch
import os
import numpy as np
import datasets.crowd as crowd
from models import vgg19

root = '/content/VisDrone2020-CC/'
part_B_train = os.path.join(root,'train_data','images')
part_B_test = os.path.join(root,'test_data','images')
path_sets_B = [part_B_test]
img_paths_B = []
for path in path_sets_B:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths_B.append(img_path)
number=0
for img_path in tqdm(img_paths_B):
    for i in range (0,3):
       for j in range (0,3):
          image_path=img_path.replace('images','cropped-images').replace('.jpg','_{}_{}.jpg'.format(i,j))
          dataloader = torch.utils.data.DataLoader('VisDrone', 1, shuffle=False,num_workers=1, pin_memory=True)
          model_path = 'pretrained_models/model_qnrf.pth'
          model = vgg19()
          model.to('0')
          model.load_state_dict(torch.load(model_path, '0'))
          image_errs = []
          
          inputs = cv2.imread(image_path,mode='RGB')
          #assert inputs.size(0) == 1, 'the batch size should equal to 1'
          with torch.set_grad_enabled(False):
              outputs, _ = model(inputs)
          img_err = count[0].item() - torch.sum(outputs).item()

          print(name, img_err, count[0].item(), torch.sum(outputs).item())
          image_errs_temp.append(img_err)
     
 image_errs = np.reshape(image_errs_temp,(3,3))
 with open(img_path.replace('images','base_dir_metric_fd').replace('.jpg','.npy'), 'wb') as f:
  np.save(f, image_errs)
 image_errs_temp.clear()
