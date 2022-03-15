import numpy as np
import PIL
import os
import pandas as pd
from torch.utils.data import DataLoader
import torch
import random
from cv2 import equalizeHist
import torchvision.transforms as transforms
torch.cuda.manual_seed(0)
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'
class CXR_DataLoader(object):
  def __init__(self,data_path=None, label_sheet_path=None,dataloadertype='train_valid'):
    self.data_path=data_path
    if label_sheet_path != None:
        self.label_sheet=pd.read_csv(label_sheet_path)
    self.data_list=[]
    self.dataloadertype=dataloadertype
    self.generate_data_list()
    
  def generate_data_list(self):
    for image_name in os.listdir(self.data_path):
      current_image=PIL.Image.open(os.path.join(self.data_path,image_name))
      current_array=np.asarray(current_image.convert('L'))
      if self.dataloadertype=='train_valid':
        current_label=self.label_sheet.loc[self.label_sheet['FileName']==image_name,'Pathology'].iloc[0]
      elif self.dataloadertype=='covid':
        current_label=True
      elif self.dataloadertype=='normal':
        current_label=False
      else:
        raise ValueError('Invalid dataloader type. Choose train_valid, covid, or normal')
      self.data_list.append((image_name,current_array,current_label))
    np.random.shuffle(self.data_list)
  def __getitem__(self,idx):
    images = np.zeros((3,240,240))
    current_image=equalizeHist(self.data_list[idx][1])
    images[0,:,:]=current_image
    images[1,:,:]=current_image
    images[2,:,:]=current_image
    tensor_images=torch.from_numpy(images).type(torch.FloatTensor)
    CC=transforms.CenterCrop(200)
    tensor_images=CC(tensor_images)
    resize = transforms.Resize(size=(240,240))
    tensor_images=resize(tensor_images)
    return self.data_list[idx][0],torch.from_numpy(images).type(torch.FloatTensor),int(self.data_list[idx][2]) #np.random.randint(0, 2)
  def __len__(self):
    return len(self.data_list)
def MakeDataLoader(data_path,label_sheet_path=None,batch_size=128,dataloadertype='train_valid'):
  random.seed(0)
  torch.manual_seed(0)
  np.random.seed(0)
  torch.cuda.empty_cache()
  intermediateLoader=CXR_DataLoader(data_path,label_sheet_path,dataloadertype=dataloadertype)
  if dataloadertype=='train_valid':
    shuffle=True
  else:
    shuffle=False
  return DataLoader(intermediateLoader,batch_size=batch_size,shuffle=shuffle)

