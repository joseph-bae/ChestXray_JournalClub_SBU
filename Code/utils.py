import numpy as np
import PIL
import os
import pandas as pd
from torch.utils.data import DataLoader
import torch
import random
from cv2 import equalizeHist
import matplotlib.pyplot as plt #Library for image and figure visualization
from collections import defaultdict

def ShowImages_Interactive(images,num_NoFinding=5,num_Pneum=5):
  Key=pd.read_csv("/content/Temp_JC/ChestXray_JournalClub_SBU/Key.csv")
  TotalIms=np.ceil(num_NoFinding+num_Pneum)
  Outcome_d=defaultdict(list)
  for png in images:
    Outcome=Key.loc[Key['FileName']==png,'Pathology'].values[0]
    if Outcome:
      Outcome_d['Consolidation/Pneumonia'].append(png)
    else:
      Outcome_d['No Finding'].append(png)
  
  Chosen_Pneum=np.random.choice(Outcome_d['Consolidation/Pneumonia'],
      num_Pneum,replace=False)
  Chosen_Norm=np.random.choice(Outcome_d['No Finding'],
      num_NoFinding,replace=False) 
  TotalChosenImages=np.concatenate((Chosen_Pneum,Chosen_Norm))
  plt.figure(figsize=(100,100)) #open a figure for viewing
  for i,Image_Name in enumerate(TotalChosenImages):  #loop through our chosen random images
    CurrentImage=PIL.Image.open(os.path.join("/content/Temp_JC/ChestXray_JournalClub_SBU/TrainingCXRs",Image_Name)) #read in a single image
    CurrentLabel=Key.loc[Key['FileName']==Image_Name,'Pathology'] #figure out it's label
    ax=plt.subplot(4,np.ceil(TotalIms/4),i+1) #boring figure stuff
    if CurrentLabel.values[0]:
      ax.set_title("Consolidation/Pneumonia",fontsize=72)

    else:
      ax.set_title("No Finding",fontsize=72)
    # ax.text(120,220,CurrentLabel.values[0],color='white',backgroundcolor='black',
    # horizontalalignment='center')
    ax.axis('off')
    plt.imshow(CurrentImage,cmap='gray')
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
def ShowImages(images):
  Key=pd.read_csv("/content/Temp_JC/ChestXray_JournalClub_SBU/Key.csv")
  Random_Image_Names=np.random.choice(images,10,replace=False) #choose 10 random images from this training set
  plt.figure(figsize=(20,10)) #open a figure for viewing
  for i,Image_Name in enumerate(Random_Image_Names):  #loop through our chosen random images
    CurrentImage=PIL.Image.open(os.path.join("/content/Temp_JC/ChestXray_JournalClub_SBU/TrainingCXRs",Image_Name)) #read in a single image
    CurrentLabel=Key.loc[Key['FileName']==Image_Name,'Pathology'] #figure out it's label
    ax=plt.subplot(2,5,i+1) #boring figure stuff
    if CurrentLabel.values[0]:
      ax.set_title("Consolidation/Pneumonia")
    else:
      ax.set_title("No Finding")
    # ax.text(120,220,CurrentLabel.values[0],color='white',backgroundcolor='black',
    # horizontalalignment='center')
    ax.axis('off')
    plt.imshow(CurrentImage,cmap='gray') 
