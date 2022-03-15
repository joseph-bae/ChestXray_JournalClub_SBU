import numpy as np
import PIL
import os
import pandas as pd
from torch.utils.data import DataLoader
class CXR_DataLoader(object):
  def __init__(self,data_path=None, label_sheet_path=None,dataloadertype='train_valid'):
    self.data_path=data_path
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
      elif self.dataloadertype=='SBU':
        current_label=True
      elif self.dataloadertype=='Normal':
        current_label=False
      else:
        raise ValueError('Invalid dataloader type. Choose train_valid, SBU, or normal')
      self.data_list.append((image_name,current_array,current_label))
    np.random.shuffle(self.data_list)
  def __getitem__(self,idx):
    images = np.zeros((3,240,240))
    images[0,:,:]=self.data_list[idx][1]
    images[1,:,:]=self.data_list[idx][1]
    images[2,:,:]=self.data_list[idx][1]
    return self.data_list[idx][0],torch.from_numpy(images).type(torch.FloatTensor),int(self.data_list[idx][2]) #np.random.randint(0, 2)
  def __len__(self):
    return len(self.data_list)
def MakeDataLoader(data_path,label_sheet_path,batch_size=128,shuffle=True,dataloadertype='train_valid'):
  intermediateLoader=CXRDataLoader(data_path,label_sheet_path,dataloadertype=dataloadertype)
  return DataLoader(intermediateLoader,batch_size=batch_size,shuffle=shuffle)

