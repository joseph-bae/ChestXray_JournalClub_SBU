import numpy as np
import PIL
import os
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import models
from torch import optim
import utils
import torch
class CXR_Model(object):

    def __init__(self,model_type='small',dropout=.3 ,train_loader=None,
        valid_loader=None,test_loader=None):
        self.model_type=model_type
        self.p=dropout
        self.device=torch.device("cuda:0")
        self.create_model()
        self.train_loader=train_loader
        self.valid_loader=valid_loader
        self.test_loader=test_loader
        self.latest_labels=None
        self.latest_predictions=None
        self.latest_images=None

    def create_model(self):
        if self.model_type=='small':
            model=models.resnet18(pretrained=True)
            model.fc=torch.nn.Sequential(
                torch.nn.Dropout(p=self.p),
                torch.nn.Linear(512,2),
                torch.nn.Softmax(dim=1)
                    )
        if self.model_type=='medium':
            model=models.resnet34(pretrained=True)
            model.fc=torch.nn.Sequential(
                torch.nn.Dropout(p=self.p),
                torch.nn.Linear(1000,2),
                torch.nn.Softmax(dim=1)
                    )
        if self.model_type=='large':
            model=models.resnet50(pretrained=True)
            model.fc=torch.nn.Sequential(
                torch.nn.Dropout(p=self.p),
                torch.nn.Linear(1000,2),
                torch.nn.Softmax(dim=1)
                    )
        self.model=model
        return
    def train(self,epochs=20,learning_rate=.01):

        device=self.device
        model=self.model.to(device)
        criterion = torch.nn.CrossEntropyLoss().to(device)
        optimizer = optim.SGD(model.parameters(),lr=learning_rate)
        for epoch in range(epochs):
            train_count=0
            epoch_loss=0
            epoch_correct=0
            for batch in self.train_loader:
                model.train()
                optimizer.zero_grad()
                names,images,label=batch
                images=images.to(device)
                label=label.to(device)
                output=model(images)
                loss=criterion(output,label)
                loss.backward()
                optimizer.step()
                epoch_loss+=loss.item()*images.shape[0]
                train_count+=images.shape[0]
                correct = np.where(np.argmax(output.cpu().detach().numpy(), axis=1)==label.cpu().detach().numpy())[0].shape[0]
                epoch_correct+=correct
            accuracy=epoch_correct/train_count
            total_loss=epoch_loss/train_count 
            print("epoch:",epoch+1,'\n')
            print("---------------TRAIN---------------\ntrain_loss:",total_loss, "train_accuracy:",accuracy)
            

            valid_count=0
            valid_epoch_loss=0
            valid_epoch_correct=0
            
            for vbatch in self.valid_loader:
                model.eval()
                optimizer.zero_grad()
                with torch.no_grad():

                    names,images,label=vbatch
                    images=images.to(device)
                    label=label.to(device)
                    output=model(images)
                    loss=criterion(output,label)
                    valid_epoch_loss+=loss.item()*images.shape[0]
                    valid_count+=images.shape[0]
                    correct = np.where(np.argmax(output.cpu().detach().numpy(), axis=1)==label.cpu().detach().numpy())[0].shape[0]
                    valid_epoch_correct+=correct
            self.latest_labels=label
            self.latest_predictions=output
            self.latest_images=images
            valid_accuracy=valid_epoch_correct/valid_count
            valid_total_loss=valid_epoch_loss/valid_count
            print("---------------VALID---------------\nvalid_loss:",valid_total_loss,"valid_accuracy:",valid_accuracy)
    def test(self):
        test_count=0
        test_epoch_loss=0
        test_epoch_correct=0
        device=self.device
        for batch in self.test_loader:
            model.eval()
            optimizer.zero_grad()
            with torch.no_grad():
            
                names,images,label=batch

                images=images.to(device)
                label=label.to(device)
                output=model(images)
                loss=criterion(output,label)
                optimizer.step()
                test_epoch_loss+=loss.item()*images.shape[0]
                test_count+=images.shape[0]
                correct = np.where(np.argmax(output.cpu().detach().numpy(), axis=1)==label.cpu().detach().numpy())[0].shape[0]
                test_epoch_correct+=correct
        self.latest_labels=label
        self.latest_predictions=output
        self.latest_images=images
        test_accuracy=test_epoch_correct/test_count
        test_total_loss=test_epoch_loss/test_count
        print("test_loss:",test_total_loss, "test_accuracy:",test_accuracy)
