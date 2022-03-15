import numpy as np
import PIL
import os
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import models
from torch import optim
import utils
import torch
import random
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

class CXR_Model(object):

    def __init__(self,model_type='small', learning_rate=.01, dropout=.3,train_loader=None,
        valid_loader=None,test_loader=None):
        self.model_type=model_type
        self.p=dropout
        self.lr=learning_rate
        self.device=torch.device("cuda:0")
        self.create_model()
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(),lr=self.lr)
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
                torch.nn.Linear(512,2),
                torch.nn.Softmax(dim=1)
                    )
        if self.model_type=='large':
            model=models.resnet50(pretrained=True)
            model.fc=torch.nn.Sequential(
                torch.nn.Dropout(p=self.p),
                torch.nn.Linear(2048,2),
                torch.nn.Softmax(dim=1)
                    )
        self.model=model
        return
    def train(self,epochs=10):

        device=self.device
        self.model.to(device)

        for epoch in range(epochs):
            train_count=0
            epoch_loss=0
            epoch_correct=0
            epoch_correct0=0
            epoch_correct1=0
            for batch in self.train_loader:
                self.model.train()
                self.optimizer.zero_grad()
                names,images,label=batch
                images=images.to(device)
                label=label.to(device)
                output=self.model(images)
                loss=self.criterion(output,label)
                loss.backward()
                self.optimizer.step()
                epoch_loss+=loss.item()*images.shape[0]
                train_count+=images.shape[0]
                correct = np.where(np.argmax(output.cpu().detach().numpy(), axis=1)==label.cpu().detach().numpy())[0].shape[0]
                negatives = np.where(label.cpu().detach().numpy()==0)[0]
                correct0 = np.where(np.argmax(output.cpu().detach().numpy()[negatives],axis=1)==0)[0].shape[0]
                positives = np.where(label.cpu().detach().numpy()==1)[0]
                correct1 = np.where(np.argmax(output.cpu().detach().numpy()[positives],axis=1)==1)[0].shape[0]
                epoch_correct0+=correct0
                epoch_correct1+=correct1
                epoch_correct+=correct
            sensitivity=epoch_correct1/(train_count/2)
            specificity=epoch_correct0/(train_count/2)
            accuracy=epoch_correct/train_count
            total_loss=epoch_loss/train_count 
            print("epoch:",epoch+1)
            print(("---------------TRAIN---------------\n"+"train_sensitvity: %.4f, train_specificity: %.4f, " + utils.color.RED+utils.color.BOLD+"train_accuracy: %.4f"+utils.color.END) %(sensitivity,specificity,accuracy))
            

            valid_count=0
            valid_epoch_loss=0
            valid_epoch_correct=0
            valid_epoch_correct0=0
            valid_epoch_correct1=0            
            for vbatch in self.valid_loader:
                self.model.eval()
                self.optimizer.zero_grad()
                with torch.no_grad():

                    names,images,label=vbatch
                    images=images.to(device)
                    label=label.to(device)
                    output=self.model(images)
                    loss=self.criterion(output,label)
                    valid_epoch_loss+=loss.item()*images.shape[0]
                    valid_count+=images.shape[0]
                    correct = np.where(np.argmax(output.cpu().detach().numpy(), axis=1)==label.cpu().detach().numpy())[0].shape[0]
                    negatives = np.where(label.cpu().detach().numpy()==0)[0]
                    correct0 = np.where(np.argmax(output.cpu().detach().numpy()[negatives],axis=1)==0)[0].shape[0]
                    positives = np.where(label.cpu().detach().numpy()==1)[0]
                    correct1 = np.where(np.argmax(output.cpu().detach().numpy()[positives],axis=1)==1)[0].shape[0]                    
                    valid_epoch_correct0+=correct0
                    valid_epoch_correct1+=correct1
                    valid_epoch_correct+=correct
            self.latest_labels=label
            self.latest_predictions=output
            self.latest_images=images
            valid_sensitivity=valid_epoch_correct1/(valid_count/2)
            valid_specificity=valid_epoch_correct0/(valid_count/2)
            valid_accuracy=valid_epoch_correct/valid_count
            valid_total_loss=valid_epoch_loss/valid_count
            print(("---------------VALID---------------\n"+"valid_sensitvity: %.4f, valid_specificity: %.4f, "+utils.color.RED+utils.color.BOLD+"valid_accuracy: %.4f"+utils.color.END+'\n') %(valid_sensitivity,valid_specificity,valid_accuracy))
    def test(self):
        test_count=0
        test_epoch_loss=0
        test_epoch_correct=0
        device=self.device
        for batch in self.test_loader:
            self.model.eval()
            self.optimizer.zero_grad()
            with torch.no_grad():
            
                names,images,label=batch

                images=images.to(device)
                label=label.to(device)
                output=self.model(images)
                loss=self.criterion(output,label)
                test_epoch_loss+=loss.item()*images.shape[0]
                test_count+=images.shape[0]
                correct = np.where(np.argmax(output.cpu().detach().numpy(), axis=1)==label.cpu().detach().numpy())[0].shape[0]
                test_epoch_correct+=correct
        self.latest_labels=label
        self.latest_predictions=output
        self.latest_images=images
        test_accuracy=test_epoch_correct/test_count
        test_total_loss=test_epoch_loss/test_count
        print("test_accuracy:",test_accuracy)
