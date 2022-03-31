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
import matplotlib.pyplot as plt 
from IPython.display import clear_output
from google.colab.patches import cv2_imshow
from cv2 import imread
torch.cuda.manual_seed(0)
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True

class CXR_Model(object):
    torch.cuda.empty_cache()
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
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
        self.latest_names=None
        self.validation_list=None
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
        fig, axs = plt.subplots(1, 4, figsize=(20,5))
        device=self.device
        self.model.to(device)
        loss_list_validation=[]
        loss_list_train=[]
        train_sens_list=[]
        valid_sens_list=[]
        train_spec_list=[]
        valid_spec_list=[]
        train_acc_list=[]
        valid_acc_list=[]
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
            train_sens_list.append(sensitivity)
            specificity=epoch_correct0/(train_count/2)
            train_spec_list.append(specificity)
            accuracy=epoch_correct/train_count
            train_acc_list.append(accuracy)
            total_loss=epoch_loss/train_count 
            loss_list_train.append(total_loss)
            # print(("---------------TRAIN---------------\n"+"train_sensitvity: %.4f, train_specificity: %.4f, " + utils.color.RED+utils.color.BOLD+"train_accuracy: %.4f"+utils.color.END) %(sensitivity,specificity,accuracy))
            

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
            self.latest_names=names
            valid_sensitivity=valid_epoch_correct1/(valid_count/2)
            valid_sens_list.append(valid_sensitivity)

            
            valid_specificity=valid_epoch_correct0/(valid_count/2)
            valid_spec_list.append(valid_specificity)
            valid_accuracy=valid_epoch_correct/valid_count
            valid_acc_list.append(valid_accuracy)
            valid_total_loss=valid_epoch_loss/valid_count
            loss_list_validation.append(valid_total_loss)
            self.validation_list=loss_list_validation
            
            fig.suptitle("Epoch "+str(epoch+1),fontsize=20)
            
            axs[0].plot([x+1 for x in range(epoch+1)],loss_list_train,'bo',linestyle='dashed',label='train')
            axs[0].plot([x+1 for x in range(epoch+1)],loss_list_validation,'r+',linestyle='solid',label='valid')
            axs[0].set(xlabel="Epochs",ylabel="Loss",title="Loss")

            axs[1].plot([x+1 for x in range(epoch+1)],train_sens_list,'bo',linestyle='dashed',label='train')
            axs[1].plot([x+1 for x in range(epoch+1)],valid_sens_list,'r+',linestyle='solid',label='valid')
            axs[1].set(xlabel="Epochs",ylabel="Sensitivity",title="Sensitivity")
            axs[1].set_ylim([0,1])
            
            axs[2].plot([x+1 for x in range(epoch+1)],train_spec_list,'bo',linestyle='dashed',label='train')
            axs[2].plot([x+1 for x in range(epoch+1)],valid_spec_list,'r+',linestyle='solid',label='valid')
            axs[2].set(xlabel="Epochs",ylabel="Specificity",title="Specificity")
            axs[2].set_ylim([0,1])
            
            axs[3].plot([x+1 for x in range(epoch+1)],train_acc_list,'bo',linestyle='dashed',label='train')
            axs[3].plot([x+1 for x in range(epoch+1)],valid_acc_list,'r+',linestyle='solid',label='valid')
            axs[3].set(xlabel="Epochs",ylabel="Accuracy",title="Accuracy")            
            axs[3].set_ylim([0,1])
            if epoch == 0:
                handles, labels = axs[1].get_legend_handles_labels()
                fig.legend(handles, labels, loc='upper right') 
            ### The following is a painful fix, but seems necessary for Google Colab for demo purposes

            fig.savefig("temp.png")
            img=imread("temp.png")
            clear_output()
            cv2_imshow(img)
        clear_output()
        print(("train_loss: %.8f \n"+"train_sensitvity: %.4f, train_specificity: %.4f, "+utils.color.BOLD+"train_accuracy: %.4f"+utils.color.END+'\n') %(total_loss,sensitivity,specificity,accuracy))
        print(("valid_loss: %.8f \n"+"valid_sensitvity: %.4f, valid_specificity: %.4f, "+utils.color.RED+utils.color.BOLD+"valid_accuracy: %.4f"+utils.color.END+'\n') %(valid_total_loss,valid_sensitivity,valid_specificity,valid_accuracy))

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
        self.latest_names=names
        test_accuracy=test_epoch_correct/test_count
        test_total_loss=test_epoch_loss/test_count
        print("test_accuracy:",test_accuracy)
