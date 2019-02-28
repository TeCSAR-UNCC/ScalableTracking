from __future__ import print_function
import numpy as np
import random


import os.path
import datetime
import utils 

import random
import torch
import torch.nn as nn
import torch.optim as optim

'''     Device configuration      '''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

''' HYPERPARAMETERS'''
num_steps = 6
input_size = 54
hidden_size = 64
num_layers=1
num_classes = 4
batch_size = 1
epoches =  50
num_videos = 7  
path_to_data = sorted(os.listdir('../duke_dataset/testing/'))


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size,num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size,num_classes)

    def forward(self,input_x):

        #Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, input_x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers, input_x.size(0), self.hidden_size).to(device)

        #Forward propagate LSTM
        #out: tensor of shape (batch_size, seq_length, hidden_size)
        out, _ = self.lstm(input_x,(h0,c0))

        out = self.fc(out[:,-1,:])
        return out

model = RNN(input_size,hidden_size,num_layers,num_classes).to(device)
model.load_state_dict(torch.load('model1/model_epoch50_.ckpt'))

iters = num_videos * epoches

randlist=random.sample(range(0,len(path_to_data),2),1)

for i in (randlist):
    #i = i % num_videos
    #in ROLO utils file, the sequence for MOT16 starts with 30 Modification 3

    [w_img, h_img, sequence_name ]= 1920.0,1080.0,path_to_data[i]

    x_path = os.path.join('../duke_dataset/testing/', sequence_name) #Modification 4
    #x_bbox_path = os.path.join('own_images/testing/', sequence_name, 'bbox_video.npy')

    y_path = '../duke_dataset/testing/'+ sequence_name[:-4]+ '_gt.npy' ##Modification 5
    #print y_path

    filegt = np.load(y_path)
    filefeatures = np.load(x_path)

    training_iters = filefeatures.shape[0]                
    #filebboxes = np.load(x_bbox_path)

    print('Sequence '+ sequence_name+' chosen')
    id =0
    total_loss=0

    gtout=[]
    detectionout=[]
    while id  < training_iters- num_steps*batch_size:

        # Load testing data & ground truth
        batch_input = utils.load_openpose_features_train(w_img,h_img,filefeatures , batch_size, num_steps, id) # [num_of_examples, input_size] (depth == 1)

        batch_groundtruth = utils.load_openpose_gt_train(w_img,h_img,filegt, batch_size, num_steps, id)


        batch_input = np.reshape(batch_input, [batch_size, num_steps, input_size])

        batch_input = (torch.from_numpy(batch_input)).to(device)
        #print(batch_input, batch_input.shape)
        batch_groundtruth = np.reshape(batch_groundtruth, [batch_size, num_classes]) #2*4

        batch_groundtruth = (torch.from_numpy(batch_groundtruth)).to(device)
        #print(batch_groundtruth , batch_groundtruth.shape)        
        outputs = model(batch_input)
        #loss = criterion(outputs,batch_groundtruth)*100
        with torch.no_grad():
            batch_groundtruth = batch_groundtruth.cpu().numpy()
            outputs = outputs.cpu().numpy()

        tempgt= [(batch_groundtruth[0][0]-(batch_groundtruth[0][2]/2.0))*w_img,
                (batch_groundtruth[0][1]-(batch_groundtruth[0][3]/2.0))*h_img,
                batch_groundtruth[0][2]*w_img,
                batch_groundtruth[0][3]*h_img]

        tempout= [(outputs[0][0]-(outputs[0][2]/2.0))*w_img,
                (outputs[0][1]-(outputs[0][3]/2.0))*h_img,
                outputs[0][2]*w_img,
                outputs[0][3]*h_img]

        gtout.append(tempgt)
        detectionout.append(tempout)

        print('GT:',tempgt)
        print('Pred: ',tempout,'\n')
        #optimizer.zero_grad()
        #loss.backward()
        #optimizer.step()
        #print(id)
        #total_loss += loss.item()
        id = id +1
    print('Frame ID', id)
    np.save('test_results/'+sequence_name[:-4]+'_gt',np.asarray(gtout))
    np.save('test_results/'+sequence_name[:-4]+'_pred',np.asarray(detectionout))