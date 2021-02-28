import os
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
import pandas as pd
from numpy.random import choice
import random

num_epochs = 100
batch_size = 128
learning_rate = 1e-3


class Autoencoder(nn.Module):
    def __init__(self, args=None):
        super(Autoencoder, self).__init__()
        self.args = args
        self.encoder = nn.Sequential(nn.Dropout(0.5),
            nn.Linear(args.input_size, args.hidden_size, bias=True),
            nn.Sigmoid())
        
        self.decoder = nn.Sequential(
            nn.Linear(args.hidden_size, args.input_size, bias=True),
            nn.Sigmoid())

    def forward(self, x):
        #x = torch.tensor(x).float()
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def infer(dataloader, validation, model):
    accuracy = 0
    counter = 0
    index = 0
    comapre_acc=0
    model.eval()
    popularity_priviews = pd.read_csv("popularity_priviews.csv", sep=',', header=0)
    with torch.no_grad():
        for userVec in dataloader:
            userVec = torch.tensor(userVec).float()
            output = model(userVec)
            if(len(validation[index]) == 0):
                counter += 1
                continue
            validationUserSeenItem = validation[index][0]
            itemDrawn = choice(dataloader.sample_popular[index])
            while(itemDrawn == validationUserSeenItem):
                itemDrawn =choice(dataloader.sample_popular[index])
            #print(output.shape, validationUserSeenItem, itemDrawn,
            #      output[validationUserSeenItem].item(), output[itemDrawn].item())
            if(output[validationUserSeenItem].item() > output[itemDrawn].item()):
                accuracy += 1
            compare_user=popularity_priviews[popularity_priviews['UserID']==index+1].to_numpy()[0]
            item1=compare_user[1]
            item2=compare_user[2]
            score=compare_user[3]
            if score==0:
                if (output[item1-1].item() > output[item2-1].item()):
                    comapre_acc+=1
            if score==1:
                if (output[item1-1].item() < output[item2-1].item()):
                    comapre_acc+=1
            index += 1
    acc=accuracy/(dataloader.numOfUsers() - counter)
    comapre_Accurecy=comapre_acc/6040
    return acc,comapre_Accurecy


def training_loop(args,
                  model,
                  tr_dataloader=None,
                  validation=None,
                  criterion_func=nn.MSELoss,
                  ):
    accuracy_by_epoch = []
    criterion = criterion_func()
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)

    for epoch in range(args.num_epochs):
        model.train()
        for userVec in tqdm(tr_dataloader):
            userVec = torch.tensor(userVec).float()
            #==============Noise Adding======================
            noisyuservecter=userVec+torch.randn(userVec.shape[0]) 
            #print(noisyuservecter.shape)
            # ===================forward=====================
            output = model(noisyuservecter)
            loss = criterion(output, userVec)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        if epoch % 1 == 0:
            currentAccuracy,comapre_Acc = infer(tr_dataloader, validation, model)
            accuracy_by_epoch.append(currentAccuracy)
            print(f" epoch: { epoch+1} validation accuracy: {currentAccuracy} comapre acc:{comapre_Acc}")

    return accuracy_by_epoch


