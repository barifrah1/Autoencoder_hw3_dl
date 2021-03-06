import os
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
import pandas as pd
from numpy.random import rand
import random
import pickle as pickle
import numpy as np
num_epochs = 100
batch_size = 128
learning_rate = 1e-3


class AutoencoderPopular(nn.Module):
    def __init__(self, args=None):
        super(AutoencoderPopular, self).__init__()
        self.args = args
        self.encoder = nn.Sequential(
            nn.Linear(args.input_size,
                      args.hidden_size, bias=True),
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
    model.eval()
    popProb = dataloader.popularity_prob
    with torch.no_grad():
        for userVec in tqdm(dataloader):
            userPopProb = popProb.copy()
            userItems = dataloader.userSeenItems(index)
            for item in userItems:
                userPopProb[item] = 1
            userVec = torch.tensor(userVec).float()
            output = model(userVec)
            new_output = output
            if(len(validation[index]) == 0):
                index += 1
                counter += 1
                continue
            validationUserSeenItem = validation[index][0]
            # unseenPopularItemsList[index][0][epoch]
            prob = userPopProb.copy()
            for item in userItems:
                prob[item] = 0
            itemDrawn = random.choices(
                range(model.args.input_size), weights=prob, k=1)
            while(itemDrawn == validationUserSeenItem):
                itemDrawn = random.choices(
                    range(model.args.input_size), weights=prob, k=1)
            if(new_output[validationUserSeenItem].item() > new_output[itemDrawn].item()):
                accuracy += 1
            index += 1
    acc = accuracy/(dataloader.numOfUsers() - counter)
    return acc


def training_loop(args,
                  model,
                  tr_dataloader=None,
                  validation=None,
                  criterion_func=nn.MSELoss,
                  ):
    c = 0
    accuracy_by_epoch = []
    criterion = criterion_func()
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    popProb = tr_dataloader.popularity_prob

    for epoch in range(args.num_epochs):
        model.train()
        index = 0
        for userVec in tqdm(tr_dataloader):
            userItems = tr_dataloader.userSeenItems(index)
            userPopProb = popProb.copy()*args.popularity_multiplyer
            for item in userItems:
                userPopProb[item] = 1
            rand_vec = rand(args.input_size)
            mask = np.zeros(args.input_size)
            for j in range(args.input_size):
                if(rand_vec[j] < userPopProb[j]):
                    mask[j] = 1
                    c += 1
            # print(c)
            c = 0
            userVec = torch.tensor(userVec).float()
            # ===================forward=====================
            output = model(userVec)
            new_output = output*torch.tensor(mask)
            loss = criterion(new_output, userVec.double())
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            index += 1
        # ===================log========================
        if epoch % 1 == 0:
            currentAccuracy = infer(
                tr_dataloader, validation, model)
            accuracy_by_epoch.append(currentAccuracy)
            print(
                f" epoch: { epoch+1} tr_loss: {loss} validation accuracy: {currentAccuracy} ")
            if(epoch % 8 == 0 and epoch != 0):
                args.lr = args.lr*0.1
        if currentAccuracy > 0.87:
            break
    predict = pd.read_csv("PopularityTest.csv")
    data = predict.values
    for x in data:
        user = x[0]-1
        item1 = x[1]-1
        item2 = x[2]-1
        uservector = tr_dataloader.__getitem__(user)
        uservector = torch.tensor(uservector).float()
        with torch.no_grad():
            output = model(uservector)
            if output[item1] >= output[item2]:
                x[3] = 0
            if output[item1] < output[item2]:
                x[3] = 1
    df = pd.DataFrame(
        data, columns=['UserID', 'Item1', 'Item2', 'bitClassification'])
    df.to_csv(r'popularity_205592652_312425036.csv', index=False)
    return accuracy_by_epoch
