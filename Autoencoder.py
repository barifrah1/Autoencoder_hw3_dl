import os
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
import pandas as pd
import numpy as np


class Autoencoder(nn.Module):
    def __init__(self, args=None):
        super(Autoencoder, self).__init__()
        self.args = args
        self.encoder = nn.Sequential(nn.Dropout(0.5),
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
    with torch.no_grad():
        for userVec in dataloader:
            userVec = torch.tensor(userVec).float()
            output = model(userVec)
            if(len(validation[index]) == 0):
                index += 1
                counter += 1
                continue
            validationUserSeenItem = validation[index][0]
            itemDrawn = dataloader.drawUnseenItem(index)
            while(itemDrawn == validationUserSeenItem):
                itemDrawn = dataloader.drawUnseenItem(index)
            if(output[validationUserSeenItem].item() > output[itemDrawn].item()):
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
    accuracy_by_epoch = []
    criterion = criterion_func()
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)

    for epoch in range(args.num_epochs):
        model.train()
        for userVec in tqdm(tr_dataloader):
            userVec = torch.tensor(userVec).float()
            # ===================forward=====================
            output = model(userVec)
            loss = criterion(output, userVec)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        if epoch % 1 == 0:
            currentAccuracy = infer(
                tr_dataloader, validation, model)
            accuracy_by_epoch.append(currentAccuracy)
            print(
                f" epoch: { epoch+1} validation accuracy: {currentAccuracy}")
            if currentAccuracy > 0.932:
                break
    predict = pd.read_csv("RandomTest.csv")
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
