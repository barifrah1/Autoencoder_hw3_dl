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


def infer(dataloader, validation, model, epoch=None):
    accuracy = 0
    counter = 0
    index = 0
    comapre_acc = 0
    model.eval()
    popularity_priviews = pd.read_csv(
        "popularity_priviews.csv", sep=',', header=0)
    #pickle_in = open("PickleOfPopularUnseenListsValidation", "rb")
    #unseenPopularItemsList = pickle.load(pickle_in)
    popProb = dataloader.popularity_prob
    with torch.no_grad():
        for userVec in tqdm(dataloader):
            userPopProb = popProb.copy()
            userItems = dataloader.userSeenItems(index)
            for item in userItems:
                userPopProb[item] = 1
            userVec = torch.tensor(userVec).float()
            output = model(userVec)
            new_output = output  # *torch.tensor(userPopProb).double()
            # compare to recsys
            compare_user = popularity_priviews[popularity_priviews['UserID'] == index+1].to_numpy()[
                0]
            item1 = compare_user[1]
            item2 = compare_user[2]
            score = compare_user[3]
            if score == 0:
                if (new_output[item1-1].item() > new_output[item2-1].item()):
                    comapre_acc += 1
            if score == 1:
                if (new_output[item1-1].item() < new_output[item2-1].item()):
                    comapre_acc += 1

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
            """while(itemDrawn == validationUserSeenItem):
                itemDrawn = choice(dataloader.sample_popular[index])"""
            # print(output.shape, validationUserSeenItem, itemDrawn,
            #      output[validationUserSeenItem].item(), output[itemDrawn].item())
            if(new_output[validationUserSeenItem].item() > new_output[itemDrawn].item()):
                accuracy += 1
            index += 1
    acc = accuracy/(dataloader.numOfUsers() - counter)
    comapre_Accurecy = comapre_acc/6040
    return acc, comapre_Accurecy


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
    """pickle_in = open("PickleOfPopularUnseenLists", "rb")
    unseenPopularItemsList = pickle.load(pickle_in)"""
    popProb = tr_dataloader.popularity_prob

    for epoch in range(args.num_epochs):
        model.train()
        index = 0
        for userVec in tqdm(tr_dataloader):
            userItems = tr_dataloader.userSeenItems(index)
            userPopProb = popProb.copy()*200
            for item in userItems:
                userPopProb[item] = 1
            rand_vec = rand(args.input_size)
            mask = np.zeros(args.input_size)
            for j in range(args.input_size):
                if(rand_vec[j] < userPopProb[j]):
                    mask[j] = 1
            #currenUserPopularItemList = unseenPopularItemsList[index]
            userVec = torch.tensor(userVec).float()
            # ==============Noise Adding======================
            #noisyuservecter = userVec+torch.randn(userVec.shape[0])
            # print(noisyuservecter.shape)
            # ===================forward=====================
            output = model(userVec)
            new_output = output*torch.tensor(mask)
            """currenUserPopularItemListCurrentEpoch = currenUserPopularItemList[epoch][:100]
            for item in currenUserPopularItemListCurrentEpoch:
                new_output[item] = output[item]"""
            loss = criterion(new_output, userVec.double())
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            index += 1
        # ===================log========================
        if epoch % 1 == 0:
            currentAccuracy, comapre_Acc = infer(
                tr_dataloader, validation, model, epoch=epoch)
            accuracy_by_epoch.append(currentAccuracy)
            print(
                f" epoch: { epoch+1} validation accuracy: {currentAccuracy} comapre acc:{comapre_Acc}")

    return accuracy_by_epoch
