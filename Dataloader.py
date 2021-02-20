import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from collections import Counter
import random
USER_IND = 0
ITEM_IND = 1


def initialProcessData(path):
    data = pd.read_csv(
        path, sep=',', header=0).to_numpy()
    train = {}
    # create training data
    for row in data:
        if row[USER_IND] not in train.keys():
            train[row[USER_IND]] = []
        train[row[USER_IND]].append(row[ITEM_IND])
    validation = {}
    # create validation data
    for user in train.keys():
        if(len(train[user]) > 1):
            validation_item = train[user].pop()
            validation[user] = [validation_item]
    return train, validation


class DataLoader_RecSys(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.users = list(self.dataset)
        self.items = []
        for user in self.users:
            self.items = self.items + self.dataset[user]
        self.items = Counter(self.items)

    def nextitem(self, user, ind):
        return self.dataset[user][ind]

    def userSeenItems(self, user):
        return self.dataset[user]

    def userUnseenItems(self, user):
        return set(self.items).difference(set(self.userSeenItems(user)))

    def __getitem__(self, ind):
        return False

    def __len__(self):
        return sum(len(self.dataset[user]) for user in self.dataset.keys())
