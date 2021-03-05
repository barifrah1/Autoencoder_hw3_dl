import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from numpy.random import choice
import pandas as pd
from collections import Counter
import random as random
import pickle
from tqdm import tqdm
USER_IND = 0
ITEM_IND = 1
INPUT_SIZE = 3706


def initialProcessData(path):
    data = pd.read_csv(
        path, sep=',', header=0).to_numpy()
    train = {}
    popularity = {}
    for i in range(INPUT_SIZE):
        popularity[i] = 0
    # create training data
    for row in data:
        if row[USER_IND]-1 not in train.keys():
            train[row[USER_IND]-1] = []
        train[row[USER_IND]-1].append(row[ITEM_IND]-1)
        popularity[row[ITEM_IND]-1] += 1
    validation = {}
    # create validation data
    for user in train.keys():
        if(len(train[user]) > 1):
            validation_item = random.choice(train[user])
            train[user].remove(validation_item)
            validation[user] = [validation_item]
    return train, validation, popularity


class DataLoader_RecSys(Dataset):
    def __init__(self, dataset, popularity):
        self.dataset = dataset
        self.popularity = popularity
        self.popularity_prob = np.array(
            list(self.popularity.values()))/sum(np.array(list(self.popularity.values())))
        self.users = list(self.dataset.keys())
        self.items = []
        for user in self.users:
            self.items = self.items + self.dataset[user]
        self.items = Counter(self.items)
        self.max_item_index = max(self.items)
        self.max_user_index = max(self.users)

    def nextitem(self, user, ind):
        return self.dataset[user][ind]

    def userSeenItems(self, user):
        return self.dataset[user]

    def userBinaryVector(self, user):
        userVector = np.zeros(self.max_item_index + 1)
        userItems = self.userSeenItems(user)
        for item in userItems:
            userVector[item] = 1
        return userVector

    def userUnseenItems(self, user):
        return list(set(self.items).difference(set(self.userSeenItems(user))))

    def numOfUsers(self):
        return self.max_user_index + 1

    def numOfItems(self):
        return self.max_item_index + 1

    def drawUnseenItem(self, user):
        return random.choice(self.userUnseenItems(user))

    """def drawPopularunseen(self, user):
        Unseen = self.userUnseenItems(user)
        a_subset = {key: value for key,
                    value in self.popularity.items() if key in Unseen}
        prob = list(a_subset.values())
        normprob = [number/sum(prob) for number in prob]
        UnseenItem = np.random.choice(Unseen, 100, p=normprob)
        return UnseenItem"""

    def __getitem__(self, ind):
        if(ind >= self.__len__()):
            raise IndexError
        userVec = self.userBinaryVector(ind)
        return userVec

    def __len__(self):
        return self.max_user_index + 1
