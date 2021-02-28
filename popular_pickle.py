import numpy as np
import pandas as pd
import random
import pickle

USER_IND = 0
ITEM_IND = 1


data = pd.read_csv("Train.csv", sep=',', header=0).to_numpy()

popularity={}
train={}
for row in data:
    if row[USER_IND]-1 not in train.keys():
        train[row[USER_IND]-1] = []
    train[row[USER_IND]-1].append(row[ITEM_IND]-1)
    if row[ITEM_IND]-1 not in popularity.keys():
        popularity[row[ITEM_IND]-1]=0
    popularity[row[ITEM_IND]-1]+=1

users={}
for x in train.keys():
    s=0
    for y in x:
        s+=popularity[y]
    train[x]=[x,[number/s for number in x]]
print(train[2])


#sample_dict={}
#for x in popularity.keys()
#for samp in range(100):





