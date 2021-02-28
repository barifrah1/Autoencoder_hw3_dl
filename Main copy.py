
import Dataloader as dataloader
from Dataloader import DataLoader_RecSys
#from Autoencoder import Autoencoder, training_loop
from Consts import *
from torch import nn


if __name__ == "__main__":

    d1, validation_data,popularity = dataloader.initialProcessData(TRAIN_DATA_PATH)
    train_dataloader = DataLoader_RecSys(d1,popularity)
    print(train_dataloader.max_user_index)
    sample={}
    for x in range(train_dataloader.max_user_index+1):
        if x not in sample.keys():
            sample[x]=[]
        for x in range(100):
            train_dataloader
