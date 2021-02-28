
import Dataloader as dataloader
from Dataloader import DataLoader_RecSys
#from Autoencoder import Autoencoder, training_loop
from Consts import *
from torch import nn
import pickle


if __name__ == "__main__":

    d1, validation_data,popularity = dataloader.initialProcessData(TRAIN_DATA_PATH)
    train_dataloader = DataLoader_RecSys(d1,popularity)
    print(train_dataloader.max_user_index)
    sample={}
    for x in range(train_dataloader.max_user_index+1):
        if x not in sample.keys():
            sample[x]=[]
        hundred=train_dataloader.drawPopularunseen(x)
        sample[x].append(hundred)
        if x%400==0:
            print('x:',x)

    with open('popluar_sample.pickle', 'wb') as handle:
        pickle.dump(sample, handle, protocol=pickle.HIGHEST_PROTOCOL)

