
import Dataloader as dataloader
from Dataloader import DataLoader_RecSys
from Autoencoder import Autoencoder, training_loop
import AutoencodePopular as AutoencodePopular
from AutoencodePopular import training_loop as tr_loop
from Consts import *
from torch import nn
import pickle as pickle

if __name__ == "__main__":

    d1, validation_data, popularity = dataloader.initialProcessData(
        TRAIN_DATA_PATH)
    train_dataloader = DataLoader_RecSys(d1, popularity)
    pickle_in = open("PickleOfPopularUnseenLists", "rb")
    unseenPopularItemsList = pickle.load(pickle_in)
    #train_dataloader.createPickleOfPopularUnseenLists(100, 1)
    args = AutoEncoderArgs()
    model = AutoencodePopular.AutoencoderPopular(args=args)
    #train_dataloader.createPickleOfPopularUnseenLists(1000, 50)
    tr_loop(args,
            model,
            tr_dataloader=train_dataloader,
            validation=validation_data,
            criterion_func=nn.MSELoss)
