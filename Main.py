
import Dataloader as dataloader
from Dataloader import DataLoader_RecSys
from Autoencoder import Autoencoder, training_loop
from Consts import *
from torch import nn


if __name__ == "__main__":

    d1, validation_data,popularity = dataloader.initialProcessData(TRAIN_DATA_PATH)
    train_dataloader = DataLoader_RecSys(d1,popularity)
    args = AutoEncoderArgs()
    model = Autoencoder(args=args)
    print(train_dataloader.numOfItems())
    print(train_dataloader.userSeenItems(1))
    print(train_dataloader.userUnseenItems(1))

    training_loop(args,
                  model,
                  tr_dataloader=train_dataloader,
                  validation=validation_data,
                  criterion_func=nn.MSELoss)
