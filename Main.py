
import Dataloader as dataloader
from Dataloader import DataLoader_RecSys
from Autoencoder import Autoencoder, training_loop
from Consts import *


if __name__ == "__main__":

    d1, validation_data = dataloader.initialProcessData(TRAIN_DATA_PATH)
    train_dataloader = DataLoader_RecSys(d1)
    args = AutoEncoderArgs()
    model = Autoencoder(args=args)
    print(train_dataloader.userSeenItems(1))
    print(train_dataloader.userUnseenItems(1))
