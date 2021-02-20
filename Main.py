
import Dataloader as dataloader
from Dataloader import DataLoader_RecSys
from Consts import *


if __name__ == "__main__":

    d1, d2 = dataloader.initialProcessData(TRAIN_DATA_PATH)
    train_dataloader = DataLoader_RecSys(d1)
    val_dataloader = DataLoader_RecSys(d1)
    print(train_dataloader.nextitem(1, 0))
