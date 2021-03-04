
TRAIN_DATA_PATH = "Train.csv"
TEST_DATA_RANDOM_PATH = "RandomTest.csv"
TEST_DATA_POPULAR_PATH = "PopularityTest.csv"


class AutoEncoderArgs:
    num_epochs = 100
    batch_size = 128
    lr = 1e-4
    weight_decay = 1e-15
    input_size = 3706
    hidden_size = 80
