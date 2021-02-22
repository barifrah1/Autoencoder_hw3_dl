import os
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch import optim

num_epochs = 100
batch_size = 128
learning_rate = 1e-3


class Autoencoder(nn.Module):
    def __init__(self, args=None):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(args.input_size, args.hidden_size, bias=True),
            nn.Sigmoid())
        self.decoder = nn.Sequential(
            nn.Linear(args.hidden_size, args.input_size, bias=True),
            nn.Sigmoid())
        self.args = args

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def infer(dataloader, validation, model):
    accuracy = 0
    for userVec in dataloader:
        output = model(userVec)
        index = dataloader.currentUserIndex()
        validationUserSeenItem = validation[index]
        itemDrawn = dataloader.drawUnseenItem(index)
        while(itemDrawn == validationUserSeenItem):
            itemDrawn = dataloader.drawUnseenItem(index)
        if(output[validationUserSeenItem] > output[itemDrawn]):
            accuracy += 1
        accuracy /= dataloader.numOfUsers()
    return accuracy


def training_loop(args,
                  model,
                  tr_dataloader=None,
                  validation=None,
                  criterion_func=nn.MSELoss,
                  ):
    accuracy_by_epoch = []
    criterion = criterion_func()
    optimizer = optim.Adam(net.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)

    for epoch in range(args.num_epochs):
        for userVec in tr_dataloader:
            # ===================forward=====================
            output = model(userVec)
            loss = criterion(output, userVec)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        if epoch % 1 == 0:
            currentAccuracy = infer(dataloader, validation, model)
            accuracy_by_epoch.append(currentAccuracy)
            print(f" epoch: { epoch+1} validation accuracy: {currentAccuracy}")

    return accuracy_by_epoch


"""
def training_loop(
    args,
    net,
    tr_dataloader=None,
    validation,
    criterion_func=nn.MSELoss,
    optimizer_func=optim.SGD,
):
    train_on_gpu = torch.cuda.is_available()
    criterion = criterion_func()
    optimizer = optim.Adam(net.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    tr_loss, val_loss = [None] * args.num_epochs, [None] * args.num_epochs
    tr_auc, val_auc = [None] * args.num_epochs, [None] * args.num_epochs
    test_loss, untrained_test_loss = None, None
    test_auc, untrained_test_auc = None, None
    # Note that I moved the inferences to a function because it was too much code duplication to read.
    # calculate error before training
    auc_and_loss = infer(
        net,  criterion, X=X_test, y=y_test, dataloader=val_dataloader)
    untrained_test_loss = auc_and_loss[0]
    untrained_test_auc = auc_and_loss[1]
    for epoch in range(args.num_epochs):
        net.train()
        running_tr_loss = 0
        running_tr_auc = 0
        data_size = len(X_train)
        if(data_size % args.batch_size == 0):
            no_of_batches = data_size // args.batch_size
        else:
            no_of_batches = (data_size // args.batch_size)+1
        for i in tqdm(range(no_of_batches)):
            start = i*args.batch_size
            end = i*args.batch_size + args.batch_size
            x = X_train[start:end]
            y = y_train[start:end]
            optimizer.zero_grad()
            if train_on_gpu:
                net.cuda()
                x, y = x.cuda(), y.cuda()
            pred = net(x)
            auc = roc_auc_score(y, pred.detach().numpy()[:, 1])
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            running_tr_loss += loss*x.shape[0]
            running_tr_auc += auc*x.shape[0]

        tr_loss[epoch] = running_tr_loss.item() / data_size
        tr_auc[epoch] = running_tr_auc.item() / data_size
        auc_and_loss = infer(
            net, criterion, X=X_test, y=y_test, dataloader=None)
        val_loss[epoch] = auc_and_loss[0]
        val_auc[epoch] = auc_and_loss[1]
        print(
            f"Train loss: {tr_loss[epoch]:.2e}, Val loss: {val_loss[epoch]:.2e}")
        print(
            f"Best val loss is: {min(x for x in val_loss if x is not None):.2e}")
        if epoch >= args.early_stopping_num_epochs:
            improvement = (
                val_loss[epoch - args.early_stopping_num_epochs] -
                val_loss[epoch]
            )
            if improvement < args.early_stopping_min_improvement:
                break
    auc_and_loss = infer(net, criterion, X=X_test, y=y_test, dataloader=None)
    test_loss = auc_and_loss[0]
    test_auc = auc_and_loss[1]
    print(f"Stopped training after {epoch+1}/{args.num_epochs} epochs.")
    print(
        f"The loss is {untrained_test_loss:.2e} before training and {test_loss:.2e} after training."
    )
    print(
        f"The training and validation losses are "
        f"\n\t{tr_loss}, \n\t{val_loss}, \n\tover the training epochs, respectively."
    )
    return net, tr_loss, val_loss, test_loss, tr_auc, val_auc, untrained_test_loss, untrained_test_auc
"""
