from sklearn.model_selection import train_test_split
from torch import np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.optim as optim
from lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch
import os

class PerovskiteDataset(Dataset):
    """
    class to conform data to pytorch API
    the features are the electronic structure of 
    """
    def __init__(self, train_labels, train_values, train_targets, test_labels, test_values, test_targets, n_features, dtype):
        self.train_labels = train_labels
        self.train_values = train_values
        self.train_targets = train_targets
        self.test_labels = test_labels
        self.test_values = test_values
        self.test_targets = test_targets
        self.n_features = n_features

    def __getitem__(self, index):
        return self.train_values[index], self.train_targets[index]

    def __len__(self):
        return self.train_values.shape[0]

class PerovskiteTestDataset(Dataset):
    """
    class to conform data to pytorch API
    the features are the electronic structure of 
    """
    def __init__(self, train_labels, train_values, train_targets, test_labels, test_values, test_targets, n_features, dtype):
        self.train_labels = train_labels
        self.train_values = train_values
        self.train_targets = train_targets
        self.test_labels = test_labels
        self.test_values = test_values
        self.test_targets = test_targets
        self.n_features = n_features

    def __getitem__(self, index):
        return self.test_values[index], self.test_targets[index]

    def __len__(self):
        return self.test_values.shape[0]


def generate_train_val_dataloader(dataset, batch_size, num_workers, shuffle=False, split=0.8, fraction_of_data=1., train_inds=None, val_inds=None):
    """
    return two Data`s split into training and validation
    `split` sets the train/val split fraction (0.9 is 90 % training data)
    u
    """
    if train_inds == None:
    	inds = np.arange(len(dataset))
    	inds = inds[:int(np.ceil(len(inds)*fraction_of_data))]
    	if fraction_of_data < 1:
        	print 'using ' + str(len(inds)) + ' data points total'
    	train_inds, val_inds = train_test_split(inds, test_size=1-split, train_size=split)

    train_loader = DataLoader(
        dataset,
        sampler=SubsetRandomSampler(train_inds),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        dataset,
        sampler=SubsetRandomSampler(val_inds),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    return train_loader, val_loader, train_inds, val_inds


def train_epoch(loader_train, model, loss_fn, optimizer, dtype, print_every=20):
    """
    train `model` on data from `loader_train` for one epoch
    inputs:
    `loader_train` object subclassed from torch.data.DataLoader
    `model` neural net, subclassed from torch.nn.Module
    `loss_fn` loss function see torch.nn for examples
    `optimizer` subclassed from torch.optim.Optimizer
    `dtype` data type for variables
        eg torch.FloatTensor (cpu) or torch.cuda.FloatTensor (gpu)
    """
    loss_history = []
    model.train()
    inds = loader_train.sampler.indices
    for t, (x, y) in enumerate(loader_train):
        x_var = Variable(x.type(dtype))
        y_var = Variable(y.type(dtype))

        y_pred = model(x_var)

        loss = loss_fn(y_pred, y_var)
        loss_history.append(loss.data[0])

        if (t + 1) % print_every == 0:
            print('t = %d, loss = %.4f, f2 = %.4f' % (t + 1, loss.data[0], acc))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if len(loader_train) == 1:
        returnvals = y_pred, y_var
    else:
        sys.exit('many batches not implemented')

    return np.sqrt(loss.data.numpy()[0]), returnvals

def validate_epoch(model, loader, loss_fn, dtype):
    """
    validation for MultiLabelMarginLoss using f2 score
    `model` is a trained subclass of torch.nn.Module
    `loader` is a torch.dataset.DataLoader for validation data
    `dtype` data type for variables
        eg torch.FloatTensor (cpu) or torch.cuda.FloatTensor (gpu)
    """
    loss_history = []
    model.eval()
    for t, (x, y) in enumerate(loader):
        x_var = Variable(x.type(dtype))
        y_var = Variable(y.type(dtype))

        y_pred = model(x_var)

        loss = loss_fn(y_pred, y_var)
        loss_history.append(loss.data[0])

    if len(loader) == 1:
        returnvals = (y_pred, y_var)
    else:
        sys.exit('many batches not implemented')

    return np.sqrt(loss.data.numpy()[0]), returnvals


def get_pred(model, loader, dtype):
    y_pred_array = np.zeros(len(loader.sampler))
    bs = loader.batch_size
    model.eval()
    for i, (x, y) in enumerate(loader):
        x_var = Variable(x.type(dtype))
        y_var = Variable(y.type(dtype))

        y_pred = model(x_var)
        y_pred_array[i*bs:(i+1)*bs] = y_pred.data.numpy()[:,0]

    return y_pred


def un_normalized_RMSE(y_pred, y_var, mean_target, std_target):
    # errors = (y_pred - y_var).data.numpy()
    y_pred = np.multiply(y_pred.data.numpy(),std_target) + mean_target
    y = np.multiply(y_var.data.numpy(),std_target) + mean_target
    y_pred = y_pred[:,0]
    ntest = len(y)
    errors = np.abs(y_pred - y)
    RMSE = np.sqrt(np.dot(errors, errors) / ntest)
    return RMSE























    






















    




























