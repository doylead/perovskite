import sys
sys.path.append('../../database_scripts')
sys.path.append('../../nn_scripts')

# Imports
from database import insert_experiment
from os import getcwd
import numpy as np
import pickle
from sklearn import gaussian_process
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process.kernels import WhiteKernel as WK

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
from nn_utils import PerovskiteDataset, PerovskiteTestDataset, generate_train_val_dataloader, train_epoch, validate_epoch, get_pred, un_normalized_RMSE

normalization_data = pickle.load(open('../normalization_parameters.pickle'))
mean_target = normalization_data['mean_vals'][-1]
std_target = normalization_data['std_vals'][-1]

# Load feature matrices
feature_matrices = pickle.load(open('../data.pickle'))
train_labels = feature_matrices['train_labels']
train_values = feature_matrices['train_values']
train_targets = feature_matrices['train_targets']
test_labels = feature_matrices['test_labels']
test_values = feature_matrices['test_values']
test_targets = feature_matrices['test_targets']
n_features = train_values.shape[1]

# Model Specific Information
# -----------------------
mode = 'train_continue'
# mode = 'train_from_scratch'
mode = 'test'


dtype = 'torch.FloatTensor'
lr = 1e-2
reg = 1e-3
num_epochs = 10
shuffle = False

root = "model" # name of model
save_model_path = "{}_state_dict.pkl".format(root)
save_pkl_path = "{}_loss".format(root)

if os.path.exists(save_model_path) == False and mode == 'train_continue':
    mode = 'train_from_scratch'


dataset = PerovskiteDataset(train_values, train_values, train_targets, test_labels, test_values, test_targets, n_features, dtype)
batch_size = len(dataset)
num_workers = 4
train_loader, val_loader = generate_train_val_dataloader(dataset, batch_size, num_workers, shuffle=shuffle, split=0.8, fraction_of_data = 1.)

N = len(train_loader.sampler)
p = dataset.train_values.shape[1]

model = nn.Sequential(
    nn.Linear(p,10),
    nn.ReLU(inplace = True),
    nn.Linear(10,10),
    nn.ReLU(inplace = True),
    nn.Linear(10,10),
    nn.ReLU(inplace = True),
    nn.Linear(10,10),
    nn.ReLU(inplace = True),
    nn.Linear(10,1)
)
model.type(dtype)

loss_fn = nn.MSELoss().type(dtype)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay = reg)
scheduler = ReduceLROnPlateau(optimizer, patience=1, factor=0.5, min_lr=0.01*lr)

if mode == 'train_continue':
    state_dict = torch.load(save_model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    print("model loaded from {}".format(os.path.abspath(save_model_path)))
    (train_RMSE_history, val_RMSE_history) = pickle.load(open(save_pkl_path, 'r'))
elif mode == 'train_from_scratch':
    train_RMSE_history = []
    val_RMSE_history = []

if mode in ['train_from_scratch', 'train_continue']:
    i0 = len(train_RMSE_history)
    i = len(train_RMSE_history)
    for epoch in range(num_epochs):
        print("Begin epoch {}/{}".format(i+1, i0+num_epochs))
        train_loss, (x_var_train, y_var_train) = train_epoch(train_loader, model, loss_fn, optimizer, dtype, print_every=20)
        train_RMSE = un_normalized_RMSE(x_var_train, y_var_train, mean_target, std_target)
        train_RMSE_history.append(np.sqrt(train_RMSE))

        scheduler.step(train_loss, epoch)

        val_loss, (y_pred_val, y_var_val) = validate_epoch(model, val_loader, loss_fn, dtype)
        val_RMSE = un_normalized_RMSE(y_pred_val, y_var_val, mean_target, std_target)
        val_RMSE_history.append(val_RMSE)

        print 'training loss:', train_RMSE_history[-1]
        print 'validation loss:', val_RMSE_history[-1]
        i += 1

    # print train_RMSE_history
    # print val_RMSE_history
    torch.save(model.state_dict(), save_model_path)
    pickle.dump((train_RMSE_history, val_RMSE_history), open(save_pkl_path, 'w'))
    # print("model saved as {}".format(os.path.abspath(save_model_path)))
    plt.plot(range(len(train_RMSE_history)), train_RMSE_history, label = 'train')
    plt.plot(range(len(val_RMSE_history)), val_RMSE_history, label = 'validation')
    plt.legend(loc='best')
    plt.show()



if mode == 'test':
    state_dict = torch.load(save_model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    print("model loaded from {}".format(os.path.abspath(save_model_path)))
    test_dataset = PerovskiteTestDataset(train_values, train_values, train_targets, test_labels, test_values, test_targets, n_features, dtype)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    test_model = get_pred(model, test_loader, dtype)
    test_model = test_model.data.numpy()[:,0]

    complete_train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    train_model = get_pred(model, complete_train_loader, dtype)
    train_model = train_model.data.numpy()

if mode != 'test':
    exit()

''' doylead's code - sacred variable names are test_model train_model
lbound = 1e-3
rbound = 1e1
n_restarts = 5
kernel = C(1.0, (lbound,rbound)) * RBF(n_features*[10], (lbound,rbound)) + WK(noise_level_bounds=(lbound,rbound))
gp = GPR(kernel=kernel, n_restarts_optimizer=n_restarts, alpha=1e-1)
gp.fit(train_values, train_targets)

test_model, sigma2_pred_test = gp.predict(test_values, return_std=True)
train_model, sigma2_pred_train = gp.predict(train_values, return_std=True)
'''
# -----------------------

# Undo normalization in FP generation
test_model = np.multiply(test_model,std_target) + mean_target
# sigma2_pred_test = np.multiply(sigma2_pred_test,std_target) # GP Specific
test_targets = np.multiply(test_targets,std_target) + mean_target

train_model = np.multiply(train_model,std_target) + mean_target
# sigma2_pred_train = np.multiply(sigma2_pred_train,std_target) # GP Specific
train_targets = np.multiply(train_targets,std_target) + mean_target

# Update database
ntest = len(test_targets)
err_model_test = np.abs(test_model - test_targets)
RMSE = np.sqrt(np.dot(err_model_test,err_model_test)/ntest)
print 'RMSE', RMSE


# Print standard pickle file for later comparison
#
# Format:
# {'train_set_values': 1D-array, 'train_set_predictions': 1D-array,
#  'test_set_values': 1D-array, 'test_set_predictions': 1D-array,
#  'train_set_labels': 1D-array, 'test_set_labels': 1D-array,
#  'RMSE': scalar}
std_output = {'train_set_value': train_targets,
        'train_set_predictions': train_model,
        'train_set_labels': train_labels,
        'test_set_value': test_targets,
        'test_set_predictions': test_model,
        'test_set_labels': test_labels,
        'RMSE': RMSE}
std_output_file = open('standard_output.pickle','w')
pickle.dump(std_output,std_output_file)
std_output_file.close()

# Print model-specific pickle file for later
# Avoid post-processing data
#
# Format for GP:
# {'train_set_uncertainty': 1D-Array, 'test_set_uncertainty': 1D-Array,
# 'kernel_params': optimized_parameters}
#
# Format for NN:
# TBD

# Model Specific Information
# -----------------------

# model_output_file = open('model_specific_output.pickle','w')
# pickle.dump(model_output,model_output_file)
# model_output_file.close()

# Append Results to Database
cwd = getcwd()
FSDIR = cwd.split('/')[-2]
FSID = int(FSDIR.strip('fp_'))
model_type = 'nn'
architecture = str(model)
activation_function = 'ReLU'
learning_rate = lr
regularization = reg
dbpath = '../../data.db'

ID, status = insert_experiment(featuresubsetid = FSID,
        dbpath = dbpath,
        model_type = model_type,
        RMSE = RMSE,
        architecture = architecture,
        activation_function = activation_function,
        learning_rate = learning_rate,
        regularization = regularization
)

if status != 'inserted':
        print 'Error inserting experiment records into database. ID, status =', ID, status
# -----------------------
