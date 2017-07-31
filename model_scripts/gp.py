import sys
sys.path.append('../../database_scripts')

# Imports
from database import insert_experiment
from os import getcwd
import numpy as np
import pickle
from sklearn import gaussian_process
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process.kernels import WhiteKernel as WK

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
lbound = 1e-3
rbound = 1e1
n_restarts = 5
kernel = C(1.0, (lbound,rbound)) * RBF(n_features*[10], (lbound,rbound)) + WK(noise_level_bounds=(lbound,rbound))
gp = GPR(kernel=kernel, n_restarts_optimizer=n_restarts, alpha=1e-1)
gp.fit(train_values, train_targets)

test_model, sigma2_pred_test = gp.predict(test_values, return_std=True)
train_model, sigma2_pred_train = gp.predict(train_values, return_std=True)
# -----------------------

# Undo normalization in FP generation
test_model = np.multiply(test_model,std_target) + mean_target
sigma2_pred_test = np.multiply(sigma2_pred_test,std_target) # GP Specific
test_targets = np.multiply(test_targets,std_target) + mean_target

train_model = np.multiply(train_model,std_target) + mean_target
sigma2_pred_train = np.multiply(sigma2_pred_train,std_target) # GP Specific
train_targets = np.multiply(train_targets,std_target) + mean_target

# Update database
ntest = len(test_targets)
err_model_test = np.abs(test_model - test_targets)
RMSE = np.sqrt(np.dot(err_model_test,err_model_test)/ntest)

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
kernel_params = gp.kernel_
print kernel_params

model_output = {'train_set_uncertainty': sigma2_pred_test,
        'test_set_uncertainty': sigma2_pred_train,
        'kernel_params': kernel_params}

model_output_file = open('model_specific_output.pickle','w')
pickle.dump(model_output,model_output_file)
model_output_file.close()

# Append Results to Database
cwd = getcwd()
FSDIR = cwd.split('/')[-2]
FSID = int(FSDIR.strip('fp_'))
model_type = 'gp' 
architecture = None
activation_function = None
learning_rate = None
regularization = None
dbpath = '../../data.db'

ID, status = insert_experiment(featuresubsetid = FSID,
        dbpath = dbpath,
        model_type = model_type, 
        RMSE = RMSE, 
        architecture = architecture, 
        activation_function = activation_function,
        learning_rate = learning_rate,
        regularization = regularization)

if status != 'inserted':
        print 'Error inserting experiment records into database'
# -----------------------
