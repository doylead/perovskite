# Imports
import numpy as np
import pickle
from sklearn import gaussian_process
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

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

# Model Specific Information
# -----------------------
lbound = 1e-2
rbound = 1e1
n_restarts = 20
kernel = C(1.0, (lbound,rbound)) * RBF(n_features*[1.0], (lbound,rbound))
# for some reason had n_features*[10] - what was "10" doing here?
gp = GPR(kernel=kernel, n_restarts_optimizer=n_restarts)
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
#  'RMSE': scalar}
std_output = {'train_set_value': train_targets,
        'train_set_predictions': train_model,
        'test_set_value': test_targets,
        'test_set_predictions': test_model,
        'RMSE': RMSE}
std_output_file = open('standard_output.pickle','w')
pickle.dump(std_output,std_output_file)
std_output_file.close()

# Print model-specific pickle file for later
# Avoid post-processing data
#
# Format for GP:
# {'train_set_uncertainty': 1D-Array, 'test_set_uncertainty': 1D-Array,
# 'kernel_params_dict': dictionary}
#
# Format for NN:
# TBD

# Model Specific Information
# -----------------------
regression_params = open('../fp_instructions.md').read().split('\n')
regression_params = filter(None,regression_params)
regression_params = regression_params[:-1] # Number of training points
                                           # not used in regression
kernel_params = gp.kernel_

model_output = {'train_set_uncertainty': sigma2_pred_test,
        'test_set_uncertainty': sigma2_pred_train,
        'kernel_params_dict': kernel_params_dict}
# -----------------------

model_output_file = open('model_specific_output.pickle','w')
pickle.dump(model_output,model_output_file)
model_output_file.close()
