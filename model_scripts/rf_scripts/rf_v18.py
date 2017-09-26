
import sys
sys.path.append('../../database_scripts')

# Imports
from database import insert_experiment
from os import getcwd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor

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
ntrain = len(train_targets)
ntest = len(test_targets)
n_estimators = 100
max_features = 'log2'

rf = RandomForestRegressor(n_estimators=n_estimators, max_features=max_features)
rf.fit(train_values, train_targets)
test_model = rf.predict(test_values)
train_model = rf.predict(train_values)

# -----------------------

# Undo normalization in FP generation
test_model = np.multiply(test_model,std_target) + mean_target
test_targets = np.multiply(test_targets,std_target) + mean_target

train_model = np.multiply(train_model,std_target) + mean_target
train_targets = np.multiply(train_targets,std_target) + mean_target

# Update database
err_model_test = np.abs(test_model - test_targets)
err_model_train = np.abs(train_model - train_targets)
RMSE_train = np.sqrt(np.dot(err_model_train, err_model_train)/ntrain)
RMSE_test = np.sqrt(np.dot(err_model_test,err_model_test)/ntest)

# Print standard pickle file for later comparison
# 
# Format:
# {'train_set_values': 1D-array, 'train_set_predictions': 1D-array,
#  'test_set_values': 1D-array, 'test_set_predictions': 1D-array,
#  'train_set_labels': 1D-array, 'test_set_labels': 1D-array,
#  'RMSE_train': scalar, 'RMSE_test': scalar}
std_output = {'train_set_value': train_targets,
        'train_set_predictions': train_model,
        'train_set_labels': train_labels,
        'test_set_value': test_targets,
        'test_set_predictions': test_model,
        'test_set_labels': test_labels,
        'RMSE_train': RMSE_train,
        'RMSE_test': RMSE_test}
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
#
# Format for RF:
# {'feature_importances': 1D-array}

# Model Specific Information
# -----------------------
train_predictions = [ tree.predict(train_values) for tree in rf.estimators_ ]
train_predictions = np.vstack(train_predictions).T
train_predictions = np.multiply(train_predictions, std_target) + mean_target
test_predictions = [ tree.predict(test_values) for tree in rf.estimators_ ]
test_predictions = np.vstack(test_predictions).T
test_predictions = np.multiply(test_predictions, std_target) + mean_target

model_output = {'feature_importances': rf.feature_importances_,
                'train_prediction_trees': train_predictions,
                'test_prediction_trees': test_predictions}

model_output_file = open('model_specific_output.pickle','w')
pickle.dump(model_output,model_output_file)
model_output_file.close()

# Append Results to Database
cwd = getcwd()
FSDIR = cwd.split('/')[-2]
FSID = int(FSDIR.strip('fp_'))
model_type = 'rf_v18' 
architecture = None
activation_function = None
learning_rate = None
regularization = None
dbpath = '../../data.db'

ID, status = insert_experiment(featuresubsetid = FSID,
        dbpath = dbpath,
        model_type = model_type, 
        RMSE_test = RMSE_test,
        RMSE_train = RMSE_train,
        architecture = architecture, 
        activation_function = activation_function,
        learning_rate = learning_rate,
        regularization = regularization)

if status != 'inserted':
        print 'Error inserting experiment records into database'
# -----------------------

