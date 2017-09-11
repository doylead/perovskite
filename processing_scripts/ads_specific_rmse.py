
import numpy as np
import pickle
import sys

# args[1] must be a path to the directory to be analyzed
# args[2] may be a plotting method (A, B, ads) - to be implemented

def get_rmse(errors):
        n = len(errors)
        rmse = np.sqrt(np.dot(errors,errors)/n)
        return rmse

args = sys.argv

if len(args)==1:
        raise ValueError('Please specify a path to analyze, e.g. fp_00001/gaussian_process')
else:
        p = args[1]

data = pickle.load(open(p+'/standard_output.pickle'))
train_labels = data['train_set_labels']
train_dft = data['train_set_value']
train_predictions = data['train_set_predictions']
test_labels = data['test_set_labels']
test_dft = data['test_set_value']
test_predictions = data['test_set_predictions']
RMSE = data['RMSE']

all_ads = ['H_M', 'O', 'OH', 'OOH', 'N', 'NH', 'NH2', 'NNH']
all_ads_dict = {}
for ads in all_ads:
        all_ads_dict[ads] = []

ntest = len(test_labels)
for i in range(ntest):
        label = test_labels[i]
        err = test_dft[i] - test_predictions[i]
        ads = label.split(' ')[1]
        all_ads_dict[ads].append(err)

print 'Ads\t n\t RMSE (eV)'
print 25*'-'
for ads in all_ads:
        err = all_ads_dict[ads]
        n = len(err)
        rmse = get_rmse(err)
        line = '%s\t %d\t %3.2f'%(ads,n,rmse)
        print line
