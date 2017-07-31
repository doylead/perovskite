
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process.kernels import WhiteKernel as WK
import pylab as plt
import numpy as np
import pickle
import sys

# args[1] must be a path to the directory to be analyzed
# args[2] may be a plotting method (A, B, ads) - to be implemented

def rgb(r,g,b):
        return (r/255., g/255., b/255.)

args = sys.argv

if len(args)==1:
        raise ValueError('Please specify a path to analyze, e.g. fp_00001/gaussian_process')
else:
        p = args[1]

if len(args)==3:
        method = args[2]
else:
        method = 'ads'

if method=='ads':
        colors = {
        'H_M': 'black',
        'O': rgb(0, 51, 102),
        'OH': rgb(0, 102, 204),
        'OOH': rgb(51, 153, 255),
        'N': rgb(102, 0, 102),
        'NH': rgb(204, 0, 204),
        'NH2': rgb(255, 51, 255),
        'NNH': rgb(255, 153, 255),
        }

data = pickle.load(open(p+'/standard_output.pickle'))
train_labels = data['train_set_labels']
train_dft = data['train_set_value']
train_predictions = data['train_set_predictions']
test_labels = data['test_set_labels']
test_dft = data['test_set_value']
test_predictions = data['test_set_predictions']
RMSE = data['RMSE']

gp_data = pickle.load(open(p+'/model_specific_output.pickle'))
train_uncertainty = gp_data['train_set_uncertainty']
test_uncertainty = gp_data['test_set_uncertainty']
kernel_params = gp_data['kernel_params']
print kernel_params

train_abs_err = np.abs(train_dft-train_predictions)
test_abs_err = np.abs(test_dft-test_predictions)

# Plotting

ntest = len(test_labels)
for i in range(ntest):
        datum = test_labels[i].split(' ')
        if method=='ads':
                colorspec = datum[1]
        plt.plot(test_abs_err[i], test_uncertainty[i], marker='o', ms=12,
                lw=0, color=colors[colorspec])


if True: # Show labels?
        x = [100, 101]
        y = [100, 101]
        for c in ['H_M','O','OH','OOH']:
                plt.plot(x,y,lw=5,color=colors[c],label=c)
        for c in ['N','NH','NH2','NNH']:
                plt.plot(x,y,lw=5,color=colors[c],label=c)
        plt.legend(ncol=2,numpoints=1,loc='upper left')

plt.xlabel('Absolute Model Error [eV]',size=18)
plt.ylabel('Model Uncertainty [eV]',size=18)
plt.xlim(0,5)
plt.ylim(0,5)

plt.tight_layout()
plt.show()

