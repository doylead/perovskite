
from math import sqrt
import pylab as plt
import numpy as np
import pickle

standard_output = pickle.load(open('standard_output.pickle'))
train_set_labels = standard_output['train_set_labels']
train_set_values = standard_output['train_set_value']
train_set_predictions = standard_output['train_set_predictions']
test_set_labels = standard_output['test_set_labels']
test_set_values = standard_output['test_set_value']
test_set_predictions = standard_output['test_set_predictions']

model_specific_output = pickle.load(open('model_specific_output.pickle'))
per_tree_train = model_specific_output['train_prediction_trees']
per_tree_test = model_specific_output['test_prediction_trees']

# Plot the distribution of errors for randomly selected cases
ntrain = len(train_set_values)
ntest = len(test_set_values)

train_std = []
train_err = []
total_y_lte_x_count = 0
total_y_lte_sqrt_x_count = 0
for i in range(ntrain):
        x = np.std(per_tree_train[i])
        y = abs(train_set_values[i]-train_set_predictions[i])
        if y<=x:
                total_y_lte_x_count += 1
        if y<=sqrt(x):
                total_y_lte_sqrt_x_count += 1
        train_std.append(x)
        train_err.append(y)

test_std = []
test_err = []
test_y_lte_x_count = 0
test_y_lte_sqrt_x_count = 0
for i in range(ntest):
        x = np.std(per_tree_test[i])
        y = abs(test_set_values[i]-test_set_predictions[i])
        if y<=x:
                test_y_lte_x_count += 1
        if y<=sqrt(x):
                test_y_lte_sqrt_x_count += 1
        test_std.append( x )
        test_err.append( y )

total_y_lte_x_count += test_y_lte_x_count
test_frac_y_lte_x = 100. * test_y_lte_x_count / ntest
total_frac_y_lte_x = 100. * total_y_lte_x_count / (ntest + ntrain)

total_y_lte_sqrt_x_count += test_y_lte_sqrt_x_count
test_frac_y_lte_sqrt_x = 100. * test_y_lte_sqrt_x_count / ntest
total_frac_y_lte_sqrt_x = 100. * total_y_lte_sqrt_x_count / (ntest + ntrain)

base = np.linspace(0,4)
sqrt = np.sqrt(base)

lin_label = 'Y=X Bounds \n%2.1f%%: Test\n%2.1f%%: Train'%(test_frac_y_lte_x,total_frac_y_lte_x)
sqr_label = 'Y=sqrt(X) Bounds \n%2.1f%%: Test\n%2.1f%%: Train'%(test_frac_y_lte_sqrt_x,total_frac_y_lte_sqrt_x)
plt.plot(base,base,lw=2,color='black',label=lin_label)
plt.plot(base,sqrt,lw=2,ls='--',color='black',label=sqr_label)
plt.plot(train_std,train_err,ms=12,marker='s',lw=0,color='red',label='Training Set')
plt.plot(test_std,test_err,ms=12,marker='o',lw=0,color='blue',label='Test Set')
plt.xlabel('STD of Trees (eV)',size=18)
plt.ylabel('Absolute Error (eV)',size=18)
plt.legend(numpoints=1,loc='upper left',ncol=2)
plt.tight_layout()
plt.show()

