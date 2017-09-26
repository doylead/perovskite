
from random import randint
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
train_case = randint(0, ntrain-1)
test_case = randint(0, ntest-1)

per_tree_train_err = train_set_values[train_case] - \
        per_tree_train[train_case]
per_tree_test_err = test_set_values[test_case] - \
        per_tree_test[test_case]

p = 95
q = float(100-p)
train_l = np.percentile(per_tree_train_err,q=q/2.,axis=0,interpolation='lower')
train_r = np.percentile(per_tree_train_err,q=100-q/2.,axis=0,interpolation='higher')
in_train = [x for x in per_tree_train_err if (x>=train_l and x<=train_r)]
perc_in_train = 100. * len(in_train) / len(per_tree_train_err)
test_l = np.percentile(per_tree_test_err,q=q/2.,axis=0,interpolation='lower')
test_r = np.percentile(per_tree_test_err,q=100-q/2.,axis=0,interpolation='higher')
in_test = [x for x in per_tree_test_err if (x>=test_l and x<=test_r)]
perc_in_test = 100. * len(in_test) / len(per_tree_test_err)

plt.figure(figsize=(12,6))
plt.subplot(121)
train_hist_label = '%s'%train_set_labels[train_case]
plt.hist(per_tree_train_err, bins=np.arange(-3.9,4.1,0.2), label=train_hist_label)
plt.axvline(train_l,lw=3,color='red',label='%s%% Confidence Interval \n%3.2f eV, %3.1f%%'%(p,train_r-train_l,perc_in_train))
plt.axvline(train_r,lw=3,color='red')
plt.xlabel('Per-Tree Error (eV)',size=18)
plt.ylabel('Frequency',size=18)
plt.title('Training Set; NTree=1000',size=18)
plt.legend()
plt.yscale('log')

plt.subplot(122)
train_hist_label = '%s'%test_set_labels[test_case]
plt.hist(per_tree_test_err, bins=np.arange(-3.9,4.1,0.2), label=train_hist_label)
plt.axvline(test_l,lw=3,color='red',label='%s%% Confidence Interval \n%3.2f eV, %3.1f%%'%(p,test_r-test_l,perc_in_test))
plt.axvline(test_r,lw=3,color='red')
plt.xlabel('Per-Tree Error (eV)',size=18)
plt.ylabel('Frequency',size=18)
plt.title('Training Set; NTree=1000',size=18)
plt.legend()
plt.yscale('log')

plt.tight_layout()
plt.show()
