
import sys
sys.path.append('..')

import pickle
from ase import Atoms
import pylab as plt
import numpy as np

def rgb(r,g,b):
        return (r/255., g/255., b/255.)

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

fp = 13
filename = '../../fp_%05d/data.pickle'%fp
data = pickle.load(open(filename))

training_set_labels = data['train_labels']
training_set_targets = data['train_targets']
ntrain = len(training_set_labels)
test_set_labels = data['test_labels']
test_set_targets = data['test_targets']
ntest = len(test_set_labels)

all_ads = ['H_M', 'O', 'OH', 'OOH',
        'N', 'NH', 'NH2', 'NNH']
ads_cdf_train = {}
ads_cdf_test = {}
ads_N = {}
for ads in all_ads:
        ads_cdf_train[ads] = []
        ads_cdf_test[ads] = []

for i in range(ntrain):
        key = training_set_labels[i]
        perovskite, ads = key.split(',')
        ads = ads.lstrip(' ').rstrip(' ads')
        val = training_set_targets[i]
        ads_cdf_train[ads].append(val)

for i in range(ntest):
        key = test_set_labels[i]
        perovskite, ads = key.split(',')
        ads = ads.lstrip(' ').rstrip(' ads')
        val = test_set_targets[i]
        ads_cdf_test[ads].append(val)

for ads in all_ads:
        ads_N[ads] = float(len(ads_cdf_train[ads]) + len(ads_cdf_test[ads]))

all_ads = ['O']
for ads in all_ads:
        color = colors[ads]

        ads_cdf_train[ads].sort()
        ads_cdf_test[ads].sort()
        ads_cdf_train[ads] = np.array(ads_cdf_train[ads])
        ads_cdf_test[ads] = np.array(ads_cdf_test[ads])

        plt.subplot(121)
        plt.plot(ads_cdf_train[ads],np.array(range(len(ads_cdf_train[ads])))/ads_N[ads],lw=2,color=color)#,label=ads)
        plt.plot(ads_cdf_test[ads],np.array(range(len(ads_cdf_test[ads])))/ads_N[ads],lw=2,ls='--',color=color)
        #plt.legend(loc='upper left')
        plt.xlabel('Adsorption Energy (eV)',size=18)
        plt.ylabel('Cumulative Fraction',size=18)

        plt.subplot(122)
        plt.plot(ads_cdf_train[ads],np.array(range(len(ads_cdf_train[ads])))/float(len(ads_cdf_train[ads])),lw=2,color=color,label=ads)
        plt.plot(ads_cdf_test[ads],np.array(range(len(ads_cdf_test[ads])))/float(len(ads_cdf_test[ads])),lw=2,ls='--',color=color)
        plt.legend(loc='upper left')
        plt.xlabel('Adsorption Energy (eV)',size=18)
        plt.ylabel('Relative Cumulative Fraction',size=18)

plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

