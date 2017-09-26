
import sys
sys.path.append('..')

import pickle
from ase import Atoms
import pylab as plt
import numpy as np
from plot_pt import plot_periodic_table

fp = 32
filename = '../../fp_%05d/data.pickle'%fp
data = pickle.load(open(filename))

training_set = data['train_labels']
test_set = data['test_labels']

all_A = []
all_B = []
all_ads = []
train_A = {}
train_B = {}
train_ads = {}
test_A = {}
test_B = {}
test_ads = {}

for key in training_set:
        perovskite, ads = key.split(',')
        ads = ads.lstrip(' ').rstrip(' ads')
        atoms = Atoms(perovskite)
        A = atoms[0].symbol
        B = atoms[1].symbol
        if A not in all_A:
                all_A.append(A)
        if B not in all_B:
                all_B.append(B)
        if ads not in all_ads:
                all_ads.append(ads)
        if A in train_A.keys():
                train_A[A] += 1
        if A not in train_A.keys():
                train_A[A] = 1
        if B in train_B.keys():
                train_B[B] += 1
        if B not in train_B.keys():
                train_B[B] = 1
        if ads in train_ads.keys():
                train_ads[ads] += 1
        if ads not in train_ads.keys():
                train_ads[ads] = 1


for key in test_set:
        perovskite, ads = key.split(',')
        ads = ads.lstrip(' ').rstrip(' ads')
        atoms = Atoms(perovskite)
        A = atoms[0].symbol
        B = atoms[1].symbol
        if A not in all_A:
                all_A.append(A)
        if B not in all_B:
                all_B.append(B)
        if ads not in all_ads:
                all_ads.append(ads)
        if A in test_A.keys():
                test_A[A] += 1
        if A not in test_A.keys():
                test_A[A] = 1
        if B in test_B.keys():
                test_B[B] += 1
        if B not in test_B.keys():
                test_B[B] = 1
        if ads in test_ads.keys():
                test_ads[ads] += 1
        if ads not in test_ads.keys():
                test_ads[ads] = 1

for A in all_A:
        if A not in train_A.keys():
                train_A[A] = 0
        if A not in test_A.keys():
                test_A[A] = 0

for B in all_B:
        if B not in train_B.keys():
                train_B[B] = 0
        if B not in test_B.keys():
                test_B[B] = 0

for ads in all_ads:
        if ads not in train_ads.keys():
                train_ads[ads] = 0
        if ads not in test_ads.keys():
                test_ads[ads] = 0

'''
plot_periodic_table(train_A, title='A Metals in Training Set',
        vmin=0, vmax=max(train_A.values()), num_format='%d', show=False,
        filename='A_train_fp_%05d.png'%fp)

plot_periodic_table(train_B, title='B Metals in Training Set',
        vmin=0, vmax=max(train_B.values()), num_format='%d', show=False,
        filename='B_train_fp_%05d.png'%fp)
'''

rat_A = {}
rat_B = {}
for A in all_A:
        rat_A[A] = float('%d.%d'%(train_A[A],test_A[A]))

plot_periodic_table(rat_A, title='"Ratio" of A Metals in Training Set',
        vmin=0, vmax=1, show=True, num_format='%s')

for B in all_B:
        rat_B[B] = float('%d.%d'%(train_B[B],test_B[B]))

plot_periodic_table(rat_B, title='"Ratio" of B Metals in Training Set',
        vmin=0, vmax=1, show=True, num_format='%s')

all_ads.sort()
header = 'Ads\t N Train\t N Test'
linebreak = 32*'-'
f = open('ads_train_fp_%05d.txt'%fp,'w')
print >>f,header
print >>f,linebreak
for ads in all_ads:
        line = '%s\t %d\t\t %d'%(ads,train_ads[ads],test_ads[ads])
        print >>f,line

f.close()

