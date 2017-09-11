
from ase import Atoms
import pylab as plt
import numpy as np
import operator
import pickle
import sys
sys.path.append('/home/doylead/doylead/data_science/perovskite/collab/perovskite/processing_scripts')
from plot_pt import plot_periodic_table

# args[1] must be a path to the directory to be analyzed
# args[2] may be a plotting method (A, B, ads) - to be implemented

def get_rmse(errors):
        n = len(errors)
        rmse = np.sqrt(np.dot(errors,errors)/n)
        return rmse

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

data = pickle.load(open(p+'/standard_output.pickle'))
train_labels = data['train_set_labels']
train_dft = data['train_set_value']
train_predictions = data['train_set_predictions']
test_labels = data['test_set_labels']
test_dft = data['test_set_value']
test_predictions = data['test_set_predictions']
RMSE = data['RMSE_test']

if True: # A sort, B sort
        
        all_A_err = {}
        all_A_rmse = {}

        all_B_err = {}
        all_B_rmse = {}

        n_cats = len(test_labels)
        for i in range(n_cats):
                cat = test_labels[i].split(',')[0]
                err = test_dft[i] - test_predictions[i]
                atoms = Atoms(cat)
                A = atoms[0].symbol
                B = atoms[1].symbol
                if A not in all_A_err.keys():
                        all_A_err[A] = []
                if B not in all_B_err.keys():
                        all_B_err[B] = []
                all_A_err[A].append(err)
                all_B_err[B].append(err)

        for key in all_A_err.keys():
                all_A_rmse[key] = get_rmse(all_A_err[key])
        for key in all_B_err.keys():
                all_B_rmse[key] = get_rmse(all_B_err[key])

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

        all_ads_err = {}
        all_ads_rmse = {}
        for ads in colors.keys():
                all_ads_err[ads] = []

        ntest = len(test_labels)
        for i in range(ntest):
                label = test_labels[i]
                err = test_dft[i] - test_predictions[i]
                ads = label.split(' ')[1]
                all_ads_err[ads].append(err)

        for ads in colors.keys():
                all_ads_rmse[ads] = get_rmse(all_ads_err[ads])

# Text output
f = open(p+'diagnostics.txt','w')
print >>f,'A Metal\t\t Subset RMSE (eV)'
print >>f,33*'-'
sorted_A_rmse = sorted(all_A_rmse.items(), key=operator.itemgetter(1), reverse=True)
for key in sorted_A_rmse:
        print >>f,key[0]+'\t\t %3.2f'%key[1]

print >>f,''
print >>f,''
print >>f,'B Metal\t\t Subset RMSE (eV)'
print >>f,33*'-'
sorted_B_rmse = sorted(all_B_rmse.items(), key=operator.itemgetter(1), reverse=True)
for key in sorted_B_rmse:
        print >>f,key[0]+'\t\t %3.2f'%key[1]

print >>f,''
print >>f,''
print >>f,'Ads\t\t Subset RMSE (eV)'
print >>f,33*'-'
sorted_ads_rmse = sorted(all_ads_rmse.items(), key=operator.itemgetter(1), reverse=True)
for key in sorted_ads_rmse:
        print >>f,key[0]+'\t\t %3.2f'%key[1]

# Plotting

ntrain = len(train_labels)
showTrain = False # Plot the training set?
if showTrain:
        for i in range(ntrain):
                datum = train_labels[i].split(' ')
                if method=='ads':
                        colorspec = datum[1]
                plt.plot(train_dft[i], train_predictions[i], marker='s', ms=8,
                        lw=0, color=colors[colorspec], alpha=0.5)

ntest = len(test_labels)
for i in range(ntest):
        datum = test_labels[i].split(' ')
        if method=='ads':
                colorspec = datum[1]
        plt.plot(test_dft[i], test_predictions[i], marker='o', ms=12,
                lw=0, color=colors[colorspec])

if True: # Show labels?
        x = [100, 101]
        y = [100, 101]
        for c in ['H_M','O','OH','OOH']:
                label = '%s, %3.2f'%(c.rstrip('_M'),all_ads_rmse[c])
                plt.plot(x,y,lw=5,color=colors[c],label=label)
        if showTrain:
                plt.plot(x,y,lw=0,marker='s', ms=8, label='Train',
                        mfc='white', mec='black')
        for c in ['N','NH','NH2','NNH']:
                label = '%s, %3.2f'%(c.rstrip('_M'),all_ads_rmse[c])
                plt.plot(x,y,lw=5,color=colors[c],label=label)
        if showTrain:
                plt.plot(x,y,lw=0,marker='o', ms=12, label='Test',
                        mfc='white', mec='black')
        plt.legend(title='Ads, Ads-Specific RMSE (eV)',ncol=2,numpoints=1,loc='upper left')

plt.xlabel('DFT [eV]',size=18)
plt.ylabel('Model [eV]',size=18)
plt.title('Overall RMSE=%4.3f eV'%RMSE,size=18)
plt.xlim(-6,8)
plt.ylim(-6,8)
plt.tight_layout()
plt.show()

plot_periodic_table(all_A_rmse, title='RMSE by A Metal')
plot_periodic_table(all_B_rmse, title='RMSE by B Metal')
