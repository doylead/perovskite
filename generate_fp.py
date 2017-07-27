
from database import add_feature_set
from os import getcwd, mkdir, chdir
from mendeleev import element
from ase import Atoms
from math import ceil
from sys import exit
import numpy as np
import warnings
import pickle

warnings.simplefilter('error',UserWarning)

def gen_fp(fp, # List of fingerprint types to use
        A_id, # String containing elemental abbreviation for A metal
        B_id, # String containing elemental abbreviation for B metal
        ads_id, # String containing elemental abbreviation for
                     # adsorbate atoms
        target_value, # Adsorption energy; float
        ):

        fp_vals = []
        # Bulk properties will need to be loaded
        # or passed here intelligently somehow
        # - not currently implemented

# Warnings
        if ('primary_ads.pauling_electronegativity' in fp) and \
                ('all_ads.sum_pauling_electronegativity' in fp):
		warnings.warn('Warning: '
                'primary_ads.pauling_electronegativity and '
                'all_ads.sum_pauling_enegativity are identical for '
                'monatomic adsorbates')


# A Block
        if 'A.atomic_number' in fp:
                ele = element(A_id)
                Z = ele.atomic_number
                fp_vals.append(Z)

        if 'A.atomic_radius' in fp:
                ele = element(A_id)
                rad = ele.atomic_radius
                fp_vals.append(rad)

        if 'A.pauling_electronegativity' in fp:
                ele = element(A_id)
                eneg = ele.en_pauling
                fp_vals.append(eneg)

        if 'A.dipole_polarizability' in fp:
                ele = element(A_id)
                polarization = ele.dipole_polarizability	
                fp_vals.append(polarization)

        if 'A.first_ionization_energy' in fp:
                ele = element(A_id)
                first_ionization = ele.ionenergies[1]
                fp_vals.append(first_ionization)

        if 'A.period' in fp:
                ele = element(A_id)
                period = ele.period
                fp_vals.append(period)

# B Block
        if 'B.atomic_number' in fp:
                ele = element(B_id)
                Z = ele.atomic_number
                fp_vals.append(Z)

        if 'B.atomic_radius' in fp:
                ele = element(B_id)
                rad = ele.atomic_radius
                fp_vals.append(rad)

        if 'B.pauling_electronegativity' in fp:
                ele = element(B_id)
                eneg = ele.en_pauling
                fp_vals.append(eneg)

        if 'B.dipole_polarizability' in fp:
                ele = element(B_id)
                polarization = ele.dipole_polarizability	
                fp_vals.append(polarization)

        if 'B.first_ionization_energy' in fp:
                ele = element(B_id)
                first_ionization = ele.ionenergies[1]
                fp_vals.append(first_ionization)

        if 'B.period' in fp:
                ele = element(B_id)
                period = ele.period
                fp_vals.append(period)

# Ads Block

	# Determine the primary adsorbate
	if ads_id in ['O', 'OH', 'OOH']:
		primary_ads_id = 'O'
	elif ads_id in ['H_M']:
		primary_ads_id = 'H'
	elif ads_id in ['N', 'NH', 'NH2', 'NNH']:
		primary_ads_id = 'N'

	# Determine the secondary adsorbates
	if ads_id in ['OH', 'NH']:
		secondary_ads_id = ['H']
	elif ads_id in ['OOH']:
		secondary_ads_id = ['O']
	elif ads_id in ['NH2']:
		secondary_ads_id = ['H', 'H']
	elif ads_id in ['NNH']:
		secondary_ads_id = ['N']
	elif ads_id in ['O', 'H_M', 'N']:
		secondary_ads_id = []

        if 'primary_ads.atomic_number' in fp:
                ele = element(primary_ads_id)
                Z = ele.atomic_number
                fp_vals.append(Z)

        if 'primary_ads.atomic_radius' in fp:
                ele = element(primary_ads_id)
                rad = ele.atomic_radius
                fp_vals.append(rad)

        if 'primary_ads.pauling_electronegativity' in fp:
                ele = element(primary_ads_id)
                eneg = ele.en_pauling
                fp_vals.append(eneg)

        if 'primary_ads.dipole_polarizability' in fp:
                ele = element(primary_ads_id)
                polarization = ele.dipole_polarizability	
                fp_vals.append(polarization)

        if 'primary_ads.first_ionization_energy' in fp:
                ele = element(primary_ads_id)
                first_ionization = ele.ionenergies[1]
                fp_vals.append(first_ionization)

        if 'primary_ads.period' in fp:
                ele = element(primary_ads_id)
                period = ele.period
                fp_vals.append(period)

        if 'all_ads.sum_pauling_electronegativity' in fp:
                base_ele = element(primary_ads_id)
                sum_eneg = base_ele.en_pauling
                for secondary_ele_id in secondary_ads_id:
                        secondary_ele = element(secondary_ele_id)
                        secondary_eneg = secondary_ele.en_pauling
                        sum_eneg -= secondary_eneg
                fp_vals.append(sum_eneg)

# Adds target value
	fp_vals.append(target_value)

# Return
	return fp_vals

def normalize(M,normalize_target=True):
	# Normalize a matrix based on mean and
	# variance.  May or may not include
	# target values
        if not normalize_target:
                temp = M[:,-1]

        mean_vals = np.average(M,axis=0)
        std_vals = np.std(M,axis=0)
        M_p = np.divide(M-mean_vals,std_vals)

        if not normalize_target:
                M_p[:,-1] = temp
                std_vals[-1] = 1
                mean_vals[-1] = 0

        output_file = open('normalization_parameters.pickle','w')
        output = {'mean_vals': mean_vals, 'std_vals': std_vals}
        pickle.dump(output,output_file)
        output_file.close()

        return M_p

# Set up environment
cwd = getcwd()

# Load Data
data = pickle.load(open('../data_summary.pickle'))

fp_all_options = [
        'A.atomic_number', 'A.atomic_radius', 'A.pauling_electronegativity',
        'A.dipole_polarizability', 'A.first_ionization_energy', 'A.period',
        'B.atomic_number', 'B.atomic_radius', 'B.pauling_electronegativity',
        'B.dipole_polarizability', 'B.first_ionization_energy', 'B.period',
	# primary_ads refers only to the adsorbate atom bound to the surface
        'primary_ads.atomic_number', 'primary_ads.atomic_radius',
	'primary_ads.pauling_electronegativity','primary_ads.dipole_polarizability',
	'primary_ads.first_ionization_energy', 'primary_ads.period',
	# all_ads refer to all adsorbate atoms
	'all_ads.sum_pauling_electronegativity'
	]

this_fp = [
           'A.atomic_number',
           'B.atomic_number',
           'primary_ads.atomic_number'
          ]


# if fp has not been used before
ID,status = add_feature_set(this_fp,'data.db')
if status=='already existed':
        print 'Feature set already exists for fisd %d'%ID
	exit()

newdir_index = ID
newdir = 'fp_%05d'%newdir_index
mkdir(newdir)
chdir(newdir)
md_filename = 'fp_instructions.md'
md_file = open(md_filename,'w')
for instruction in this_fp:
        print >>md_file,instruction
md_file.close()

exclude_ads = ['O2', 'H_O'] # Leaves 1,534 points
n_features = len(this_fp)
fp_vals = np.empty(n_features+1)
fp_keys = []

for key in data.keys():
        atoms_object = Atoms(key)
        A_id = atoms_object[0].symbol
        B_id = atoms_object[1].symbol
        for ads_id in data[key].keys():
                if ads_id not in exclude_ads:
                        this_target = data[key][ads_id]
                        this_fp_val = gen_fp(this_fp, A_id, B_id,
	                        ads_id, this_target)
                        fp_vals = np.vstack((fp_vals, this_fp_val))
			fp_keys.append('%s, %s ads'%(key,ads_id))

fp_vals = fp_vals[1:,:]
fp_vals = normalize(fp_vals)

# Distributes the training and test sets
## Trying 10% test, 90% training (and validation)
n_total = len(fp_keys)
n_test = int(ceil(0.1*n_total))
n_train = int(n_total - n_test)

centroids = []
centroid_ids = []
centroids.append( fp_vals[0,:] )
centroid_ids.append( fp_keys[0] )
for i in range(1,n_train):
        diff = np.linalg.norm(fp_vals - centroids[0], axis=1)
        for j in range(1,i):
                new_diff = np.linalg.norm(fp_vals - centroids[j], axis=1)
                diff = np.minimum( diff, new_diff )
	argmax = np.argmax(diff)
        centroids.append( fp_vals[argmax,:] )
        centroid_ids.append( fp_keys[argmax] )

train_fp_vals = np.empty(n_features+1)
train_fp_keys = []
test_fp_vals = np.empty(n_features+1)
test_fp_keys = []

for i in range(n_total):
	key = fp_keys[i]
	if key in centroid_ids:
		train_fp_keys.append(key)
		train_fp_vals = np.vstack((train_fp_vals,fp_vals[i,:]))
	else:
		test_fp_keys.append(key)
		test_fp_vals = np.vstack((test_fp_vals,fp_vals[i,:]))

train_fp_vals = train_fp_vals[1:,:]
test_fp_vals = test_fp_vals[1:,:]

feature_matrices = {'training_values':train_fp_vals, 'training_labels':train_fp_keys,
        'test_values':test_fp_vals, 'test_labels':test_fp_keys}

output_file = open('data.pickle','w')
pickle.dump(feature_matrices,output_file)
output_file.close()

