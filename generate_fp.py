
import sys
sys.path.append('database_scripts')
sys.path.append('data_files')

from database import add_feature_set
from mendeleev import element
from os import mkdir, chdir
from ase import Atoms
from math import ceil
from sys import exit
import numpy as np
import pickle
from conversions import element_to_valence_list
from copy import deepcopy


class Atomic_Properties(object):
        def __init__(self):
                self._cache = {}
        def __call__(self, atomtype):
                if atomtype not in self._cache:
                        self._cache[atomtype] = element(atomtype)
                return self._cache[atomtype]

def gen_fp(fp, # List of fingerprint types to use
        A_id, # String containing elemental abbreviation for A metal
        B_id, # String containing elemental abbreviation for B metal
        ads_id, # String containing elemental abbreviation for
                     # adsorbate atoms
        target_value, # Adsorption energy; float
        atomic_properties, # Atomic_Properties caching object
        data_passed = None,
        ):

        n_fp = len(fp)
        fp_vals = []
        all_vals = {}
        # Bulk properties will need to be loaded
        # or passed here intelligently somehow
        # - not currently implemented

# A Block
        ele = atomic_properties(A_id)
        all_vals['A.atomic_number'] = ele.atomic_number
        all_vals['A.atomic_radius'] = ele.atomic_radius
        all_vals['A.pauling_electronegativity'] = ele.en_pauling
        all_vals['A.dipole_polarizability'] = ele.dipole_polarizability	
        all_vals['A.first_ionization_energy'] = first_ionization = ele.ionenergies[1]
        all_vals['A.period'] = ele.period
        all_vals['A.electron_affinity'] = ele.electron_affinity

# B Block
        ele = atomic_properties(B_id)
        all_vals['B.atomic_number'] = ele.atomic_number
        all_vals['B.atomic_radius'] = ele.atomic_radius
        all_vals['B.pauling_electronegativity'] = ele.en_pauling
        all_vals['B.dipole_polarizability'] = ele.dipole_polarizability
        all_vals['B.first_ionization_energy'] = first_ionization = ele.ionenergies[1]
        all_vals['B.period'] = ele.period
        all_vals['B.electron_affinity'] = ele.electron_affinity

# Ads Block
	# Determine the primary adsorbate
	if ads_id in ['O', 'OH', 'OOH']:
		primary_ads_id = 'O'
	elif ads_id in ['H_M']:
		primary_ads_id = 'H'
	elif ads_id in ['N', 'NH', 'NH2', 'NNH']:
		primary_ads_id = 'N'

	# Determine the secondary adsorbates
	if ads_id in ['OH','NH']:
		secondary_ads_id = ['H']
	elif ads_id in ['OOH']:
		secondary_ads_id = ['O']
	elif ads_id in ['NH2']:
		secondary_ads_id = ['H', 'H']
	elif ads_id in ['NNH']: # Double bond b/w N=N
		secondary_ads_id = ['N', 'N']
	elif ads_id in ['O', 'H_M', 'N']:
		secondary_ads_id = []

        ele = atomic_properties(primary_ads_id)
        all_vals['primary_ads.atomic_number'] = ele.atomic_number
        all_vals['primary_ads.atomic_radius'] = ele.atomic_radius
        all_vals['primary_ads.pauling_electronegativity'] = ele.en_pauling
        all_vals['priamry_ads.dipole_polarizability'] = ele.dipole_polarizability	
        all_vals['primary_ads.first_ionization_energy'] = ele.ionenergies[1]
        all_vals['primary_ads.period'] = ele.period
        all_vals['primary_ads.electron_affinity'] = ele.electron_affinity

        if ads_id in ['H_M', 'OH', 'OOH', 'NH2', 'NNH']:
                nbw = 1
        elif ads_id in ['O', 'NH']:
                nbw = 2
        elif ads_id in ['N']:
                nbw = 3
        all_vals['primary_ads.num_bonds_wanted'] = nbw

        sum_eneg = ele.en_pauling
        for secondary_ele_id in secondary_ads_id:
                secondary_ele = atomic_properties(secondary_ele_id)
                secondary_eneg = secondary_ele.en_pauling
                sum_eneg -= secondary_eneg
        all_vals['all_ads.sum_pauling_electronegativity'] = sum_eneg

        if ads_id == 'H_M': # Ni, Pt, Ag, Ir, Au, Fe, Pd, Rh, Cu
                site, Ni_terrace_energy = data_passed['metal_ads']['Ni']['H']
                site, Pt_terrace_energy = data_passed['metal_ads']['Pt']['H']
                site, Ag_terrace_energy = data_passed['metal_ads']['Ag']['H']
                site, Ir_terrace_energy = data_passed['metal_ads']['Ir']['H']
                site, Au_terrace_energy = data_passed['metal_ads']['Au']['H']
                site, Fe_terrace_energy = data_passed['metal_ads']['Fe']['H']
                site, Pd_terrace_energy = data_passed['metal_ads']['Pd']['H']
                site, Rh_terrace_energy = data_passed['metal_ads']['Rh']['H']
                site, Cu_terrace_energy = data_passed['metal_ads']['Cu']['H']
        else:
                site, Ni_terrace_energy = data_passed['metal_ads']['Ni'][ads_id]
                site, Pt_terrace_energy = data_passed['metal_ads']['Pt'][ads_id]
                site, Ag_terrace_energy = data_passed['metal_ads']['Ag'][ads_id]
                site, Ir_terrace_energy = data_passed['metal_ads']['Ir'][ads_id]
                site, Au_terrace_energy = data_passed['metal_ads']['Au'][ads_id]
                site, Fe_terrace_energy = data_passed['metal_ads']['Fe'][ads_id]
                site, Pd_terrace_energy = data_passed['metal_ads']['Pd'][ads_id]
                site, Rh_terrace_energy = data_passed['metal_ads']['Rh'][ads_id]
                site, Cu_terrace_energy = data_passed['metal_ads']['Cu'][ads_id]
        all_vals['all_ads.Ni_terrace_binding_energy'] = Ni_terrace_energy
        all_vals['all_ads.Pt_terrace_binding_energy'] = Pt_terrace_energy
        all_vals['all_ads.Ag_terrace_binding_energy'] = Ag_terrace_energy
        all_vals['all_ads.Ir_terrace_binding_energy'] = Ir_terrace_energy
        all_vals['all_ads.Au_terrace_binding_energy'] = Au_terrace_energy
        all_vals['all_ads.Fe_terrace_binding_energy'] = Fe_terrace_energy
        all_vals['all_ads.Pd_terrace_binding_energy'] = Pd_terrace_energy
        all_vals['all_ads.Rh_terrace_binding_energy'] = Rh_terrace_energy
        all_vals['all_ads.Cu_terrace_binding_energy'] = Cu_terrace_energy

# Catalyst Properties

        all_vals['cat.lattice_constant'] = data_passed['perovskite_lc'][A_id+B_id+'O3']

# Extracts from dictionary
        for i in range(n_fp):
                this_feature = fp[i]
                fp_vals.append( all_vals[this_feature] )

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

# Load Data
data = pickle.load(open('data_files/data_summary.pickle'))

# ----- <START: Users should only modify code within this block> ----- #
exclude_ads = ['O2', 'H_O'] # Leaves 1,534 points
n_total = 0
for key in data.keys():
        all_ads = data[key].keys()
        included_ads = [ads for ads in all_ads if ads not in exclude_ads]
	n_total += len(included_ads)

if True:
        train_frac = 0.90 # Train on 90% of the data
        n_train = int(ceil(train_frac * n_total))
        n_test = int(n_total - n_train)
else:
        n_train = 1 # User-specified
        n_train = int(n_train)
        n_test = int(n_total - n_train)

# Provided for easy reference, never called within script
fp_all_options = [
        'A.atomic_number', 'A.atomic_radius', 'A.pauling_electronegativity',
        'A.dipole_polarizability', 'A.first_ionization_energy', 'A.period',
        'A.electron_affinity', # Not defined for all elements, avoid
        'B.atomic_number', 'B.atomic_radius', 'B.pauling_electronegativity',
        'B.dipole_polarizability', 'B.first_ionization_energy', 'B.period',
        'B.electron_affinity', # Not defined for all elements, avoid
        # primary_ads refers only to the adsorbate atom bound to the surface
        'primary_ads.atomic_number', 'primary_ads.atomic_radius',
        'primary_ads.pauling_electronegativity','primary_ads.dipole_polarizability',
        'primary_ads.first_ionization_energy', 'primary_ads.period',
        'primary_ads.electron_affinity', 'primary_ads.num_bonds_wanted',
        # all_ads refer to all adsorbate atoms
        'all_ads.sum_pauling_electronegativity',
        'all_ads.Pt_terrace_binding_energy',
        # cat refers to catalyst properties
        'cat.lattice_constant',
	]

this_fp = [ # So far have all values for 1x(A.X),1x(B.X),all_ads.Pt_terrace_binding_energy
        'A.pauling_electronegativity',
        'A.atomic_radius',
        #'A.dipole_polarizability',
        #'A.first_ionization_energy',
        'B.pauling_electronegativity',
        'B.atomic_radius',
        'B.dipole_polarizability',
        #'B.first_ionization_energy',
        #'primary_ads.pauling_electronegativity',
        #'primary_ads.num_bonds_wanted',
        #'all_ads.sum_pauling_electronegativity',
        #'all_ads.Ni_terrace_binding_energy',
        'all_ads.Pt_terrace_binding_energy',
        #'all_ads.Ag_terrace_binding_energy',
        #'all_ads.Ir_terrace_binding_energy',
        #'all_ads.Au_terrace_binding_energy',
        #'all_ads.Fe_terrace_binding_energy',
        #'all_ads.Pd_terrace_binding_energy',
        #'all_ads.Rh_terrace_binding_energy',
        'all_ads.Cu_terrace_binding_energy',
        #'cat.lattice_constant',
        'train.n=%d'%n_train,
        ]


# ------ <END: Users should only modify code within this block> ------ #


ID,status = add_feature_set(this_fp,'data.db')

if status=='already existed':
        print 'Feature set already exists, check %05f'%ID
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
this_fp.remove('train.n=%d'%n_train) # Because train.n is not used in regression

n_features = len(this_fp)
fp_vals = np.empty(n_features+1)
fp_keys = []
atomic_properties = Atomic_Properties()

metal_ads = pickle.load(open('../data_files/min_elec_energies_fcc.pkl','rb'))
perovskite_lc = pickle.load(open('../data_files/lc_1x1.pckl','rb'))
data_passed = {'metal_ads': metal_ads,
                'perovskite_lc': perovskite_lc}

for key in data.keys():
        atoms_object = Atoms(key)
        A_id = atoms_object[0].symbol
        B_id = atoms_object[1].symbol
        for ads_id in data[key].keys():
                if ads_id not in exclude_ads:
                        this_target = data[key][ads_id]
                        this_fp_val = gen_fp(fp=this_fp,
                                A_id=A_id,
                                B_id=B_id,
                                ads_id=ads_id,
                                target_value=this_target,
                                atomic_properties=atomic_properties,
                                data_passed=data_passed)
                        fp_vals = np.vstack((fp_vals, this_fp_val))
			fp_keys.append('%s, %s ads'%(key,ads_id))

fp_vals = fp_vals[1:,:]
fp_vals = normalize(fp_vals)

# Distributes the training and test sets
centroids = []
centroid_ids = []
centroids.append( fp_vals[0,:] )
centroid_ids.append( fp_keys[0] )
diff = np.linalg.norm(fp_vals - centroids[0], axis=1)
for i in range(1,n_train):
        argmax = np.argmax(diff)
        centroids.append( fp_vals[argmax,:] )
        centroid_ids.append( fp_keys[argmax] )
        new_diff = np.linalg.norm(fp_vals - centroids[i], axis=1)
        diff = np.minimum(diff, new_diff)

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

train_targets = train_fp_vals[1:,-1]
test_targets = test_fp_vals[1:,-1]

train_fp_vals = train_fp_vals[1:,:-1]
test_fp_vals = test_fp_vals[1:,:-1]

feature_matrices = {'train_values':train_fp_vals, 'train_labels':train_fp_keys,
        'test_values':test_fp_vals, 'test_labels':test_fp_keys,
        'train_targets': train_targets, 'test_targets': test_targets}

output_file = open('data.pickle','w')
pickle.dump(feature_matrices,output_file)
output_file.close()


