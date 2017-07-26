
from os import getcwd, mkdir, chdir
from mendeleev import element
from ase import Atoms
import numpy as np
import pickle

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
	elif ads_id in ['NH2']
		secondary_ads_id = ['H', 'H']
	elif ads_id in ['NNH']
		secondary_ads_id = ['N']
	elif ads_id in ['O', 'H_M', 'N']:
		secondary_ads_id = None # This should probably be updated

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

# Return
	return fp

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
                M[:,-1] = temp
                std_vals[-1] = 1
                mean_vals[-1] = 0

        output_file = open('normalization_parameters.pickle','w')
        output = {'mean_vals': mean_vals, 'std_vals': std_vals}
        pickle.dump(output_file,output)
        output_file.close()

        return M

# Set up environment
cwd = getcwd()
newdir_index = 1 # Set to minimum unused integer based on SQL ids
newdir = 'fp_%05d'%newdir_index

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
	'all_ads.sum_pauling_enegativity'
	]

this_fp = ['A.atomic_number','B.atomic_number']

# if fp has not been used before
#mkdir(newdir)
chdir(newdir)
md_filename = 'fp_instructions.md'
md_file = open(md_filename,'w')
for instruction in this_fp:
        print >>md_file,instruction
md_file.close()

fp_vals = np.array()
for key in data.keys():
        atoms_object = Atoms(key)
        A_id = atoms_object[0].symbol
        B_id = atoms_object[1].symbol
	for ads_key in data[key].keys():
		this_fp_val = gen_fp(this_fp, A_id, B_id,
			
        fp_vals = np.vstack(fp_vals,
		

# if fp has been used before
        #print 'fp has been used before, check '+path_to_fp


