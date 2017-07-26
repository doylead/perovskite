
from mendeleev import element
from os import getcwd, chdir
import numpy as np
import pickle

def gen_fp_file(fp,A,B,primary_ads,secondary_ads=None):
	fp_vals = []

	# Bulk properties will need to be loaded
	# or passed here intelligently somehow
	# - not currently implemented

# A Block
	if 'A.atomic_number' in fp:
		ele = element(A)
		Z = ele.atomic_number
		fp_vals.append(Z)

	if 'A.atomic_radius' in fp:
		ele = element(A)
		rad = ele.atomic_radius
		fp_vals.append(rad)

	if 'A.pauling_electronegativity' in fp:
		ele = element(A)
		eneg = ele.en_pauling
		fp_vals.append(eneg)

	if 'A.dipole_polarizability' in fp:
		ele = element(A)
		polarization = ele.dipole_polarizability	
		fp_vals.append(polarization)

	if 'A.first_ionization_energy' in fp:
		ele = element(A)
		first_ionization = ele.ionenergies[1]
		fp_vals.append(first_ionization)

	if 'A.period' in fp:
		ele = element(A)
		period = ele.period
		fp_vals.append(period)

# B Block
	if 'B.atomic_number' in fp:
		ele = element(B)
		Z = ele.atomic_number
		fp_vals.append(Z)

	if 'B.atomic_radius' in fp:
		ele = element(B)
		rad = ele.atomic_radius
		fp_vals.append(rad)

	if 'B.pauling_electronegativity' in fp:
		ele = element(B)
		eneg = ele.en_pauling
		fp_vals.append(eneg)

	if 'B.dipole_polarizability' in fp:
		ele = element(B)
		polarization = ele.dipole_polarizability	
		fp_vals.append(polarization)

	if 'B.first_ionization_energy' in fp:
		ele = element(B)
		first_ionization = ele.ionenergies[1]
		fp_vals.append(first_ionization)

	if 'B.period' in fp:
		ele = element(B)
		period = ele.period
		fp_vals.append(period)

# Ads Block
	# Currently empty

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
newdir = 'fp_'+newdir_index # make this prettier

# Load Data
data = pickle.load(open('data_summary.pickle'))

fp_all_options = [
	'A.atomic_number', 'A.atomic_radius', 'A.pauling_electronegativity',
	'A.dipole_polarizability', 'A.first_ionization_energy', 'A.period',
	'B.atomic_number', 'B.atomic_radius', 'B.pauling_electronegativity',
	'B.dipole_polarizability', 'B.first_ionization_energy', 'B.period',
	]

this_fp = ['A.atomic_number','B.atomic_number']

# if fp has not been used before
chdir(newdir)
md_filename = 'fp_instructions.md'
md_file = open(md_filename,'w')
for instruction in this_fp:
	print >>md_file,instruction
md_file.close()

# if fp has been used before
	#print 'fp has been used before, check '+path_to_fp


