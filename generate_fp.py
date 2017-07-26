
from mendeleev import element
from os.path import isfile
import numpy as np
import hashlib
import pickle
import time

def gen_fp(A,B,bound_ads,secondary_ads,hash_id):
	fp = []
	fp_instructions = []

	# Bulk properties will need to be loaded
	# or passed here intelligently somehow
	# - not currently implemented

# A Block
	if False: # Use A.atomic_number
		ele = element(A)
		Z = ele.atomic_number
		fp.append(Z)
		fp_instructions.append('A.atomic_number')

	if False: # Use A.atomic_radius
		ele = element(A)
		rad = ele.atomic_radius
		fp.append(rad)
		fp_instructions.append('A.atomic_radius')

	if False: # Use A.pauli_electronegativity
		ele = element(A)
		eneg = ele.en_pauling
		fp.append(eneg)
		fp_instructions.append('A.pauli_electronegativity')

	if False: # Use A.polarizability
		ele = element(A)
		polarization = ele.dipole_polarizability	
		fp.append(polarization)
		fp_instructions.append('A.dipole_polarizability')

	if False: # Use A.first_ionization_energy
		ele = element(A)
		first_ionization = ele.ionenergies[1]
		fp.append(first_ionization)
		fp_instructions.append('A.first_ionization_energy')

	if False: # Use A.period
		ele = element(A)
		period = ele.period
		fp.append(period)
		fp_instructions.append('A.period')

# B Block
	if False: # Use B.atomic_number
		ele = element(B)
		Z = ele.atomic_number
		fp.append(Z)
		fp_instructions.append('B.atomic_number')

	if False: # Use B.atomic_radius
		ele = element(B)
		rad = ele.atomic_radius
		fp.append(rad)
		fp_instructions.append('B.atomic_radius')

	if False: # Use B.pauli_electronegativity
		ele = element(B)
		eneg = ele.en_pauling
		fp.append(eneg)
		fp_instructions.append('B.pauli_electronegativity')

	if False: # Use B.polarizability
		ele = element(B)
		polarization = ele.dipole_polarizability	
		fp.append(polarization)
		fp_instructions.append('B.dipole_polarizability')

	if False: # Use B.first_ionization_energy
		ele = element(B)
		first_ionization = ele.ionenergies[1]
		fp.append(first_ionization)
		fp_instructions.append('B.first_ionization_energy')

	if False: # Use B.period
		ele = element(B)
		period = ele.period
		fp.append(period)
		fp_instructions.append('B.period')

# Bound Ads Block


# Secondary Ads Block


# Output Instructions and Return
        md_filename = 'fp_%s.md'%hash_id
        if not isfile(md_filename):
                md_file = open(md_filename,'w')
		for instruction in fp_instructions:
			print >>md_file,instruction
		md_file.close()
	return fp


def gen_hash_id():
	# Generates a unique string of length 32
	# to easily separate all attempts
	h = hashlib.md5()
	h.update(str(time.time()))
	h_id = h.hexdigest()
	return h_id

def normalize(M,hash_id,normalize_target=True):
	# Normalize a matrix based on mean and
	# variance.  May or may not include
	# target values
	if normalize_target:
		temp = M[:,-1]

	mean_vals = np.average(M,axis=0)
	std_vals = np.std(M,axis=0)
	M_p = np.divide(M-mean_vals,std_vals)

	if normalize_target:
		M[:,-1] = temp
		std_vals[-1] = 1
		mean_vals[-1] = 0

	

	return M

# Load Data
data = pickle.load(open('data_summary.pickle'))


