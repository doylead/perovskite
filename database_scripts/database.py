import sqlite3
import time
import os
import sys
from getpass import getuser

def sqlexecute(dbpath, sqlcommand, binds = []):
	connection = sqlite3.connect(dbpath)
	cursor = connection.cursor()
	cursor.execute(sqlcommand, binds)
	if sqlcommand.lower()[:6] == 'select' or sqlcommand.lower()[:4] == 'with':
		output_array = cursor.fetchall()
		connection.close()
		return output_array
	else:
		connection.commit()
		connection.close()
		return


def insert_feature_featuresubset(featuresubsetid, featureid, dbpath = 'data.db'):
	query_out = sqlexecute(dbpath, "select ID from feature_featuresubset where featuresubsetid = ? and featureid = ?", [featuresubsetid, featureid])
	L = len(query_out)
	if L == 0:
		created = time.time()
		createdby = getuser()
		connection = sqlite3.connect(dbpath)
		cursor = connection.cursor()
		cursor.execute('insert into feature_featuresubset(created, createdby, featuresubsetid, featureid) values(?,?,?,?)', [created, createdby, featuresubsetid, featureid])
		ID = cursor.lastrowid
		connection.commit()
		connection.close()
		status = 'inserted'
	elif L == 1:
		ID = query_out[0][0]
		status = 'already existed'
	else:
		sys.exit("Multiple rows found in feature_featuresubset with featuresubsetid " + str(featuresubsetid) + " and featureid " + str(featureid))
	return ID, status


def insert_feature(name, dbpath = 'data.db'):
	query_out = sqlexecute(dbpath, "select ID from feature where name = ?", [name])
	L = len(query_out)
	if L == 0:
		created = time.time()
		createdby = getuser()
		connection = sqlite3.connect(dbpath)
		cursor = connection.cursor()
		cursor.execute('insert into feature(created, createdby, name) values(?,?,?)', [created, createdby, name])
		ID = cursor.lastrowid
		connection.commit()
		connection.close()
		status = 'inserted'
	elif L == 1:
		ID = query_out[0][0]
		status = 'already existed'
	else:
		sys.exit("Multiple features found with name " + name)
	return ID, status


def insert_experiment(featuresubsetid, model_type, RMSE_train = None, RMSE_test = None ,architecture = None, activation_function = None, learning_rate = None, regularization = None, dbpath = 'data.db'):
	query_out = sqlexecute(dbpath, "select ID from experiment where featuresubsetid = ? and model_type = ? and architecture = ? and activation_function = ? and learning_rate = ? and regularization = ?", [featuresubsetid, model_type, architecture, activation_function, learning_rate, regularization])
	L = len(query_out)
	if L == 0:
		created = time.time()
		createdby = getuser()
		connection = sqlite3.connect(dbpath)
		cursor = connection.cursor()
		cursor.execute('insert into experiment(created, createdby, featuresubsetid, model_type, RMSE_train, RMSE_test, architecture, activation_function, learning_rate, regularization) values(?,?,?,?,?,?,?,?,?,?)', [created, createdby, featuresubsetid, model_type, RMSE_train, RMSE_test, architecture, activation_function, learning_rate, regularization])
		ID = cursor.lastrowid
		connection.commit()
		connection.close()
		status = 'inserted'
	elif L == 1:
		ID = query_out[0][0]
		status = 'already existed'
	else:
		sys.exit("Multiple experiments found with featuresubsetid = " + str(featuresubsetid) + ", model_type = " + model_type + ", architecture = " + str(architecture) + ", activation_function = " + activation_function + ", learning_rate = " + str(learning_rate) + ", and regularization = " + str(regularization))
	return ID, status


def list_to_sql_set_string(input_list):
	return str(input_list).replace('[','(').replace(']',')')

'''
def query_out_to_set(query_out):
	returnset = set()
	for tup in query_out:
		try:
			returnset.add(int(tup[0]))
		except:
			returnset.add(str(tup[0]))
		
	return returnset
'''


def features_exist(feature_names, dbpath = 'data.db'):
	for name in feature_names:
		query_out = sqlexecute(dbpath, 'select case when exists(select 1 from feature where name = ?) then 1 else 0 end',[name])
		if int(query_out[0][0]) == 0:
			return False
	return True



def feature_names_to_ID_set(feature_names, dbpath = 'data.db'):
	### Input:
	# feature_names: A list of feature names
	# dbpath: string: a path to the database
	### Output
	# A set of IDs corresponding to those feature names
	if features_exist(feature_names, dbpath) == False:
		sys.exit('not all feature names in list exist')
	feature_names = list_to_sql_set_string(feature_names)
	query_out = sqlexecute(dbpath, "select id from feature where name in " + feature_names)
	returnset = set()
	for tup in query_out:
		returnset.add(int(tup[0]))
	return returnset
	

def get_existing_feature_sets(dbpath = 'data.db'):
	### Input:
	# dbpath: string: a path to the database
	### Output
	# List of sets: contains all feature subsets that already exist
	query_out = sqlexecute(dbpath, "select featuresubsetid, group_concat(featureid) from feature_featuresubset group by featuresubsetid")
	IDs = []
	feature_sets = []
	for tup in query_out:
		IDs.append(int(tup[0]))
		tempset = set()
		for ID in tup[1].split(','):
			tempset.add(int(ID))
		feature_sets.append(tempset)
	return IDs, feature_sets
	


def does_feature_subset_already_exist(feature_names, dbpath = 'data.db'):
	### Input:
	# feature_names: A list of feature names
	# dbpath: string: a path to the database
	### Output
	# Boolean: True if the feature subset already exists, False if not
	if features_exist(feature_names, dbpath) == False:
		return False, None
	feature_set = feature_names_to_ID_set(feature_names, dbpath)
	featuresubsetids, existing_feature_sets = get_existing_feature_sets(dbpath)
	if feature_set in existing_feature_sets:
		featuresubsetid = featuresubsetids[existing_feature_sets.index(feature_set)]
		return True, featuresubsetid
	else:
		return False, None


def add_features(feature_names, dbpath = 'data.db', verbose = False):
	### Input:
	# feature_names: A list of feature names
	### Output
	# None
	### Side Effects
	# Adds any new features to the feature table
	# Includes data validation that won't let you add non-strings as feature names and won't insert dupicate feature names.
	# If verbose, prints out what it's done
	already_existed = []
	inserted = []
	if not type(feature_names) == type(list()):
		sys.exit('feature_names not a list:' + str(feature_names))
	for feature_name in feature_names:
		if not type(feature_name) == type(str()):
			sys.exit('feature name not a string:' + str(feature_name))
		ID, status = insert_feature(feature_name)
		if status == 'already existed':
			already_existed.append(ID)
		elif status == 'inserted':
			inserted.append(ID)
		else:
			sys.exit('got unexpected status string from insert_feature: ' + status)
	if verbose:
		print 'inserted ' + str(len(inserted)) + ' features: ' + str(inserted)
		print str(len(already_existed)) + ' features already existed: ' + str(already_existed)
	return


def add_feature_set(feature_names, dbpath = 'data.db', verbose = False):
	### Input:
	# feature_names: A list of feature names
	### Output
	# A tuple of 2 items
	#	1: an int: the ID of the feature subset
	#	2: a string: 'inserted' if the feature subset was newly created, and 'already existed' if the feature subset already existed
	### Side Effects
	# If the feature subset does not already exist: adds it to the database
	already_existed, featuresubsetid = does_feature_subset_already_exist(feature_names, dbpath = 'data.db')
	if already_existed:
		return (featuresubsetid, 'already existed')
	else:
		add_features(feature_names, dbpath, verbose)
		max_fsid = sqlexecute(dbpath, "select max(featuresubsetid) from feature_featuresubset")[0][0]
		if max_fsid:
			featuresubsetid = int(max_fsid) + 1
		else:
			featuresubsetid = 1
#		featuresubsetid = int(sqlexecute(dbpath, "select max(featuresubsetid) from feature_featuresubset")[0][0]) + 1
		featureids = sorted(feature_names_to_ID_set(feature_names, dbpath))
		for featureid in featureids:
			ID, status = insert_feature_featuresubset(featuresubsetid, featureid, dbpath)
		return (featuresubsetid, 'inserted')














































