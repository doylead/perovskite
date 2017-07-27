from database import *

dbpath = 'data.db'

feature_names = ['test1','test3']
feature_names2 = ['test1','test2']

# print add_feature_set(feature_names, dbpath = dbpath, verbose = True)
a = sqlexecute(dbpath, "select * from feature_featuresubset order by featuresubsetid")
for i in a:
	print i