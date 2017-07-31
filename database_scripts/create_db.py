from database import sqlexecute

dbpath = '../data.db'
commands = [
	"CREATE TABLE experiment (id integer primary key, created numeric, createdby varchar, lastmodified numeric, lastmodifiedby varchar, deleted numeric, deletedby varchar, featuresubsetid integer, model_type varchar, RMSE numeric, architecture varchar, activation_function varchar, learning_rate numeric, regularization numeric, foreign key(featuresubsetid) references feature_featuresubset(featuresubsetid))",
	"CREATE TABLE feature (id integer primary key, created numeric, createdby varchar, lastmodified numeric, lastmodifiedby varchar, deleted numeric, deletedby varchar, name varchar)",
	"CREATE TABLE feature_featuresubset (id integer primary key, created numeric, createdby varchar, lastmodified numeric, lastmodifiedby varchar, deleted numeric, deletedby varchar, featureid integer, featuresubsetid integer, foreign key(featureid) references feature(id))"
]

for command in commands:
	sqlexecute(dbpath, command)
