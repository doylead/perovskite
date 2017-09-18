import sqlite3

conn = sqlite3.connect('data.db')
c = conn.cursor()
c.execute("SELECT id, featuresubsetid, model_type, RMSE_train, RMSE_test FROM experiment ORDER BY RMSE_train")

header = 'exp id\t fsid\t model\t RMSE_train (eV)\t RMSE_test (eV)'
print header

for i in range(25):
#for i in range(40):
        line = c.fetchone()
        print '%s\t %s\t %s\t %3.2f\t\t\t %3.2f'%line

