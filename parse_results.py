import sqlite3

conn = sqlite3.connect('data.db')
c = conn.cursor()
#c.execute("SELECT id, featuresubsetid, model_type, RMSE_train, RMSE_test FROM experiment ORDER BY RMSE_test")
c.execute("SELECT id, featuresubsetid, model_type, RMSE_train, RMSE_test FROM experiment WHERE featuresubsetid>=61 ORDER BY RMSE_test")

header = 'exp id\t fsid\t model\t RMSE_train (eV)\t RMSE_test (eV)'
print header

for i in range(16):
        line = c.fetchone()
        print '%s\t %s\t %s\t %3.2f\t\t\t %3.2f'%line

