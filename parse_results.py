import sqlite3

conn = sqlite3.connect('data.db')
c = conn.cursor()
c.execute("SELECT id, featuresubsetid, model_type, RMSE FROM experiment ORDER BY RMSE")

header = 'id\t fsid\t model\t RMSE (eV)'
print header

for line in c.fetchall():  # Consier changing to range(n) later
        print '%s\t %s\t %s\t %3.2f'%line

