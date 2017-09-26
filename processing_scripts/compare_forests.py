from subprocess import check_output
import pylab as plt
import pickle

max_feature_set = range(1,9)
this_fp = 59

ntrees = []
rmse_test = []
rmse_train = []

for i in max_feature_set:
        command = ['grep','n_estimators','../fp_%05d/rf_v%d/rf_v%d.py'%(this_fp,i,i)]
        lines = check_output(command).split('\n')
        ntrees.append( int(lines[0].split('=')[1]) )
        
        standard_output_pickle = '../fp_%05d/rf_v%d/standard_output.pickle'%(this_fp,i)
        standard_output_data = pickle.load(open(standard_output_pickle))
        rmse_test.append(standard_output_data['RMSE_test'])
        rmse_train.append(standard_output_data['RMSE_train'])


plt.plot(ntrees,rmse_test,lw=0,ms=12,marker='o',color='orange',label='Test Set')
plt.plot(ntrees,rmse_train,lw=0,ms=12,marker='o',color='blue',label='Train Set')
plt.legend(numpoints=1)
plt.xlabel('Number of Estimators',size=18)
plt.ylabel('RMSE (eV)',size=18)

plt.show()

