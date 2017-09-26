from shutil import copyfile
from os import mkdir,chdir,getcwd
from os.path import isdir
from subprocess import call
from glob import glob

cwd = getcwd()
fp_number = 59

files = glob('gp_v*.py')
files.sort()

base_dir = cwd+'/../../fp_%05d'%fp_number
chdir(base_dir)

for f in files:
        model_number = f.rstrip('.py')
        this_dir = base_dir+'/'+model_number
        if not isdir(this_dir):
                mkdir(this_dir)
                chdir(this_dir)
                copyfile(cwd+'/'+f,f)
                command = ['sbatch',f]
                call(command)

