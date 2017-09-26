#!/opt/rh/python27/root/usr/bin/python
#SBATCH -p iric,normal,owners
#SBATCH --exclusive
#SBATCH --output=job.out
#SBATCH --error=job.err
#SBATCH --time=0-01:00
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=4000
#SBATCH --ntasks-per-node=16
#SBATCH --mail-user=doylead@stanford.edu
#SBATCH --mail-type=END

from shutil import copyfile
from os import mkdir,chdir,getcwd
from os.path import isdir
from subprocess import call
from glob import glob
import sys

cwd = getcwd()
fps = glob('../../fp*')
fps.sort()

files = glob('rf_v*.py')
files.sort()

for fp_number in fps:
        base_dir = cwd+'/'+fp_number
        chdir(base_dir)

        for f in files:
                model_number = f.rstrip('.py')
                this_dir = base_dir+'/'+model_number
                if not isdir(this_dir):
                        print this_dir
                        sys.stdout.flush()
                        mkdir(this_dir)
                        chdir(this_dir)
                        copyfile(cwd+'/'+f,f)
                        command = ['python',f]
                        call(command)
