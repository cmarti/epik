from os.path import join, abspath, dirname

VERSION = '0.1.0'

BASE_DIR = abspath(join(dirname(__file__)))
BIN_DIR = join(BASE_DIR, 'bin')

KERNELS = ['Additive', 'Pairwise', 'VC', 
           'Exponential', 'Connectedness', 'Jenga', 
           'GeneralProduct']
