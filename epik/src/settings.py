from os.path import join, abspath, dirname

VERSION = '0.1.0'

BASE_DIR = abspath(join(dirname(__file__), '..'))
BIN_DIR = join(BASE_DIR, '..', 'bin')
TEST_DATA_DIR = join(BASE_DIR, 'test', 'data')
