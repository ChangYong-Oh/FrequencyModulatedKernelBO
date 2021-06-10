import os
import socket

# if this value is used in randint to generate seed than upper is exclusive 2 ** 31 is allowed
# for simplicity below is used in randint and thus generated random seed is between 0 and 2 ** 31 - 2 (both inclusive)
# if a seed value larger than 2 ** 31 (inclusive) is used then depending on machines
# np.random.RandomState(seed).randint(0, MAX_RANDOM_SEED) works differently
# due to the machine dependency of Mersenne Twister
MAX_RANDOM_SEED = 2 ** 31 - 1
FILENAME_ZFILL_SIZE = 4


def exp_dir_root():
    hostname = socket.gethostname()
    if hostname == 'Example':
        exp_dir_name = '/home/username/Experiments/FrequencyModulatedKernelBO'
    else:
        raise NotImplementedError
    if not os.path.exists(exp_dir_name):
        os.makedirs(exp_dir_name)
    return exp_dir_name


def data_dir_root():
    hostname = socket.gethostname()
    if hostname == 'Example':
        data_dir_name = '/home/username/Data'
    else:
        raise NotImplementedError
    if not os.path.exists(data_dir_name):
        os.makedirs(data_dir_name)
    return data_dir_name
