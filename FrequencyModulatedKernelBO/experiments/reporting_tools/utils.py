import os
import dill
import pickle

import torch

from FrequencyModulatedKernelBO.experiments.config import FILENAME_ZFILL_SIZE


MODEL_LIST = ['smac', 'tpe',
              'Diffusion', 'Laplacian',
              'AdditiveDiffusion', 'AdditiveLaplacian',
              'ProductDiffusion', 'ProductLaplacian',
              'ModulatedDiffusion', 'ModulatedLaplacian',
              'CoCaBO-0.0', 'CoCaBO-0.5', 'CoCaBO-1.0']


COLOR_DICT = {
    'smac': 'tab:olive',
    'tpe': 'lawngreen',
    'AdditiveDiffusion': 'tab:green',
    'AdditiveLaplacian': 'tab:brown',
    'ProductDiffusion': 'tab:red',
    'ProductLaplacian': 'black',
    'ModulatedDiffusion': 'blue',
    'ModulatedLaplacian': 'magenta',
    'RegularizedEvolution': 'tab:cyan',
    'CoCaBO-0.0': 'navy',
    'CoCaBO-0.5': 'orangered',
    'CoCaBO-1.0': 'darkgreen',

}


def read_GP_data(dirname):
    last_ind = max([int(os.path.splitext(elm)[0].split('_')[-1])
                    for elm in os.listdir(dirname) if os.path.splitext(elm)[1] == '.pkl'])
    last_filename_suffix = '_%s.pkl' % str(last_ind).zfill(FILENAME_ZFILL_SIZE)
    last_filename = [elm for elm in os.listdir(dirname) if last_filename_suffix in elm][0]
    # with open(os.path.join(dirname, last_filename), 'rb') as f:
    #     checkpoint = pickle.load(f)
    # if isinstance(checkpoint, int):
    checkpoint = torch.load(os.path.join(dirname, last_filename))
    return checkpoint['data']