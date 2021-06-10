
from typing import Dict, Optional

import os
import urllib.request
from enum import Enum

import numpy as np
import pandas as pd


UCI_DATA_DIR = os.path.join(os.path.split(os.path.abspath(__file__))[0], 'uci_data')


class DataType(Enum):
    Real = 0
    Nominal = 1
    TargetBinary = 2
    TargetReal = 3


def _download_file(url):
    filename = os.path.join(UCI_DATA_DIR, url.split('/')[-1])
    if not os.path.exists(UCI_DATA_DIR):
        os.makedirs(UCI_DATA_DIR)
    if not os.path.exists(filename):
        urllib.request.urlretrieve(url=url, filename=filename)
    return filename


def _preprocess_dataframe(df: pd.DataFrame, attributes: Dict):
    keys_real = [key for key, value in attributes.items() if value[1] == DataType.Real]
    keys_nominal = [key for key, value in attributes.items() if value[1] == DataType.Nominal]
    keys_target = [key for key, value in attributes.items() if value[1] in [DataType.TargetBinary, DataType.TargetReal]]

    n_data = len(df)

    df_real = df.loc[:, keys_real]
    df_nominal = df.loc[:, keys_nominal]
    df_target = df.loc[:, keys_target]

    np_real = df_real.to_numpy()
    np_nominal_onehot = np.empty((n_data, 0))
    np_nominal_integer = np.empty((n_data, 0))
    np_nominal_num_cat = []
    for i, col in enumerate(df_nominal.columns):
        categories = np.sort(np.unique(df_nominal[col]))
        one_hot = np.repeat(categories.reshape((1, -1)), n_data, axis=0) == df_nominal[col].to_numpy().reshape((-1, 1))
        np_nominal_onehot = np.hstack((np_nominal_onehot, one_hot))
        np_nominal_integer = np.hstack((np_nominal_integer, np.where(one_hot)[1].reshape((-1, 1))))
        np_nominal_num_cat.append(categories.size)
    np_target = np.empty((n_data, 0))
    for i, col in enumerate(df_target.columns):
        if attributes[col][1] == DataType.TargetBinary:
            np_target = np.hstack((np_target, (df_target[col] == df_target[col].iloc[0]).to_numpy().reshape((-1, 1))))
        else:
            np_target = np.hstack((np_target, df_target[col].to_numpy().reshape((-1, 1))))

    return np_real.astype(np.float32), \
           np_nominal_integer.astype(np.float32), \
           np_nominal_onehot.astype(np.float32), \
           np_nominal_num_cat, \
           np_target.astype(np.float32)



def _load_cylinder_bands():
    data_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/cylinder-bands/bands.data'
    data_filename = _download_file(data_url)
    attribute_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/cylinder-bands/bands.names'
    attribute_filename = _download_file(attribute_url)
    with open(attribute_filename, 'rt') as f:
        attribute_text = f.read()
        attribute_text = attribute_text.split('7. Attribute Information:\n')[1].split('8. Missing Attribute Values')[0]

    attributes = {
        1: ('timestamp', DataType.Real),
        2: ('cylinder_number', DataType.Nominal),
        3: ('customer', DataType.Nominal),
        4: ('job_number', DataType.Nominal),
        5: ('grain screened', DataType.Nominal), #######
        6: ('ink_color', DataType.Nominal),
        7: ('proof_on_coated_ink', DataType.Nominal), #######
        8: ('blade_manufacturer', DataType.Nominal), #######
        9: ('cylinder_division', DataType.Nominal), #######
        10: ('paper_type',DataType.Nominal), #######
        11: ('ink_type', DataType.Nominal), #######
        12: ('direct_steam', DataType.Nominal), #######
        13: ('solvent_type', DataType.Nominal), #######
        14: ('type_on_cylinder', DataType.Nominal), #######
        15: ('press_type', DataType.Nominal),
        16: ('press', DataType.Nominal), #######
        17: ('unit_number', DataType.Real), #######
        18: ('cylinder_size', DataType.Nominal), #######
        19: ('paper_mill_location', DataType.Nominal),
        20: ('plating_tank', DataType.Nominal), #######
        21: ('proof_press_cut', DataType.Real), #######
        22: ('viscosity', DataType.Real), #######
        23: ('caliper', DataType.Real),
        24: ('ink_temperature', DataType.Real), #######
        25: ('humidity', DataType.Real), #######
        26: ('roughness', DataType.Real), #######
        27: ('blade_pressure', DataType.Real), #######
        28: ('varnish_percentage', DataType.Real), #######
        29: ('press_speed', DataType.Real), #######
        30: ('ink_percentage', DataType.Real), #######
        31: ('solvent_percentage', DataType.Real), #######
        32: ('ESA_Voltage', DataType.Real), #######
        33: ('ESA_Amperage', DataType.Real), #######
        34: ('wax', DataType.Real), #######
        35: ('hardener', DataType.Real), #######
        36: ('roller_durometer', DataType.Real),
        37: ('current_density', DataType.Real), #######
        38: ('anode_space_ratio', DataType.Real), #######
        39: ('chrome_content', DataType.Real), #######
        40: ('band', DataType.TargetBinary)
    }

    df = pd.read_csv(data_filename, names=list(attributes.keys()))
    # To remove attributes not mentioned in the paper in UCI page
    for col in [1, 2, 3, 4, 6, 15, 19, 23, 36]:
        df = df.drop(columns=col)
        del attributes[col]
    for col in [9, 10, 11, 12, 14, 18]:
        df[col] = df[col].apply(lambda x: str(x).upper())
    nonnan_index = np.all(~df.isna(), axis=1)
    df = df.iloc[nonnan_index.values].reset_index(drop=True)
    nonnan_index = np.all(df != '?', axis=1)
    df = df.iloc[nonnan_index.values].reset_index(drop=True)

    df = df.astype({key: 'float' for key, value in attributes.items() if value[1] == DataType.Real})

    return df, attributes


def _load_credit_approval():
    data_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data'
    data_filename = _download_file(data_url)
    attribute_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.names'
    attribute_filename = _download_file(attribute_url)
    with open(attribute_filename, 'rt') as f:
        attribute_text = f.read()
        attribute_text = attribute_text.split('7.  Attribute Information:\n\n'
                                              )[1].split('8.  Missing Attribute Values:')[0]

    attributes = {
        1: ('A1', DataType.Nominal),
        2: ('A2', DataType.Real),
        3: ('A3', DataType.Real),
        4: ('A4', DataType.Nominal),
        5: ('A5', DataType.Nominal),
        6: ('A6', DataType.Nominal),
        7: ('A7', DataType.Nominal),
        8: ('A8', DataType.Real),
        9: ('A9', DataType.Nominal),
        10: ('A10', DataType.Nominal),
        11: ('A11', DataType.Real),
        12: ('A12', DataType.Nominal),
        13: ('A13', DataType.Nominal),
        14: ('A14', DataType.Real),
        15: ('A15', DataType.Real),
        16: ('A16', DataType.TargetBinary)
    }

    df = pd.read_csv(data_filename, names=list(attributes.keys()))
    nonnan_index = np.all(df != '?', axis=1)
    df = df.iloc[nonnan_index.values].reset_index(drop=True)

    df = df.astype({key: 'float' for key, value in attributes.items() if value[1] == DataType.Real})

    return df, attributes


def _load_statlog_austrailian_credit():
    data_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/australian/australian.dat'
    data_filename = _download_file(data_url)
    attribute_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/australian/australian.doc'
    attribute_filename = _download_file(attribute_url)
    with open(attribute_filename, 'rt') as f:
        attribute_text = f.read()
        attribute_text = attribute_text.split('7.  Attribute Information:')[1].split('8.  Missing Attribute Values:')[0]

    attributes = {
        1: ('A1', DataType.Nominal),
        2: ('A2', DataType.Real),
        3: ('A3', DataType.Real),
        4: ('A4', DataType.Nominal),
        5: ('A5', DataType.Nominal),
        6: ('A6', DataType.Nominal),
        7: ('A7', DataType.Real),
        8: ('A8', DataType.Nominal),
        9: ('A9', DataType.Nominal),
        10: ('A10', DataType.Real),
        11: ('A11', DataType.Nominal),
        12: ('A12', DataType.Nominal),
        13: ('A13', DataType.Real),
        14: ('A14', DataType.Real),
        15: ('A15', DataType.TargetBinary)
    }

    df = pd.read_csv(data_filename, sep=' ', names=list(attributes.keys()))
    nonnan_index = np.all(df != '?', axis=1)
    df = df.iloc[nonnan_index.values].reset_index(drop=True)

    df = df.astype({key: 'float' for key, value in attributes.items() if value[1] == DataType.Real})

    return df, attributes


def _load_statlog_heart():
    data_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart/heart.dat'
    data_filename = _download_file(data_url)
    attribute_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart/heart.doc'
    attribute_filename = _download_file(attribute_url)
    with open(attribute_filename, 'rt') as f:
        attribute_text = f.read()

    attributes = {
        1: ('age', DataType.Real),
        2: ('sex', DataType.Nominal),
        3: ('chest_pain_type', DataType.Nominal),
        4: ('resting_blood_pressure', DataType.Real),
        5: ('serum_cholestoral', DataType.Real),
        6: ('fasting_blood_sugar', DataType.Nominal),
        7: ('resting_electrocardiographic_results', DataType.Nominal),
        8: ('maximum_heart_rate', DataType.Real),
        9: ('exercise_induced_angina', DataType.Nominal),
        10: ('oldpeak', DataType.Real),
        11: ('slope', DataType.Real),
        12: ('number_of_major_vessels', DataType.Real),
        13: ('thal', DataType.Nominal),
        14: ('num', DataType.TargetBinary)
    }

    df = pd.read_csv(data_filename, sep=' ', names=list(attributes.keys()))
    nonnan_index = np.all(df != '?', axis=1)
    df = df.iloc[nonnan_index.values].reset_index(drop=True)

    df = df.astype({key: 'float' for key, value in attributes.items() if value[1] == DataType.Real})

    return df, attributes


def _load_meta_data():
    data_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/meta-data/meta.data'
    data_filename = _download_file(data_url)
    attribute_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/meta-data/meta.names'
    attribute_filename = _download_file(attribute_url)
    with open(attribute_filename, 'rt') as f:
        attribute_text = f.read()
        attribute_text = attribute_text.split('7. Attribute Information:\n')[1].split('Summary Statistics:')[0]

    attributes = {
        1: ('dataset', DataType.Nominal),
        2: ('T', DataType.Real),
        3: ('N', DataType.Real),
        4: ('p', DataType.Real),
        5: ('k', DataType.Real),
        6: ('Bin', DataType.Real),
        7: ('Cost', DataType.Nominal),
        8: ('SDratio', DataType.Real),
        9: ('correl', DataType.Real),
        10: ('cancor1', DataType.Real),
        11: ('cancor2', DataType.Real),
        12: ('fract1', DataType.Real),
        13: ('fract2', DataType.Real),
        14: ('skewness', DataType.Real),
        15: ('kurtosis', DataType.Real),
        16: ('Hc', DataType.Real),
        17: ('Hx', DataType.Real),
        18: ('MCx', DataType.Real),
        19: ('EnArt', DataType.Real),
        20: ('NSRatio', DataType.Real),
        21: ('Algorithm', DataType.Nominal),
        22: ('error', DataType.TargetReal)
    }

    df = pd.read_csv(data_filename, names=list(attributes.keys()))
    for col in [11, 13]:
        df = df.drop(columns=col)
        del attributes[col]
    nonnan_index = np.all(df != '?', axis=1)
    df = df.iloc[nonnan_index.values].reset_index(drop=True)

    df = df.astype({key: 'float' for key, value in attributes.items() if value[1] == DataType.Real})

    return df, attributes


def _load_servo():
    data_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/servo/servo.data'
    data_filename = _download_file(data_url)
    attribute_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/servo/servo.names'
    attribute_filename = _download_file(attribute_url)
    with open(attribute_filename, 'rt') as f:
        attribute_text = f.read()
        attribute_text = attribute_text.split('7. Attribute information:')[1].split('8. Missing Attribute Values')[0]

    attributes = {
        1: ('motor', DataType.Nominal),
        2: ('screw', DataType.Nominal),
        3: ('pgain', DataType.Real),
        4: ('vgain', DataType.Real),
        5: ('class', DataType.TargetReal)
    }

    df = pd.read_csv(data_filename, names=list(attributes.keys()))
    nonnan_index = np.all(df != '?', axis=1)
    df = df.iloc[nonnan_index.values].reset_index(drop=True)

    df = df.astype({key: 'float' for key, value in attributes.items() if value[1] == DataType.Real})

    return df, attributes


def _load_optical_interconnection():
    data_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00449/optical_interconnection_network.csv'
    data_filename = _download_file(data_url)

    df = pd.read_csv(data_filename, sep=';', decimal=',')
    df = df.drop(columns=['Unnamed: 10', 'Unnamed: 11', 'Unnamed: 12', 'Unnamed: 13', 'Unnamed: 14'])

    attributes = dict()
    for i, col in enumerate(list(df.columns)):
        attributes[i + 1] = [col, None]
    for i in [1, 3, 4]:
        attributes[i][1] = DataType.Nominal
    attributes[2][1] = DataType.Real
    attributes[5][1] = DataType.Real
    for i in [6, 7, 8, 9, 10]:
        attributes[i][1] = DataType.TargetReal

    df.columns = list(attributes.keys())

    # To remove target variables except for one [6, 7, 8, 9, 10]
    for col in [7, 8, 9, 10]:
        df = df.drop(columns=col)
        del attributes[col]

    nonnan_index = np.all(df != '?', axis=1)
    df = df.iloc[nonnan_index.values].reset_index(drop=True)

    df = df.astype({key: 'float' for key, value in attributes.items() if value[1] == DataType.Real})

    return df, attributes


def load_uci_data(data_type: str, random_seed: Optional[int] = None):
    if data_type == 'REG1':
        df, attributes = _load_meta_data()
    elif data_type == 'REG2':
        df, attributes = _load_servo()
    elif data_type == 'REG3':
        df, attributes = _load_optical_interconnection()
    else:
        raise NotImplementedError

    arr_real, arr_nominal_integer, arr_nominal_onehot, nominal_num_cat, arr_target = \
        _preprocess_dataframe(df, attributes)
    n_data = arr_real.shape[0]
    n_train = int(0.8 * n_data)

    inds = list(range(n_data))
    np.random.RandomState(random_seed).shuffle(inds)
    train_inds = inds[:n_train]
    test_inds = inds[n_train:]

    train_real, test_real = arr_real[train_inds], arr_real[test_inds]
    train_nominal_integer, test_nominal_integer = arr_nominal_integer[train_inds], arr_nominal_integer[test_inds]
    train_target, test_target = arr_target[train_inds], arr_target[test_inds]

    train_real_mean = np.mean(train_real, axis=0, keepdims=True)
    train_real_std = np.std(train_real, axis=0, keepdims=True)
    train_real = (train_real - train_real_mean) / train_real_std
    test_real = (test_real - train_real_mean) / train_real_std
    train_target_mean = np.mean(train_target, axis=0, keepdims=True)
    train_target_std = np.std(train_target, axis=0, keepdims=True)
    train_target = (train_target - train_target_mean) / train_target_std
    test_target = (test_target - train_target_mean) / train_target_std

    train_input = np.hstack([train_real, train_nominal_integer])
    test_input = np.hstack([test_real, test_nominal_integer])

    return (train_input, train_target), \
           (test_input, test_target), \
           nominal_num_cat


if __name__ == '__main__':
    load_uci_data('REG3')
