import os
import json

import torch


def read_smac(dirname):
    dir_list = [elm for elm in os.listdir(dirname) if 'SMAC' in elm and os.path.isdir(os.path.join(dirname, elm))]
    dir_grouped = dict()
    for elm in dir_list:
        exp_type = elm.split('_')[0]
        try:
            dir_grouped[exp_type].append(elm)
        except KeyError:
            dir_grouped[exp_type] = [elm]

    for k, v in dir_grouped.items():
        for elm in v:
            read_smac_json(os.path.join(dirname, elm, 'run_1', 'runhistory.json'))


def read_smac_json(json_filename):
    with open(json_filename, 'rt') as f:
        json_data = json.load(f)
    x_list = []
    y_list = []
    configs_keys = sorted([int(elm) for elm in json_data['configs'].keys()])
    for i in configs_keys:
        k = str(i)
        v = json_data['configs'][k]
        data_id, data_result = json_data['data'][i - 1]
        assert data_id[0] == int(k)
        assert data_result[2]['__enum__'] == 'StatusType.SUCCESS'
        y_list.append(data_result[0])
        x_list.append(torch.tensor([float(v[d]) for d in sorted(v.keys())]))
    return torch.stack(x_list), torch.tensor(y_list)


def x_str(x: torch.Tensor):
    assert x.numel() in x.size()
    x_str_list = ''
    for i in range(x.numel()):
        elm = x.view(-1)[i].item()
        x_str_list.append('%d' % int(elm) if int(elm) == elm else '%.4f' % elm)
    return ', '.join(x_str_list)
