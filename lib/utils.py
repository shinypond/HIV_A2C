import os
import yaml
import shutil
import argparse
import torch


def load_ckpt(ckpt_dir, info, device, train=True, ckpt_num=None):
    if not os.path.exists(ckpt_dir):
        return info
    file = 'ckpt.pt' if ckpt_num is None else f'ckpt_{ckpt_num}.pt'
    loaded_info = torch.load(
        os.path.join(ckpt_dir, file),
        map_location=device,
    )
    info['net'].load_state_dict(loaded_info['net'])
    info['episode'] = loaded_info['episode']
    if train:
        info['optim'].load_state_dict(loaded_info['optim'])
        info['sched'].load_state_dict(loaded_info['sched'])
    return info

        
def save_ckpt(ckpt_dir, info, archive=False):
    saved_info = {
        'net': info['net'].state_dict(),
        'optim': info['optim'].state_dict(),
        'sched': info['sched'].state_dict(),
        'episode': info['episode'],
    }
    if archive:
        ckpt = os.path.join(ckpt_dir, f'ckpt_{info["episode"]}.pt')
    else:
        ckpt = os.path.join(ckpt_dir, 'ckpt.pt')
    torch.save(saved_info, ckpt)


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def load_config(config_path):
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return dict2namespace(config_dict)


def load_and_copy_config(config_path, copy_path):
    config = load_config(config_path)
    try:
        shutil.copy(config_path, copy_path)
    except shutil.SameFileError:
        pass
    return config


def str2bool(x):
    if isinstance(x, bool):
        return x
    else:
        assert isinstance(x, str)
        if x.lower() in ['yes', 'true', 't', 'y', '1']:
            return True
        elif x.lower() in ['no', 'false', 'f', 'n', '0']:
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

