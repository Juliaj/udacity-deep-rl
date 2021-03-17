"""
Module for utility methods.
"""
from datetime import datetime
import logging
import json
import yaml

import torch
import hyperparams as hp
from visualizer import VisdomWriter

logger = logging.getLogger(__name__)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
logger.debug(f'Device Info:{device}')


def soft_update(source_model, target_model, tau: float):
    """Soft update model parameters
    theta_target = tau * theta_local + (1 - tau)*theta_target

    Params
    =====
        source_model: model that weights will be copied from
        target_model: model that weights will be copied to
        tau: soft update factor
    """
    for target_param, source_param in zip(target_model.parameters(),
                                          source_model.parameters()):
        target_param.data.copy_(tau * source_param.data +
                                (1 - tau) * target_param.data)


def hard_update(source_model, target_model):
    """Update parameters of targart model to be the same as that of source model
    Params
    =====
        source_model: model that weights will be copied from
        target_model: model that weights will be copied to
    """
    for target_param, source_param in zip(target_model.parameters(),
                                          source_model.parameters()):
        target_param.data.copy_(source_param.data)


def path_gen():
    """Generate a folder path with current date time
    """
    now = datetime.now().strftime('%Y-%m-%d-%H%M%S')
    return f'runs/{now}'


def load_from_yaml(fname):
    hyperparams = None
    with open(fname, 'r') as f:
        hyperparams = yaml.safe_load(f)
    return hyperparams


def save_to_yaml(dict_item, fname):
    with open(fname, 'w') as f:
        yaml.dump(dict_item, f, default_flow_style=False)


# Credits https://github.com/katnoria/unityml-tennis
def flatten(tensor):
    """Reshape tensor to keep first dimension
    """
    return torch.reshape(tensor, (
        tensor.shape[0],
        -1,
    ))


# Credits https://github.com/katnoria/unityml-tennis
def save_to_txt(item, fname):
    with open(fname, 'a') as f:
        f.write('{}\n'.format(item))


# Credits https://github.com/katnoria/unityml-tennis
class VisWriter:
    """Wrapper of Visdom Writer
    Params
    =====
    vis: False when running without Visdom
    """
    def __init__(self, vis: bool = False):
        self.vis = vis
        if self.vis:
            self.writer = VisdomWriter(enabled=True, logger=logger)

    def text(self, message, title):
        if self.vis:
            self.writer.text(message, title)

    def push(self, item, title):
        if self.vis:
            self.writer.push(item, title)
