# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import os
import argparse
import glob
import logging
import numpy as np
import matplotlib.pyplot as plt
import data_loader as loaders
import data_collate as collates
import json
from model import GradTTS, GradTTSXvector

import torch


def intersperse(lst, item):
    # Adds blank symbol
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def parse_filelist(filelist_path, split_char="|"):
    with open(filelist_path, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split_char) for line in f]
    return filepaths_and_text


def latest_checkpoint_path(dir_path, regex="grad_*.pt"):
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    x = f_list[-1]
    return x


def load_checkpoint(logdir, model, num=None):
    if num is None:
        model_path = latest_checkpoint_path(logdir, regex="grad_*.pt")
    else:
        model_path = os.path.join(logdir, f"grad_{num}.pt")
    print(f'Loading checkpoint {model_path}...')
    model_dict = torch.load(model_path, map_location=lambda loc, storage: loc)
    model.load_state_dict(model_dict, strict=False)
    return model


def save_figure_to_numpy(fig):
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def plot_tensor(tensor):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(tensor, aspect="auto", origin="lower", interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def save_plot(tensor, savepath):
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(tensor, aspect="auto", origin="lower", interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.canvas.draw()
    plt.savefig(savepath)
    plt.close()
    return


def get_correct_class(hps, train=True):
    if train:
        if hps.xvector and hps.pe:
            raise NotImplementedError
        elif hps.xvector:  # no pitch energy
            loader = loaders.XvectorLoader
            collate = collates.XvectorCollate
            model = GradTTSXvector
            dataset = loader(utts=hps.data.train_utts,
                             hparams=hps.data,
                             feats_scp=hps.data.train_feats_scp,
                             utt2phns=hps.data.train_utt2phns,
                             phn2id=hps.data.phn2id,
                             utt2phn_duration=hps.data.train_utt2phn_duration,
                             spk_xvector_scp=hps.data.train_spk_xvector_scp,
                             utt2spk_name=hps.data.train_utt2spk)
        elif hps.pe:
            raise NotImplementedError
        else:  # no PE, no xvector
            loader = loaders.SpkIDLoader
            collate = collates.SpkIDCollate
            model = GradTTS
            dataset = loader(utts=hps.data.train_utts,
                             hparams=hps.data,
                             feats_scp=hps.data.train_feats_scp,
                             utt2phns=hps.data.train_utt2phns,
                             phn2id=hps.data.phn2id,
                             utt2phn_duration=hps.data.train_utt2phn_duration,
                             utt2spk=hps.data.train_utt2spk)
    else:
        if hps.xvector and hps.pe:
            raise NotImplementedError
        elif hps.xvector:
            loader = loaders.XvectorLoader
            collate = collates.XvectorCollate
            model = GradTTSXvector
            dataset = loader(utts=hps.data.val_utts,
                             hparams=hps.data,
                             feats_scp=hps.data.val_feats_scp,
                             utt2phns=hps.data.val_utt2phns,
                             phn2id=hps.data.phn2id,
                             utt2phn_duration=hps.data.val_utt2phn_duration,
                             spk_xvector_scp=hps.data.val_spk_xvector_scp,
                             utt2spk_name=hps.data.val_utt2spk)
        elif hps.pe:
            raise NotImplementedError
        else:  # no PE, no xvector
            loader = loaders.SpkIDLoader
            collate = collates.SpkIDCollate
            model = GradTTS
            dataset = loader(utts=hps.data.val_utts,
                             hparams=hps.data,
                             feats_scp=hps.data.val_feats_scp,
                             utt2phns=hps.data.val_utt2phns,
                             phn2id=hps.data.phn2id,
                             utt2phn_duration=hps.data.val_utt2phn_duration,
                             utt2spk=hps.data.val_utt2spk)
    return dataset, collate(), model


def get_hparams(init=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default="./configs/base.json",
                        help='JSON file for configuration')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='Model name')
    parser.add_argument('-s', '--seed', type=int, default=1234)
    parser.add_argument('--not-pretrained', action='store_true', help='if set to true, then train from scratch')

    args = parser.parse_args()
    model_dir = os.path.join("./logs", args.model)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    config_path = args.config
    config_save_path = os.path.join(model_dir, "config.json")
    if init:
        with open(config_path, "r") as f:
            data = f.read()
        with open(config_save_path, "w") as f:
            f.write(data)
    else:
        with open(config_save_path, "r") as f:
            data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    hparams.model_dir = model_dir
    hparams.train.seed = args.seed
    hparams.not_pretrained = args.not_pretrained
    return hparams


class HParams():
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()


def get_logger(model_dir, filename="train.log"):
    global logger
    logger = logging.getLogger(os.path.basename(model_dir))
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    h = logging.FileHandler(os.path.join(model_dir, filename))
    h.setLevel(logging.DEBUG)
    h.setFormatter(formatter)
    logger.addHandler(h)
    return logger
