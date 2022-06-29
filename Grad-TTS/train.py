# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import params
from model import GradTTS
# from data import TextMelDataset, TextMelBatchCollate
import data_collate
import data_loader
from utils import plot_tensor, save_plot
from model.utils import fix_len_compatibility
from text.symbols import symbols
import utils

# train_filelist_path = params.train_filelist_path
# valid_filelist_path = params.valid_filelist_path
# cmudict_path = params.cmudict_path
# add_blank = params.add_blank
# n_spks = params.n_spks
# spk_emb_dim = params.spk_emb_dim

# log_dir = params.log_dir
# n_epochs = params.n_epochs
# batch_size = params.batch_size
# learning_rate = params.learning_rate
# random_seed = params.seed

# nsymbols = 148
# n_enc_channels = params.n_enc_channels
# filter_channels = params.filter_channels
# filter_channels_dp = params.filter_channels_dp
# n_enc_layers = params.n_enc_layers
# enc_kernel = params.enc_kernel
# enc_dropout = params.enc_dropout
# n_heads = params.n_heads
# window_size = params.window_size
#
# n_feats = params.n_feats
# n_fft = params.n_fft
# sample_rate = params.sample_rate
# hop_length = params.hop_length
# win_length = params.win_length
# f_min = params.f_min
# f_max = params.f_max
#
# dec_dim = params.dec_dim
# beta_min = params.beta_min
# beta_max = params.beta_max
# pe_scale = params.pe_scale

if __name__ == "__main__":
    hps = utils.get_hparams()
    logger_text = utils.get_logger(hps.model_dir)
    logger_text.info(hps)

    out_size = fix_len_compatibility(2 * hps.data.sampling_rate // hps.data.hop_length)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(hps.train.seed)
    np.random.seed(hps.train.seed)

    print('Initializing logger...')
    log_dir = hps.model_dir
    logger = SummaryWriter(log_dir=log_dir)

    train_dataset, collate, model = utils.get_correct_class(hps)
    test_dataset, _, _ = utils.get_correct_class(hps, train=False)

    print('Initializing data loaders...')

    batch_collate = collate
    loader = DataLoader(dataset=train_dataset, batch_size=hps.train.batch_size,
                        collate_fn=batch_collate, drop_last=True,
                        num_workers=4, shuffle=False)  # NOTE: if on server, worker can be 4

    print('Initializing model...')
    model = model(**hps.model).to(device)
    print('Number of encoder + duration predictor parameters: %.2fm' % (model.encoder.nparams / 1e6))
    print('Number of decoder parameters: %.2fm' % (model.decoder.nparams / 1e6))
    print('Total parameters: %.2fm' % (model.nparams / 1e6))

    print('Initializing optimizer...')
    optimizer = torch.optim.Adam(params=model.parameters(), lr=hps.train.learning_rate)

    print('Logging test batch...')
    test_batch = test_dataset.sample_test_batch(size=hps.train.test_size)
    for i, item in enumerate(test_batch):
        mel = item['mel']
        logger.add_image(f'image_{i}/ground_truth', plot_tensor(mel.squeeze()),
                         global_step=0, dataformats='HWC')
        save_plot(mel.squeeze(), f'{log_dir}/original_{i}.png')

    print('Start training...')
    iteration = 0
    for epoch in range(1, hps.train.n_epochs + 1):
        model.train()
        dur_losses = []
        prior_losses = []
        diff_losses = []
        with tqdm(loader, total=len(train_dataset) // hps.train.batch_size) as progress_bar:
            for batch_idx, batch in enumerate(progress_bar):
                model.zero_grad()
                x, x_lengths = batch['text_padded'].to(device), \
                               batch['input_lengths'].to(device)
                y, y_lengths = batch['mel_padded'].to(device), \
                               batch['output_lengths'].to(device)
                if hps.xvector:
                    spk = batch['xvector'].to(device)
                else:
                    spk = batch['spk_ids'].to(torch.long).to(device)

                dur_loss, prior_loss, diff_loss = model.compute_loss(x, x_lengths,
                                                                     y, y_lengths,
                                                                     spk=spk,
                                                                     out_size=out_size)
                loss = sum([dur_loss, prior_loss, diff_loss])
                loss.backward()

                enc_grad_norm = torch.nn.utils.clip_grad_norm_(model.encoder.parameters(),
                                                               max_norm=1)
                dec_grad_norm = torch.nn.utils.clip_grad_norm_(model.decoder.parameters(),
                                                               max_norm=1)
                optimizer.step()

                logger.add_scalar('training/duration_loss', dur_loss.item(),
                                  global_step=iteration)
                logger.add_scalar('training/prior_loss', prior_loss.item(),
                                  global_step=iteration)
                logger.add_scalar('training/diffusion_loss', diff_loss.item(),
                                  global_step=iteration)
                logger.add_scalar('training/encoder_grad_norm', enc_grad_norm,
                                  global_step=iteration)
                logger.add_scalar('training/decoder_grad_norm', dec_grad_norm,
                                  global_step=iteration)

                dur_losses.append(dur_loss.item())
                prior_losses.append(prior_loss.item())
                diff_losses.append(diff_loss.item())

                if batch_idx % 5 == 0:
                    msg = f'Epoch: {epoch}, iteration: {iteration} | dur_loss: {dur_loss.item()}, prior_loss: {prior_loss.item()}, diff_loss: {diff_loss.item()}'
                    # logger_text.info(msg)
                    progress_bar.set_description(msg)

                iteration += 1

        log_msg = 'Epoch %d: duration loss = %.3f ' % (epoch, float(np.mean(dur_losses)))
        log_msg += '| prior loss = %.3f ' % np.mean(prior_losses)
        log_msg += '| diffusion loss = %.3f\n' % np.mean(diff_losses)
        with open(f'{log_dir}/train.log', 'a') as f:
            f.write(log_msg)

        if epoch % hps.train.save_every > 0:
            continue

        model.eval()
        print('Synthesis...')
        with torch.no_grad():
            for i, item in enumerate(test_batch):
                # print(item)
                x = item['phn_ids'].to(torch.long).unsqueeze(0).to(device)
                if not hps.xvector:
                    spk = item['spk_ids']
                    spk = torch.LongTensor([spk]).to(device)
                else:
                    spk = item["xvector"]
                    spk = spk.unsqueeze(0).to(device)

                x_lengths = torch.LongTensor([x.shape[-1]]).to(device)
                # print(x.shape, spk.shape)
                y_enc, y_dec, attn = model(x, x_lengths, spk=spk, n_timesteps=50)
                logger.add_image(f'image_{i}/generated_enc',
                                 plot_tensor(y_enc.squeeze().cpu()),
                                 global_step=iteration, dataformats='HWC')
                logger.add_image(f'image_{i}/generated_dec',
                                 plot_tensor(y_dec.squeeze().cpu()),
                                 global_step=iteration, dataformats='HWC')
                logger.add_image(f'image_{i}/alignment',
                                 plot_tensor(attn.squeeze().cpu()),
                                 global_step=iteration, dataformats='HWC')
                save_plot(y_enc.squeeze().cpu(),
                          f'{log_dir}/generated_enc_{i}.png')
                save_plot(y_dec.squeeze().cpu(),
                          f'{log_dir}/generated_dec_{i}.png')
                save_plot(attn.squeeze().cpu(),
                          f'{log_dir}/alignment_{i}.png')

        ckpt = model.state_dict()
        torch.save(ckpt, f=f"{log_dir}/grad_{epoch}.pt")
