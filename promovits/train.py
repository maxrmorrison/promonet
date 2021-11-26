import argparse
import contextlib
import json
import shutil
from pathlib import Path
import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
import torchinfo

import commons
import utils
from models import (
    SynthesizerTrn,
    MultiPeriodDiscriminator)


import promovits

global_step = 0
printed = False


def train(rank, world_size, hps, gpu=None):
    global global_step
    if not rank:
        logger = utils.get_logger(hps.log_dir)
        logger.info(hps)
        utils.check_git_hash(hps.log_dir)
        writer = SummaryWriter(log_dir=hps.log_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.log_dir, "eval"))

    torch.manual_seed(promovits.RANDOM_SEED)
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')

    #######################
    # Create data loaders #
    #######################

    # TODO - args
    train_loader, eval_loader = promovits.data.loaders()

    #################
    # Create models #
    #################

    net_g = SynthesizerTrn(
        len(promovits.preprocess.text.symbols()),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model).to(device)
    net_d = MultiPeriodDiscriminator(
        hps.model.use_spectral_norm).to(device)
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps)
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps)
    if rank is not None:
        net_g = DDP(net_g, device_ids=[rank])
        net_d = DDP(net_d, device_ids=[rank])

    try:
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.log_dir, "G_*.pth"), net_g, optim_g)
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.log_dir, "D_*.pth"), net_d, optim_d)
        global_step = (epoch_str - 1) * len(train_loader)
    except:
        epoch_str = 1
        global_step = 0

    ############################################
    # Schedulers and automatic mixed precision #
    ############################################

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g,
        gamma=hps.train.lr_decay,
        last_epoch=epoch_str-2)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d,
        gamma=hps.train.lr_decay,
        last_epoch=epoch_str-2)
    scaler = GradScaler(enabled=hps.train.fp16_run)

    #########
    # Train #
    #########

    for epoch in range(epoch_str, hps.train.epochs + 1):
        if not rank:
            train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d],
                [optim_g, optim_d],
                scaler,
                [train_loader, eval_loader],
                logger,
                [writer, writer_eval],
                device)
        else:
            train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d],
                [optim_g, optim_d],
                scaler,
                [train_loader, None],
                None,
                None,
                device)
        scheduler_g.step()
        scheduler_d.step()


def train_and_evaluate(
    rank,
    epoch,
    hps,
    nets,
    optims,
    scaler,
    loaders,
    logger,
    writers,
    device):
    net_g, net_d = nets
    optim_g, optim_d = optims
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    train_loader.batch_sampler.set_epoch(epoch)
    global global_step
    global printed

    net_g.train()
    net_d.train()
    for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, speakers) in enumerate(train_loader):
        x, x_lengths = x.to(device), x_lengths.to(device)
        spec, spec_lengths = spec.to(device), spec_lengths.to(device)
        y, y_lengths = y.to(device), y_lengths.to(device)
        speakers = speakers.to(device)

        with autocast(enabled=hps.train.fp16_run):
            # Forward pass through generator
            y_hat, l_length, attn, ids_slice, x_mask, z_mask,\
            (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(x, x_lengths, spec, spec_lengths, speakers)

            # Convert to mels
            mel = promovits.preprocess.spectrogram.linear_to_mel(spec)

            y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
            y_hat_mel = promovits.preprocess.spectrogram.from_audio(
                y_hat,
                True)

            y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size) # slice

            # Print model summaries first time
            if not printed:
                print(torchinfo.summary(
                net_g,
                input_data=(x, x_lengths, spec, spec_lengths, speakers),
                device=device))
                print(torchinfo.summary(
                net_d,
                input_data=(y, y_hat.detach()),
                device=device))
                printed = True

            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = promovits.loss.discriminator(y_d_hat_r, y_d_hat_g)
                loss_disc_all = loss_disc

        optim_d.zero_grad()
        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = clip_gradients(net_d.parameters(), None)
        scaler.step(optim_d)

        with autocast(enabled=hps.train.fp16_run):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            with autocast(enabled=False):
                if l_length is not None:
                    loss_dur = torch.sum(l_length.float())
                else:
                    loss_dur = 0
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = promovits.loss.kl(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                loss_fm = promovits.loss.feature_matching(fmap_r, fmap_g)
                loss_gen, losses_gen = promovits.loss.generator(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl
        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_gradients(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        if not rank:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]['lr']
                losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_dur, loss_kl]
                logger.info('Train Epoch: {} [{:.0f}%]'.format(
                    epoch,
                    100. * batch_idx / len(train_loader)))
                logger.info(
                    [torch.tensor(x).item() for x in losses] + [global_step, lr])

                scalar_dict = {
                    "loss/g/total": loss_gen_all,
                    "loss/d/total": loss_disc_all,
                    "learning_rate": lr,
                    "grad_norm_d": grad_norm_d,
                    "grad_norm_g": grad_norm_g,
                    "loss/g/fm": loss_fm,
                    "loss/g/mel": loss_mel,
                    "loss/g/dur": loss_dur,
                    "loss/g/kl": loss_kl}
                scalar_dict.update(
                    {"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
                scalar_dict.update(
                    {"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
                scalar_dict.update(
                    {"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})

                image_dict = {
                    "slice/mel_org": utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
                    "slice/mel_gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()),
                    "all/mel": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
                    "all/attn": utils.plot_alignment_to_numpy(attn[0,0].data.cpu().numpy())}
                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    images=image_dict,
                    scalars=scalar_dict)

            if global_step % hps.train.eval_interval == 0:
                evaluate(hps, net_g, eval_loader, writer_eval, device)
            if global_step % hps.train.checkpoint_interval == 0:
                utils.save_checkpoint(
                    net_g,
                    optim_g,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(hps.log_dir, "G_{}.pth".format(global_step)))
                utils.save_checkpoint(
                    net_d,
                    optim_d,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(hps.log_dir, "D_{}.pth".format(global_step)))
        global_step += 1

    if not rank:
        logger.info('====> Epoch: {}'.format(epoch))


def evaluate(hps, generator, eval_loader, writer_eval, device):
    generator.eval()
    with torch.no_grad():
        for x, x_lengths, spec, spec_lengths, y, y_lengths, speakers in eval_loader:
            x, x_lengths = x.to(device), x_lengths.to(device)
            spec, spec_lengths = spec.to(device), spec_lengths.to(device)
            y, y_lengths = y.to(device), y_lengths.to(device)
            speakers = speakers.to(device)
            break

        audio_dict = {}
        image_dict = {}
        for i in range(8):
            y_hat, _, mask, *_ = generator.infer(
                x[i:i + 1, ..., :x_lengths[i]],
                x_lengths[i:i + 1],
                speakers[i:i + 1],
                max_len=1000)
            y_hat_lengths = mask.sum([1,2]).long() * hps.data.hop_length

            # Get true and predicted mels
            mel = promovits.preprocess.spectrogram.linear_to_mel(
                spec[i:i + 1, :, :spec_lengths[i]])
            y_hat_mel = promovits.preprocess.spectrogram.from_audio(
                y_hat.float(),
                True)

            image_dict[f"gen/mel-{i}"] = utils.plot_spectrogram_to_numpy(
                y_hat_mel[0].cpu().numpy())
            audio_dict[f"gen/audio-{i}"] = y_hat[0,:,:y_hat_lengths[0]]
            if global_step == 0:
                image_dict[f"gt/mel-{i}"] = utils.plot_spectrogram_to_numpy(
                    mel[0].cpu().numpy())
                audio_dict[f"gt/audio-{i}"] = y[i,:,:y_lengths[i]]

    utils.summarize(
      writer=writer_eval,
      global_step=global_step,
      images=image_dict,
      audios=audio_dict,
      audio_sampling_rate=hps.data.sampling_rate)
    generator.train()


###############################################################################
# Distributed data parallelism
###############################################################################


def train_ddp(rank, hps, gpus):
    """Train with distributed data parallelism"""
    with ddp_context(rank, len(gpus)):
        train(rank, len(gpus), hps, gpus)


@ contextlib.contextmanager
def ddp_context(rank, world_size):
    """Context manager for distributed data parallelism"""
    # Setup ddp
    os.environ['MASTER_ADDR']='localhost'
    os.environ['MASTER_PORT']='12355'
    torch.distributed.init_process_group(
        "nccl",
        init_method = "env://",
        world_size = world_size,
        rank = rank)

    try:

        # Execute user code
        yield

    finally:

        # Close ddp
        torch.distributed.destroy_process_group()


###############################################################################
# Entry point
###############################################################################


def clip_gradients(parameters, clip_value, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if clip_value is not None:
        clip_value = float(clip_value)

    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
        if clip_value is not None:
            p.grad.data.clamp_(min=-clip_value, max=clip_value)
    return total_norm ** (1. / norm_type)


###############################################################################
# Entry point
###############################################################################


def main(config_file, gpus = None):
    # Optionally overwrite training with same name
    directory=Path('logs') / config_file.stem

    # Create output directory
    directory.mkdir(parents = True, exist_ok = True)

    # Save configuration
    shutil.copyfile(config_file, directory / config_file.name)

    # Load configuration
    with open(config_file) as file:
        hps=utils.HParams(**json.load(file))
    hps.log_dir=directory

    if gpus is None:
        # CPU training
        train(None, 0, hps)
    elif len(gpus) > 1:
        # Distributed data parallelism
        mp.spawn(
            train_ddp,
            args = (hps, gpus),
            nprocs = len(gpus),
            join = True)
    else:
        # Single GPU training
        train(None, 1, hps, None if gpus is None else gpus[0])


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config_file',
        type=Path,
        required=True,
        help='The configuration file')
    parser.add_argument(
        '--gpus',
        type=int,
        nargs='+',
        help='The gpus to run training on')
    return parser.parse_args()


if __name__ == '__main__':
    main(**vars(parse_args()))
