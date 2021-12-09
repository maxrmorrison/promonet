import argparse
import contextlib
import json
import os
import shutil
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
import torchinfo

import promovits


global_step = 0
printed = False


def train(
    dataset,
    directory,
    config,
    train_partition='train',
    valid_partition='valid',
    adapt=False,
    gpu=None):
    # Get DDP rank
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = None

    global global_step
    if not rank:
        print(json.dumps(config, indent=4, sort_keys=True))
        writer = SummaryWriter(log_dir=directory)

    torch.manual_seed(promovits.RANDOM_SEED)
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')

    #######################
    # Create data loaders #
    #######################

    train_loader, valid_loader = promovits.data.loaders(
        dataset,
        train_partition,
        valid_partition,
        gpu,
        config.model.use_ppg,
        config.data.interp_method)

    #################
    # Create models #
    #################

    net_g = promovits.model.Generator(
        len(promovits.preprocess.text.symbols()),
        promovits.WINDOW_SIZE // 2 + 1,
        config.train.segment_size // promovits.HOPSIZE,
        n_speakers=config.data.n_speakers,
        **config.model).to(device)
    net_d = promovits.model.Discriminator().to(device)
    # TODO - adaptation optimizers
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        config.train.learning_rate,
        betas=config.train.betas,
        eps=config.train.eps)
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        config.train.learning_rate,
        betas=config.train.betas,
        eps=config.train.eps)
    if rank is not None:
        net_g = torch.nn.parallel.DistributedDataParallel(
            net_g,
            device_ids=[rank])
        net_d = torch.nn.parallel.DistributedDataParallel(
            net_d,
            device_ids=[rank])

    try:
        net_g, optim_g, epoch = promovits.load.checkpoint(
            latest_checkpoint_path(directory, "G_*.pth"),
            net_g,
            optim_g)
        net_d, optim_d, epoch = promovits.load.checkpoint(
            latest_checkpoint_path(directory, "D_*.pth"),
            net_d,
            optim_d)
        global_step = (epoch - 1) * len(train_loader)
    except:
        # TODO - fix bad checkpointing (maybe epochs vs steps?)
        epoch = 1
        global_step = 0

    ############################################
    # Schedulers and automatic mixed precision #
    ############################################

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g,
        gamma=config.train.lr_decay,
        last_epoch=epoch-2)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d,
        gamma=config.train.lr_decay,
        last_epoch=epoch-2)
    scaler = torch.cuda.amp.GradScaler(enabled=config.train.fp16_run)

    #########
    # Train #
    #########

    while epoch < config.train.epochs + 1:
        train_and_evaluate(
            rank,
            directory,
            epoch,
            config,
            [net_g, net_d],
            [optim_g, optim_d],
            scaler,
            [train_loader, valid_loader],
            writer,
            device)

        scheduler_g.step()
        scheduler_d.step()
        epoch += 1


# TODO - merge with train()
def train_and_evaluate(
    rank,
    directory,
    epoch,
    config,
    nets,
    optims,
    scaler,
    loaders,
    writer,
    device):
    net_g, net_d = nets
    optim_g, optim_d = optims
    train_loader, valid_loader = loaders

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

        with torch.cuda.amp.autocast(enabled=config.train.fp16_run):
            # Forward pass through generator
            y_hat, l_length, attn, ids_slice, x_mask, z_mask,\
            (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(x, x_lengths, spec, spec_lengths, speakers)

            # Convert to mels
            mel = promovits.preprocess.spectrogram.linear_to_mel(spec)

            y_mel = promovits.model.slice_segments(
                mel,
                ids_slice,
                config.train.segment_size // config.data.hop_length)
            y_hat_mel = promovits.preprocess.spectrogram.from_audio(
                y_hat,
                True)
            y = promovits.model.slice_segments(
                y,
                ids_slice * config.data.hop_length,
                config.train.segment_size)

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
            with torch.cuda.amp.autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = promovits.loss.discriminator(y_d_hat_r, y_d_hat_g)
                loss_disc_all = loss_disc

        optim_d.zero_grad()
        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = clip_gradients(net_d.parameters(), None)
        scaler.step(optim_d)

        with torch.cuda.amp.autocast(enabled=config.train.fp16_run):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            with torch.cuda.amp.autocast(enabled=False):
                if l_length is not None:
                    loss_dur = torch.sum(l_length.float())
                else:
                    loss_dur = 0
                loss_mel = torch.nn.functional.l1_loss(y_mel, y_hat_mel) * config.train.c_mel
                loss_kl = promovits.loss.kl(z_p, logs_q, m_p, logs_p, z_mask) * config.train.c_kl
                loss_fm = promovits.loss.feature_matching(fmap_r, fmap_g)
                loss_gen, losses_gen = promovits.loss.generator(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl
        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = clip_gradients(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        if not rank:

            ###########
            # Logging #
            ###########

            if global_step % config.train.log_interval == 0:
                lr = optim_g.param_groups[0]['lr']
                losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_dur, loss_kl]
                print('Train Epoch: {} [{:.0f}%]'.format(
                    epoch,
                    100. * batch_idx / len(train_loader)))
                print(
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
                    "slice/mel_org": promovits.evaluate.plot.spectrogram(y_mel[0].data.cpu().numpy()),
                    "slice/mel_gen": promovits.evaluate.plot.spectrogram(y_hat_mel[0].data.cpu().numpy()),
                    "all/mel": promovits.evaluate.plot.spectrogram(mel[0].data.cpu().numpy()),
                    "all/attn": promovits.evaluate.plot.alignment(attn[0,0].data.cpu().numpy())}
                summarize(
                    writer=writer,
                    global_step=global_step,
                    images=image_dict,
                    scalars=scalar_dict)

            ############
            # Evaluate #
            ############

            if global_step % config.train.eval_interval == 0:
                evaluate(config, net_g, valid_loader, writer, device)

            ###################
            # Save checkpoint #
            ###################

            if global_step % config.train.checkpoint_interval == 0:
                save_checkpoint(
                    net_g,
                    optim_g,
                    epoch,
                    directory / f'G_{global_step}.pth')
                save_checkpoint(
                    net_d,
                    optim_d,
                    epoch,
                    directory / f'D_{global_step}.pth')
        global_step += 1

    if not rank:
        print('====> Epoch: {}'.format(epoch))


def evaluate(config, generator, valid_loader, writer, device):
    generator.eval()
    with torch.no_grad():
        for x, x_lengths, spec, spec_lengths, y, y_lengths, speakers in valid_loader:
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
            y_hat_lengths = mask.sum([1,2]).long() * config.data.hop_length

            # Get true and predicted mels
            mel = promovits.preprocess.spectrogram.linear_to_mel(
                spec[i:i + 1, :, :spec_lengths[i]])
            y_hat_mel = promovits.preprocess.spectrogram.from_audio(
                y_hat.float(),
                True)

            image_dict[f"gen/mel-{i}"] = promovits.evaluate.plot.spectrogram(
                y_hat_mel[0].cpu().numpy())
            audio_dict[f"gen/audio-{i}"] = y_hat[0,:,:y_hat_lengths[0]]
            if global_step == 0:
                image_dict[f"gt/mel-{i}"] = promovits.evaluate.plot.spectrogram(
                    mel[0].cpu().numpy())
                audio_dict[f"gt/audio-{i}"] = y[i,:,:y_lengths[i]]

    summarize(
        writer=writer,
        global_step=global_step,
        images=image_dict,
        audios=audio_dict)
    generator.train()


###############################################################################
# Distributed data parallelism
###############################################################################


def train_ddp(rank, dataset, directory, config, gpus):
    """Train with distributed data parallelism"""
    with ddp_context(rank, len(gpus)):
        train(dataset, directory, config, gpus)


@ contextlib.contextmanager
def ddp_context(rank, world_size):
    """Context manager for distributed data parallelism"""
    # Setup ddp
    os.environ['MASTER_ADDR']='localhost'
    os.environ['MASTER_PORT']='12355'
    torch.distributed.init_process_group(
        "nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank)

    try:

        # Execute user code
        yield

    finally:

        # Close ddp
        torch.distributed.destroy_process_group()


###############################################################################
# Utilities
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


def latest_checkpoint_path(directory, regex="G_*.pth"):
    """Retrieve the path to the most recent checkpoint"""
    files = directory.glob(regex)
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    return files[-1]


def save_checkpoint(
    model,
    optimizer,
    iteration,
    checkpoint_path):
    """Save training checkpoint to disk"""
    print("Saving model and optimizer state at iteration {} to {}".format(
        iteration, checkpoint_path))
    checkpoint = {
        'model': model.state_dict(),
        'iteration': iteration,
        'optimizer': optimizer.state_dict()}
    torch.save(checkpoint, checkpoint_path)


def summarize(
    writer,
    global_step,
    scalars={},
    histograms={},
    images={},
    audios={}):
    """Add assets to Tensorboard"""
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)
    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)
    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats='HWC')
    for k, v in audios.items():
        writer.add_audio(k, v, global_step, promovits.SAMPLE_RATE)


###############################################################################
# Entry point
###############################################################################


def main(
    config_file,
    dataset,
    train_partition='train',
    valid_partition='valid',
    adapt=False,
    gpus=None):
    # Optionally overwrite training with same name
    directory = promovits.TRAIN_DIR / config_file.stem

    # Create output directory
    directory.mkdir(parents=True, exist_ok=True)

    # Save configuration
    shutil.copyfile(config_file, directory / config_file.name)

    # Load configuration
    config = promovits.load.config(config_file)

    if gpus is None:

        # CPU training
        train(dataset, directory, config, train_partition, valid_partition, adapt)

    elif len(gpus) > 1:

        args = (
            dataset,
            directory,
            config,
            train_partition,
            valid_partition,
            adapt,
            gpus)

        # Distributed data parallelism
        torch.multiprocessing.spawn(
            train_ddp,
            args=args,
            nprocs=len(gpus),
            join=True)

    else:

        # Single GPU training
        train(
            dataset,
            directory,
            config,
            train_partition,
            valid_partition,
            adapt,
            None if gpus is None else gpus[0])


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config_file',
        type=Path,
        required=True,
        help='The configuration file')
    parser.add_argument(
        '--dataset',
        required=True,
        help='The dataset to train on')
    parser.add_argument(
        '--train_partition',
        default='train',
        help='The data partition to train on')
    parser.add_argument(
        '--valid_partition',
        default='valid',
        help='The data partition to perform validation on')
    parser.add_argument(
        '--adapt',
        action='store_true',
        help='Whether to use hyperparameters for speaker adaptation')
    parser.add_argument(
        '--gpus',
        type=int,
        nargs='+',
        help='The gpus to run training on')
    return parser.parse_args()


if __name__ == '__main__':
    main(**vars(parse_args()))
