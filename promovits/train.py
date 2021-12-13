import argparse
import contextlib
import functools
import os
import shutil
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
import torchinfo

import promovits


###############################################################################
# Training
###############################################################################


def train(
    dataset,
    directory,
    train_partition='train',
    valid_partition='valid',
    adapt=False,
    gpu=None):
    """Train a model"""
    # Get DDP rank
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = None

    if not rank:
        writer = SummaryWriter(log_dir=directory)

    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')

    #######################
    # Create data loaders #
    #######################

    torch.manual_seed(promovits.RANDOM_SEED)
    train_loader, valid_loader = promovits.data.loaders(
        dataset,
        train_partition,
        valid_partition,
        gpu,
        promovits.PPG_FEATURES,
        promovits.PPG_INTERP_METHOD)

    #################
    # Create models #
    #################

    generator = promovits.model.Generator(
        len(promovits.preprocess.text.symbols()),
        # TODO - speaker adaptation
        n_speakers=max(int(speaker) for speaker in train_loader.dataset.speakers) + 1,
        use_ppg=promovits.PPG_FEATURES).to(device)
    discriminators = promovits.model.Discriminator().to(device)

    ##################################################
    # Maybe setup distributed data parallelism (DDP) #
    ##################################################

    if rank is not None:
        ddp_fn = functools.partial(
            torch.nn.parallel.DistributedDataParallel,
            device_ids=[rank])
        generator = ddp_fn(generator)
        discriminators = ddp_fn(discriminators)

    #####################
    # Create optimizers #
    #####################

    if adapt:
        # TODO - adaptation optimizers (may also need different loading, saving, logging, etc.)
        generator_optimizer = promovits.ADAPTATION_OPTIMIZER(
            generator.parameters())
        discriminator_optimizer = promovits.ADAPTATION_OPTIMIZER(
            discriminators.parameters())
    else:
        generator_optimizer = promovits.TRAINING_OPTIMIZER(
            generator.parameters())
        discriminator_optimizer = promovits.TRAINING_OPTIMIZER(
            discriminators.parameters())

    ##############################
    # Maybe load from checkpoint #
    ##############################

    try:

        # Load generator
        (
            generator,
            generator_optimizer,
            step
        ) = promovits.load.checkpoint(
            latest_checkpoint_path(directory, 'generator-*.pt'),
            generator,
            generator_optimizer
        )

        # Load discriminator
        (
            discriminators,
            discriminator_optimizer,
            step
        ) = promovits.load.checkpoint(
            latest_checkpoint_path(directory, 'discriminator-*.pt'),
            discriminators,
            discriminator_optimizer
        )

    except:

        # Train from scratch
        step = 0

    #####################
    # Create schedulers #
    #####################

    scheduler_fn = functools.partial(
        torch.optim.lr_scheduler.ExponentialLR,
        gamma=promovits.LEARNING_RATE_DECAY,
        last_epoch=step // len(train_loader.dataset) if step else -1)
    generator_scheduler = scheduler_fn(generator_optimizer)
    discriminator_scheduler = scheduler_fn(discriminator_optimizer)

    #########
    # Train #
    #########

    # Automatic mixed precision (amp) gradient scaler
    scaler = torch.cuda.amp.GradScaler()

    # Print model summaries on the first step
    printed = False

    while step < promovits.NUM_STEPS + 1:

        # Seed sampler
        train_loader.batch_sampler.set_epoch(step // len(train_loader.dataset))

        generator.train()
        discriminators.train()
        for batch in train_loader:

            # Unpack batch
            (
                phonemes,
                phoneme_lengths,
                spectrograms,
                spectrogram_lengths,
                audio,
                _,
                speakers
            ) = unpack_to_device(batch, device)

            with torch.cuda.amp.autocast():

                # Bundle training input
                generator_input = (
                    phonemes,
                    phoneme_lengths,
                    spectrograms,
                    spectrogram_lengths,
                    speakers)

                # Forward pass through generator
                generated, durations, attention, slice_indices, _, z_mask,\
                    (_, z_p, m_p, logs_p, _, logs_q) = generator(*generator_input)

                # Convert to mels
                mels = promovits.preprocess.spectrogram.linear_to_mel(
                    spectrograms)
                mel_slices = promovits.model.slice_segments(
                    mels,
                    slice_indices,
                    promovits.TRAINING_CHUNK_SIZE // promovits.HOPSIZE)

            # Exit autocast context, as ComplexHalf type is not supported
            # See Github issue https://github.com/jaywalnut310/vits/issues/15
            generated = generated.float()
            generated_mels = promovits.preprocess.spectrogram.from_audio(
                generated,
                True)
            audio = promovits.model.slice_segments(
                audio,
                slice_indices * promovits.HOPSIZE,
                promovits.TRAINING_CHUNK_SIZE)

            # Print model summaries first time
            if not printed:
                print(torchinfo.summary(
                    generator,
                    input_data=generator_input,
                    device=device))
                print(torchinfo.summary(
                    discriminators,
                    input_data=(audio, generated.detach()),
                    device=device))
                printed = True

            #######################
            # Train discriminator #
            #######################

            real_logits, fake_logits, _, _ = discriminators(
                audio,
                generated.detach())

            # Get discriminator loss
            (
                discriminator_losses,
                real_discriminator_losses,
                fake_discriminator_losses
            ) = promovits.loss.discriminator(real_logits, fake_logits)

            ##########################
            # Optimize discriminator #
            ##########################

            discriminator_optimizer.zero_grad()
            scaler.scale(discriminator_losses).backward()
            scaler.step(discriminator_optimizer)

            ###################
            # Train generator #
            ###################

            with torch.cuda.amp.autocast():
                (
                    _,
                    fake_logits,
                    real_feature_maps,
                    fake_feature_maps
                ) = discriminators(audio, generated)

            ####################
            # Generator losses #
            ####################

            # Get duration loss
            if durations is not None:
                duration_loss = torch.sum(durations.float())
            else:
                duration_loss = 0

            # Get melspectrogram loss
            mel_loss = torch.nn.functional.l1_loss(
                mel_slices, generated_mels) * promovits.MEL_LOSS_WEIGHT

            # Get KL divergence loss between phonemes and spectrogram
            kl_divergence_loss = promovits.loss.kl(z_p, logs_q, m_p, logs_p, z_mask)

            # Get feature matching loss
            feature_matching_loss = promovits.loss.feature_matching(
                real_feature_maps,
                fake_feature_maps)

            # Get adversarial loss
            adversarial_loss, adversarial_losses = promovits.loss.generator(
                fake_logits)
            generator_losses = (
                adversarial_loss +
                feature_matching_loss +
                mel_loss +
                duration_loss +
                kl_divergence_loss)

            ######################
            # Optimize generator #
            ######################

            generator_optimizer.zero_grad()
            scaler.scale(generator_losses).backward()
            scaler.step(generator_optimizer)
            scaler.update()

            ###########
            # Logging #
            ###########

            if not rank:

                if step % promovits.LOG_INTERVAL == 0:
                    print(
                        f'Training - step ({step}) ' +
                        f'[{100. * step / promovits.NUM_STEPS:.0f}%]')

                    # Log losses
                    scalars = {
                        'train/loss/generator/total': generator_losses,
                        'train/loss/discriminator/total': discriminator_losses,
                        'train/learning_rate':
                            generator_optimizer.param_groups[0]['lr'],
                        'train/loss/generator/feature-matching':
                            feature_matching_loss,
                        'train/loss/generator/mels': mel_loss,
                        'train/loss/generator/duration': duration_loss,
                        'train/loss/generator/kl-divergence':
                            kl_divergence_loss}
                    scalars.update(
                        {f'train/loss/generator/adversarial-{i:02d}': value
                        for i, value in enumerate(adversarial_losses)})
                    scalars.update(
                        {f'train/loss/discriminator/real-{i:02d}': value
                        for i, value in enumerate(real_discriminator_losses)})
                    scalars.update(
                        {f'train/loss/discriminator/fake-{i:02d}': value
                        for i, value in enumerate(fake_discriminator_losses)})

                    # Log mels and attention matrix
                    images = {
                        'train/mels/slice/original':
                            promovits.evaluate.plot.spectrogram(
                                mel_slices[0].data.cpu().numpy()),
                        'train/mels/slice/generated':
                            promovits.evaluate.plot.spectrogram(
                                generated_mels[0].data.cpu().numpy()),
                        'train/mels/original':
                            promovits.evaluate.plot.spectrogram(
                                mels[0].data.cpu().numpy()),
                        'train/attention':
                            promovits.evaluate.plot.alignment(
                                attention[0, 0].data.cpu().numpy())}

                    # Write to tensorboard
                    summarize(writer, step, images=images, scalars=scalars)

                ############
                # Evaluate #
                ############

                if step % promovits.EVALUATION_INTERVAL == 0:
                    evaluate(step, generator, valid_loader, writer, device)

                ###################
                # Save checkpoint #
                ###################

                if step % promovits.CHECKPOINT_INTERVAL == 0:
                    save_checkpoint(
                        generator,
                        generator_optimizer,
                        step,
                        directory / f'generator-{step:08d}.pt')
                    save_checkpoint(
                        discriminators,
                        discriminator_optimizer,
                        step,
                        directory / f'discriminator-{step:08d}.pt')

            # Update training step count
            if step >= promovits.NUM_STEPS:
                break
            step += 1

        # Update learning rate every epoch
        generator_scheduler.step()
        discriminator_scheduler.step()


###############################################################################
# Evaluation
###############################################################################


def evaluate(step, generator, valid_loader, writer, device):
    """Perform model evaluation"""
    # Prepare generator for evaluation
    generator.eval()

    # Turn off gradient computation
    with torch.no_grad():

        # Unpack batch
        (
            phonemes,
            phoneme_lengths,
            spectrogram,
            spectrogram_lengths,
            audio,
            audio_lengths,
            speakers
        ) = unpack_to_device(next(valid_loader), device)

        waveforms = {}
        images = {}
        for i in range(len(valid_loader.dataset)):

            # Generate speech
            generated, _, mask, *_ = generator.infer(
                phonemes[i:i + 1, ..., :phoneme_lengths[i]],
                phoneme_lengths[i:i + 1],
                speakers[i:i + 1],
                max_len=2048)
            generated_lengths = mask.sum([1,2]).long() * promovits.HOPSIZE

            if not step:

                # Log original audio
                waveforms[f'evaluate/original-{i:02d}'] = \
                    audio[i, :, :audio_lengths[i]]

                # Log original melspectrogram
                mels = promovits.preprocess.spectrogram.linear_to_mel(
                    spectrogram[i:i + 1, :, :spectrogram_lengths[i]])
                images[f'evaluate/mels/original-{i:02d}'] = \
                    promovits.evaluate.plot.spectrogram(mels[0].cpu().numpy())

            # Log generated audio
            waveforms[f'evaluate/generated-{i:02d}'] = \
                generated[0,:,:generated_lengths[0]]

            # Log generated melspectrogram
            generated_mels = promovits.preprocess.spectrogram.from_audio(
                generated.float(),
                True)
            images[f'evaluate/mels/generated-{i:02d}'] = \
                promovits.evaluate.plot.spectrogram(
                    generated_mels.cpu().numpy())

    # Write to Tensorboard
    summarize(writer, step, images=images, waveforms=waveforms)

    # Prepare generator for training
    generator.train()


###############################################################################
# Distributed data parallelism
###############################################################################


def train_ddp(rank, dataset, directory, gpus):
    """Train with distributed data parallelism"""
    with ddp_context(rank, len(gpus)):
        train(dataset, directory, gpus)


@ contextlib.contextmanager
def ddp_context(rank, world_size):
    """Context manager for distributed data parallelism"""
    # Setup ddp
    os.environ['MASTER_ADDR']='localhost'
    os.environ['MASTER_PORT']='12355'
    torch.distributed.init_process_group(
        'nccl',
        init_method='env://',
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


def latest_checkpoint_path(directory, regex='generator-*.pt'):
    """Retrieve the path to the most recent checkpoint"""
    files = directory.glob(regex)
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    return files[-1]


def save_checkpoint(model, optimizer, step, file):
    """Save training checkpoint to disk"""
    print(f'Saving model and optimizer at step {step} to {file}')
    checkpoint = {
        'step': step,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()}
    torch.save(checkpoint, file)


def summarize(
    writer,
    step,
    scalars={},
    histograms={},
    images={},
    waveforms={}):
    """Add assets to Tensorboard"""
    for k, v in scalars.items():
        writer.add_scalar(k, v, step)
    for k, v in histograms.items():
        writer.add_histogram(k, v, step)
    for k, v in images.items():
        writer.add_image(k, v, step, dataformats='HWC')
    for k, v in waveforms.items():
        writer.add_audio(k, v, step, promovits.SAMPLE_RATE)


def unpack_to_device(batch, device):
    """Unpack batch and place on device"""
    return (item.to(device) for item in batch)


###############################################################################
# Entry point
###############################################################################


def main(
    config,
    dataset,
    train_partition='train',
    valid_partition='valid',
    adapt=False,
    gpus=None):
    # Optionally overwrite training with same name
    directory = promovits.TRAIN_DIR / config.stem

    # Create output directory
    directory.mkdir(parents=True, exist_ok=True)

    # Save configuration
    shutil.copyfile(config, directory / config.name)

    if gpus is None:

        # CPU training
        train(dataset, directory, train_partition, valid_partition, adapt)

    elif len(gpus) > 1:

        args = (
            dataset,
            directory,
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
            train_partition,
            valid_partition,
            adapt,
            None if gpus is None else gpus[0])


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=Path,
        default=promovits.DEFAULT_CONFIGURATION,
        help='The configuration file')
    parser.add_argument(
        '--dataset',
        default='vctk',
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
