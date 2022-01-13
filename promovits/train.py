import argparse
import contextlib
import functools
import math
import os
import shutil
from pathlib import Path

import torch
import torchinfo
import tqdm

import promovits


###############################################################################
# Training interface
###############################################################################


def run(
    dataset,
    checkpoint_directory,
    output_directory,
    log_directory,
    train_partition='train',
    valid_partition='valid',
    adapt=False,
    gpus=None):
    """Run model training"""
    if gpus and len(gpus) > 1:

        args = (
            dataset,
            checkpoint_directory,
            output_directory,
            log_directory,
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

        # Single GPU or CPU training
        train(
            dataset,
            checkpoint_directory,
            output_directory,
            log_directory,
            train_partition,
            valid_partition,
            adapt,
            None if gpus is None else gpus[0])


###############################################################################
# Training
###############################################################################


def train(
    dataset,
    checkpoint_directory,
    output_directory,
    log_directory,
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

    # Get torch device
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')

    #######################
    # Create data loaders #
    #######################

    torch.manual_seed(promovits.RANDOM_SEED)
    train_loader, valid_loader = promovits.data.loaders(
        dataset,
        train_partition,
        valid_partition,
        gpu)

    #################
    # Create models #
    #################

    num_speakers = len(train_loader.dataset.speakers)
    generator = promovits.model.Generator(num_speakers).to(device)
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

    generator_optimizer = promovits.OPTIMIZER(generator.parameters())
    discriminator_optimizer = promovits.OPTIMIZER(discriminators.parameters())

    ##############################
    # Maybe load from checkpoint #
    ##############################

    generator_path = promovits.checkpoint.latest_path(
        checkpoint_directory,
        'generator-*.pt'),
    discriminator_path = promovits.checkpoint.latest_path(
        checkpoint_directory,
        'discriminator-*.pt'),

    # For some reason, returning None from latest_path returns (None,)
    generator_path = None if generator_path == (None,) else generator_path
    discriminator_path = \
        None if discriminator_path == (None,) else discriminator_path

    if generator_path and discriminator_path:

        # Load generator
        (
            generator,
            generator_optimizer,
            step
        ) = promovits.checkpoint.load(
            generator_path[0],
            generator,
            generator_optimizer
        )

        # Load discriminator
        (
            discriminators,
            discriminator_optimizer,
            step
        ) = promovits.checkpoint.load(
            discriminator_path[0],
            discriminators,
            discriminator_optimizer
        )

    else:

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

    # Get total number of steps
    if adapt:
        steps = promovits.NUM_STEPS + promovits.NUM_ADAPTATION_STEPS
    else:
        steps = promovits.NUM_STEPS

    # Setup progress bar
    if not rank:
        progress = tqdm.tqdm(
            initial=step,
            total=steps,
            dynamic_ncols=True,
            desc=f'Training {promovits.CONFIG}')
    while step < steps:

        # Seed sampler
        train_loader.batch_sampler.set_epoch(step // len(train_loader.dataset))

        generator.train()
        discriminators.train()
        for batch in train_loader:

            # Unpack batch
            (
                phonemes,
                phoneme_lengths,
                pitch,
                speakers,
                spectrograms,
                spectrogram_lengths,
                audio,
            ) = (item.to(device) for item in batch)

            with torch.cuda.amp.autocast():

                # Bundle training input
                generator_input = (
                    phonemes,
                    phoneme_lengths,
                    pitch,
                    speakers,
                    spectrograms,
                    spectrogram_lengths)

                # Forward pass through generator
                (
                    generated,
                    latent_mask,
                    slice_indices,
                    durations,
                    attention,
                    prior,
                    predicted_mean,
                    predicted_logstd,
                    true_logstd
                ) = generator(*generator_input)

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
            mel_loss = \
                promovits.MEL_LOSS_WEIGHT * \
                torch.nn.functional.l1_loss(mel_slices, generated_mels)

            # Get KL divergence loss between phonemes and spectrogram
            kl_divergence_loss = promovits.loss.kl(
                prior,
                true_logstd,
                predicted_mean,
                predicted_logstd,
                latent_mask)

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

                    # Log losses
                    scalars = {
                        'loss/generator/total': generator_losses,
                        'loss/discriminator/total': discriminator_losses,
                        'learning_rate':
                            generator_optimizer.param_groups[0]['lr'],
                        'loss/generator/feature-matching':
                            feature_matching_loss,
                        'loss/generator/mels': mel_loss,
                        'loss/generator/kl-divergence':
                            kl_divergence_loss}
                    if durations is not None:
                        scalars['loss/generator/duration'] = \
                            duration_loss
                    scalars.update(
                        {f'loss/generator/adversarial-{i:02d}': value
                        for i, value in enumerate(adversarial_losses)})
                    scalars.update(
                        {f'loss/discriminator/real-{i:02d}': value
                        for i, value in enumerate(real_discriminator_losses)})
                    scalars.update(
                        {f'loss/discriminator/fake-{i:02d}': value
                        for i, value in enumerate(fake_discriminator_losses)})
                    promovits.write.scalars(log_directory, step, scalars)

                    # Log mels and attention matrix
                    figures = {
                        'train/slice/original':
                            promovits.plot.spectrogram(
                                mel_slices[0].data.cpu().numpy()),
                        'train/slice/generated':
                            promovits.plot.spectrogram(
                                generated_mels[0].data.cpu().numpy()),
                        'train/original':
                            promovits.plot.spectrogram(
                                mels[0].data.cpu().numpy())}
                    if attention is not None:
                        figures['train/attention'] = \
                            promovits.plot.alignment(
                                attention[0, 0].data.cpu().numpy())
                    promovits.write.figures(log_directory, step, figures)

                ############
                # Evaluate #
                ############

                if step % promovits.EVALUATION_INTERVAL == 0:
                    evaluate(
                        log_directory,
                        step,
                        generator,
                        valid_loader,
                        device)

                ###################
                # Save checkpoint #
                ###################

                if step % promovits.CHECKPOINT_INTERVAL == 0:
                    promovits.checkpoint.save(
                        generator,
                        generator_optimizer,
                        step,
                        output_directory / f'generator-{step:08d}.pt')
                    promovits.checkpoint.save(
                        discriminators,
                        discriminator_optimizer,
                        step,
                        output_directory / f'discriminator-{step:08d}.pt')

            # Update training step count
            if step >= steps:
                break
            step += 1

            # Update progress bar
            if not rank:
                progress.update()

        # Update learning rate every epoch
        generator_scheduler.step()
        discriminator_scheduler.step()

    # Close progress bar
    if not rank:
        progress.close()

    # Save final model
    promovits.checkpoint.save(
        generator,
        generator_optimizer,
        step,
        output_directory / f'generator-{step:08d}.pt')
    promovits.checkpoint.save(
        discriminators,
        discriminator_optimizer,
        step,
        output_directory / f'discriminator-{step:08d}.pt')


###############################################################################
# Evaluation
###############################################################################


def evaluate(directory, step, generator, valid_loader, device):
    """Perform model evaluation"""
    # Prepare generator for evaluation
    generator.eval()

    # Turn off gradient computation
    with torch.no_grad():

        waveforms = {}
        figures = {}
        for i, batch in enumerate(valid_loader):

            # Unpack batch
            (
                phonemes,
                phoneme_lengths,
                pitch,
                speakers,
                spectrogram,
                _,
                audio,
            ) = (item.to(device) for item in batch)

            if not step:

                # Log original audio
                waveforms[f'original/{i:02d}'] = audio[0]

                # Log original melspectrogram
                mels = promovits.preprocess.spectrogram.linear_to_mel(
                    spectrogram[0]).cpu().numpy()
                figures[f'original/{i:02d}'] = \
                    promovits.plot.spectrogram(mels)

            # Generate speech
            generated, *_ = generator(
                phonemes,
                phoneme_lengths,
                pitch,
                speakers)

            # Log generated audio
            waveforms[f'generated/{i:02d}'] = generated[0]

            # Log generated melspectrogram
            figures[f'reconstruction/{i:02d}-generated'] = \
                promovits.plot.spectrogram_from_audio(generated)

            # Maybe log pitch-shifting
            if promovits.PITCH_FEATURES:
                for ratio in [.5, 2.]:

                    # Generate pitch-shifted speech
                    shifted_pitch = ratio * promovits.convert.bins_to_hz(pitch)
                    shifted, *_ = generator(
                        phonemes,
                        phoneme_lengths,
                        promovits.convert.hz_to_bins(shifted_pitch),
                        speakers)

                    # Log pitch-shifted audio
                    key = f'shifted-{int(ratio * 100):03d}/{i:02d}'
                    waveforms[key] = shifted[0]

                    # Log pitch-shifted melspectrogram
                    figures[f'shifted-{int(ratio * 100):03d}/{i:02d}'] = \
                        promovits.plot.spectrogram_from_audio(shifted)

            # Maybe log time-stretching
            if promovits.PPG_FEATURES:
                for ratio in [.5, 2.]:

                    # Generate time-stretched speech
                    stretch = promovits.interpolate.grid.constant(
                        phonemes,
                        ratio)
                    stretched_phonemes = promovits.interpolate.features(
                        phonemes,
                        stretch)
                    stretched_length = phoneme_lengths.to(torch.float) / ratio
                    stretched_length = torch.round(
                        stretched_length + 1e-4).to(torch.long)
                    stretched_pitch = promovits.interpolate.pitch(
                        promovits.convert.bins_to_hz(pitch),
                        stretch)
                    try:
                        stretched, *_ = generator(
                            stretched_phonemes,
                            stretched_length,
                            promovits.convert.hz_to_bins(stretched_pitch),
                            speakers)
                    except RuntimeError as error:
                        import pdb; pdb.set_trace()
                        print(error)

                    # Log time-stretched audio
                    key = f'stretched-{int(ratio * 100):03d}/{i:02d}'
                    waveforms[key] = stretched[0]

                    # Log time-stretched melspectrogram
                    figures[f'stretched-{int(ratio * 100):03d}/{i:02d}'] = \
                        promovits.plot.spectrogram_from_audio(stretched)

            # Maybe log loudness-scaling
            if promovits.LOUDNESS_FEATURES:
                for ratio in [.5, 2.]:

                    # Generate loudness-scaled speech
                    phonemes[:, promovits.PPG_CHANNELS] += 10 * math.log2(ratio)
                    scaled, *_ = generator(
                        phonemes,
                        phoneme_lengths,
                        pitch,
                        speakers)
                    phonemes[:, promovits.PPG_CHANNELS] -= 10 * math.log2(ratio)

                    # Log loudness-scaled audio
                    key = f'scaled-{int(ratio * 100):03d}/{i:02d}'
                    waveforms[key] = scaled[0]

                    # Log loudness-scaled melspectrogram
                    figures[f'scaled-{int(ratio * 100):03d}/{i:02d}'] = \
                        promovits.plot.spectrogram_from_audio(scaled)

    # Write to Tensorboard
    promovits.write.figures(directory, step, figures)
    promovits.write.audio(directory, step, waveforms)

    # Prepare generator for training
    generator.train()


###############################################################################
# Distributed data parallelism
###############################################################################


def train_ddp(rank, dataset, directory, gpus):
    """Train with distributed data parallelism"""
    with ddp_context(rank, len(gpus)):
        train(dataset, directory, gpus)


@contextlib.contextmanager
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
# Entry point
###############################################################################


def main(
    config,
    dataset,
    train_partition='train',
    valid_partition='valid',
    adapt=False,
    gpus=None):
    # Create output directory
    directory = promovits.RUNS_DIR / config.stem
    directory.mkdir(parents=True, exist_ok=True)

    # Save configuration
    shutil.copyfile(config, directory / config.name)

    # Train
    run(dataset,
        directory,
        directory,
        directory,
        train_partition,
        valid_partition,
        adapt,
        gpus)

    # Evaluate
    promovits.evaluate.datasets([dataset], None if gpus is None else gpus[0])


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
