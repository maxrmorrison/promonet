import argparse
import contextlib
import functools
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
    directory,
    train_partition='train',
    valid_partition='valid',
    adapt=False,
    gpus=None):
    """Run model training"""
    if gpus and len(gpus) > 1:

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

        # Single GPU or CPU training
        train(
            dataset,
            directory,
            train_partition,
            valid_partition,
            adapt,
            None if gpus is None else gpus[0])

    # Evaluate
    promovits.evaluate.datasets(
        directory,
        [dataset],
        None if gpus is None else gpus[0])


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

    # Get torch device
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')

    # Maybe setup adaptation directory
    if not rank:
        if adapt:
            adapt_directory = (
                directory /
                'adapt' /
                dataset /
                train_partition.split('-')[2])
            adapt_directory.mkdir(exist_ok=True, parents=True)
        else:
            None


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

    # TODO - speaker adaptation
    n_speakers = 1 + max(
        int(speaker) for speaker in train_loader.dataset.speakers)

    generator = promovits.model.Generator(n_speakers).to(device)
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

    generator_path = promovits.checkpoint.latest_path(
        directory,
        'generator-*.pt'),
    discriminator_path = promovits.checkpoint.latest_path(
        directory,
        'discriminator-*.pt'),
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

    # Setup progress bar
    if not rank:
        progress = tqdm.tqdm(
            initial=step,
            total=promovits.NUM_STEPS,
            dynamic_ncols=True,
            desc=f'Training {directory.stem}')
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
                    promovits.write.scalars(directory, step, scalars)

                    # Log mels and attention matrix
                    figures = {
                        'train/mels/slice/original':
                            promovits.plot.spectrogram(
                                mel_slices[0].data.cpu().numpy()),
                        'train/mels/slice/generated':
                            promovits.plot.spectrogram(
                                generated_mels[0].data.cpu().numpy()),
                        'train/mels/original':
                            promovits.plot.spectrogram(
                                mels[0].data.cpu().numpy())}
                    if attention is not None:
                        figures['train/attention'] = \
                            promovits.plot.alignment(
                                attention[0, 0].data.cpu().numpy())
                    promovits.write.figures(directory, step, figures)

                ############
                # Evaluate #
                ############

                if step % promovits.EVALUATION_INTERVAL == 0:
                    evaluate(directory, step, generator, valid_loader, device)

                ###################
                # Save checkpoint #
                ###################

                if step % promovits.CHECKPOINT_INTERVAL == 0:

                    # Maybe save to adaptation directory
                    checkpoint_directory = \
                        adapt_directory if adapt else directory

                    promovits.checkpoint.save(
                        generator,
                        generator_optimizer,
                        step,
                        checkpoint_directory / f'generator-{step:08d}.pt')
                    promovits.checkpoint.save(
                        discriminators,
                        discriminator_optimizer,
                        step,
                        checkpoint_directory / f'discriminator-{step:08d}.pt')

            # Update training step count
            if step >= promovits.NUM_STEPS:
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
        images = {}
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

            # Generate speech
            generated, *_ = generator(
                phonemes,
                phoneme_lengths,
                pitch,
                speakers)

            if not step:

                # Log original audio
                waveforms[f'original/{i:02d}'] = audio[0]

                # Log original melspectrogram
                mels = promovits.preprocess.spectrogram.linear_to_mel(
                    spectrogram[0]).cpu().numpy()
                images[f'mels/{i:02d}-original'] = \
                    promovits.plot.spectrogram(mels)

            # Log generated audio
            waveforms[f'generated/{i:02d}'] = generated[0]

            # Log generated melspectrogram
            generated_mels = promovits.preprocess.spectrogram.from_audio(
                generated.float(),
                True)
            images[f'mels/{i:02d}-generated'] = \
                promovits.plot.spectrogram(generated_mels.cpu().numpy())

    # Write to Tensorboard
    promovits.write.images(directory, step, images)
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
    directory = promovits.RUNS_DIR / config.stem

    # Create output directory
    directory.mkdir(parents=True, exist_ok=True)

    # Save configuration
    shutil.copyfile(config, directory / config.name)

    # Train
    run(
        dataset,
        directory,
        train_partition,
        valid_partition,
        adapt,
        gpus)


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
