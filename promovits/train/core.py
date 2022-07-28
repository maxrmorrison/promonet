import contextlib
import functools
import math
import os

import pyfoal
import pysodic
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
    # Distributed data parallelism
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

    # Return path to generator checkpoint
    return promovits.checkpoint.latest_path(output_directory)


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
    # printed = False

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
                pitch,
                periodicity,
                loudness,
                lengths,
                speakers,
                spectrograms,
                spectrogram_lengths,
                audio,
            ) = (item.to(device) for item in batch[1:])

            # Bundle training input
            generator_input = (
                phonemes,
                pitch,
                periodicity,
                loudness,
                lengths,
                speakers,
                spectrograms,
                spectrogram_lengths,
                audio)

            # Convert to mels
            mels = promovits.preprocess.spectrogram.linear_to_mel(
                spectrograms)

            with torch.cuda.amp.autocast():

                # Forward pass through generator
                (
                    generated,
                    latent_mask,
                    slice_indices,
                    autoregressive,
                    durations,
                    attention,
                    prior,
                    predicted_mean,
                    predicted_logstd,
                    true_logstd
                ) = generator(*generator_input)

                with torch.cuda.amp.autocast(enabled=False):

                    # Slice segments for training discriminator
                    segment_size = \
                        promovits.CHUNK_SIZE // promovits.HOPSIZE

                    # Slice mels
                    mel_slices = promovits.model.slice_segments(
                        mels,
                        start_indices=slice_indices,
                        segment_size=segment_size)

                    # Update indices to handle autoregression
                    if promovits.AUTOREGRESSIVE:
                        ar_frames = promovits.AR_INPUT_SIZE // promovits.HOPSIZE
                        indices = slice_indices - ar_frames
                        size = segment_size + ar_frames
                    else:
                        indices, size = slice_indices, segment_size

                    # Slice prosody
                    slice_fn = functools.partial(
                        promovits.model.slice_segments,
                        start_indices=indices,
                        segment_size=size)
                    pitch_slices = slice_fn(pitch, fill_value=pitch.mean())
                    periodicity_slices = slice_fn(periodicity)
                    loudness_slices = slice_fn(loudness, fill_value=loudness.min())

                    # Exit autocast context, as ComplexHalf type is not yet supported
                    # See Github issue https://github.com/jaywalnut310/vits/issues/15
                    # For progress, see https://github.com/pytorch/pytorch/issues/74537
                    generated = generated.float()
                    generated_mels = promovits.preprocess.spectrogram.from_audio(
                        generated.float(),
                        True)
                    audio = promovits.model.slice_segments(
                        audio,
                        slice_indices * promovits.HOPSIZE,
                        promovits.CHUNK_SIZE)

                    # Prepend AR input so the discriminator sees boundaries
                    if autoregressive is not None:
                        generated = torch.cat((autoregressive, generated), dim=-1)
                        audio = torch.cat((autoregressive, audio), dim=-1)

                    # Print model summaries first time
                    # if not printed:
                    #     print(torchinfo.summary(
                    #         generator,
                    #         input_data=generator_input,
                    #         device=device))
                    #     print(torchinfo.summary(
                    #         discriminators,
                    #         input_data=(audio, generated.detach()),
                    #         device=device))
                    #     printed = True

                #######################
                # Train discriminator #
                #######################

                real_logits, fake_logits, _, _ = discriminators(
                    audio,
                    generated.detach(),
                    pitch_slices,
                    periodicity_slices,
                    loudness_slices)

                with torch.cuda.amp.autocast(enabled=False):

                    # Get discriminator loss
                    (
                        discriminator_losses,
                        real_discriminator_losses,
                        fake_discriminator_losses
                    ) = promovits.loss.discriminator(
                        [logit.float() for logit in real_logits],
                        [logit.float() for logit in fake_logits])

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
                ) = discriminators(
                    audio,
                    generated,
                    pitch_slices,
                    periodicity_slices,
                    loudness_slices
                )

            ####################
            # Generator losses #
            ####################

                with torch.cuda.amp.autocast(enabled=False):

                    # Get duration loss
                    if durations is not None:
                        duration_loss = torch.sum(durations.float())
                    else:
                        duration_loss = 0

                    # Get melspectrogram loss
                    mel_loss = torch.nn.functional.l1_loss(mel_slices, generated_mels)

                    # Get KL divergence loss between features and spectrogram
                    kl_divergence_loss = promovits.loss.kl(
                        prior.float(),
                        true_logstd.float(),
                        predicted_mean.float(),
                        predicted_logstd.float(),
                        latent_mask)

                    # Get feature matching loss
                    feature_matching_loss = promovits.loss.feature_matching(
                        real_feature_maps,
                        fake_feature_maps)

                    # Get adversarial loss
                    adversarial_loss, adversarial_losses = \
                        promovits.loss.generator(
                            [logit.float() for logit in fake_logits])
                    generator_losses = (
                        adversarial_loss +
                        feature_matching_loss +
                        promovits.MEL_LOSS_WEIGHT * mel_loss +
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

                    # This context manager changes which forced aligner is
                    # used. MFA is slow and less robust to errors than P2FA,
                    # and works better with speaker adaptation, which we don't
                    # perform here. However, the installation of P2FA is
                    # more complicated. Therefore, we allow either aligner
                    # to be used to evaluate training.
                    with pyfoal.backend(promovits.TRAIN_ALIGNER):

                        evaluate(
                            log_directory,
                            step,
                            generator,
                            valid_loader,
                            gpu)

                ###################
                # Save checkpoint #
                ###################

                if step and step % promovits.CHECKPOINT_INTERVAL == 0:
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


def evaluate(directory, step, generator, valid_loader, gpu):
    """Perform model evaluation"""
    device = 'cpu' if gpu is None else f'cuda:{gpu}'

    # Prepare generator for evaluation
    generator.eval()

    # Turn off gradient computation
    with torch.no_grad():

        # Setup prosody evaluation
        metric_fn = functools.partial(
            pysodic.metrics.Prosody,
            promovits.SAMPLE_RATE,
            promovits.HOPSIZE / promovits.SAMPLE_RATE,
            promovits.WINDOW_SIZE / promovits.SAMPLE_RATE,
            gpu=gpu)
        metrics = {'reconstruction': metric_fn()}

        # Maybe add evaluation of prosody control
        if promovits.PITCH_FEATURES:
            metrics.update({
                'shifted-050': metric_fn(),
                'shifted-200': metric_fn()})
        if promovits.PPG_FEATURES:
            metrics.update({
                'stretched-050': metric_fn(),
                'stretched-200': metric_fn()})
        if promovits.LOUDNESS_FEATURES:
            metrics.update({
                'scaled-050': metric_fn(),
                'scaled-200': metric_fn()})

        for i, batch in enumerate(valid_loader):
            waveforms, figures = {}, {}

            # Unpack batch
            text = batch[0][0]
            (
                phonemes,
                pitch,
                periodicity,
                loudness,
                lengths,
                speakers,
                spectrogram,
                _,
                audio,
            ) = (item.to(device) for item in batch[1:])

            # Ensure audio and generated are same length
            audio = audio[..., :-(audio.shape[-1] % promovits.HOPSIZE)]

            if not step:

                # Log original audio
                waveforms[f'original/{i:02d}-audio'] = audio[0]

                # Log original melspectrogram
                mels = promovits.preprocess.spectrogram.linear_to_mel(
                    spectrogram[0]).cpu().numpy()
                figures[f'original/{i:02d}-mels'] = \
                    promovits.plot.spectrogram(mels)

            # Extract prosody features
            try:
                (
                    pitch,
                    periodicity,
                    loudness,
                    voicing,
                    phones,
                    _
                ) = pysodic.from_audio_and_text(
                    audio[0],
                    promovits.SAMPLE_RATE,
                    text,
                    promovits.HOPSIZE / promovits.SAMPLE_RATE,
                    promovits.WINDOW_SIZE / promovits.SAMPLE_RATE,
                    gpu)
            except Exception as error:
                print(error)
                import pdb; pdb.set_trace()
                pass

            # Reconstruct speech
            generated, *_ = generator(
                phonemes,
                pitch,
                periodicity,
                loudness,
                lengths,
                speakers)

            # Get prosody features
            (
                predicted_pitch,
                predicted_periodicity,
                predicted_loudness,
                predicted_voicing,
                predicted_phones,
                _
            ) = pysodic.from_audio_and_text(
                generated[0],
                promovits.SAMPLE_RATE,
                text,
                promovits.HOPSIZE / promovits.SAMPLE_RATE,
                promovits.WINDOW_SIZE / promovits.SAMPLE_RATE,
                gpu
            )

            # Log generated audio
            key = f'reconstruction/{i:02d}'
            metric_args = (
                pitch,
                periodicity,
                loudness,
                voicing,
                predicted_pitch,
                predicted_periodicity,
                predicted_loudness,
                predicted_voicing,
                phones,
                predicted_phones)
            log(generated, key, waveforms, figures, metrics, metric_args)

            # Maybe log pitch-shifting
            if promovits.PITCH_FEATURES:
                for ratio in [.5, 2.]:

                    # Generate pitch-shifted speech
                    shifted_pitch = ratio * pitch
                    shifted, *_ = generator(
                        phonemes,
                        shifted_pitch,
                        periodicity,
                        loudness,
                        lengths,
                        speakers)

                    # Get prosody features
                    (
                        predicted_pitch,
                        predicted_periodicity,
                        predicted_loudness,
                        predicted_voicing,
                        predicted_phones,
                        _
                    ) = pysodic.from_audio_and_text(
                        shifted[0],
                        promovits.SAMPLE_RATE,
                        text,
                        promovits.HOPSIZE / promovits.SAMPLE_RATE,
                        promovits.WINDOW_SIZE / promovits.SAMPLE_RATE,
                        gpu
                    )

                    # Log pitch-shifted audios
                    key = f'shifted-{int(ratio * 100):03d}/{i:02d}'
                    metric_args = (
                        shifted_pitch,
                        periodicity,
                        loudness,
                        voicing,
                        predicted_pitch,
                        predicted_periodicity,
                        predicted_loudness,
                        predicted_voicing,
                        phones,
                        predicted_phones)
                    log(shifted, key, waveforms, figures, metrics, metric_args)

            # Maybe log time-stretching
            if promovits.PPG_FEATURES:
                for ratio in [.5, 2.]:

                    # Create time-stretch grid
                    grid = promovits.interpolate.grid.constant(
                        phonemes,
                        ratio)

                    # Stretch phonetic posteriorgram
                    stretched_phonemes = promovits.interpolate.ppgs(
                        phonemes,
                        grid)

                    # Stretch prosody features
                    stretched_pitch = promovits.interpolate.pitch(pitch, grid)
                    stretched_periodicity = promovits.interpolate.grid_sample(
                        periodicity,
                        grid)
                    stretched_loudness = promovits.interpolate.grid_sample(
                        loudness,
                        grid)
                    stretched_voicing = pysodic.features.voicing(
                        stretched_pitch,
                        stretched_periodicity)
                    stretched_phones = promovits.interpolate.grid_sample(
                        phones,
                        grid,
                        method='nearest')

                    # Stretch feature lengths
                    stretched_length = torch.tensor(
                        [stretched_phonemes.shape[-1]],
                        dtype=torch.long,
                        device=device)

                    # Generate
                    stretched, *_ = generator(
                        stretched_phonemes,
                        stretched_pitch,
                        stretched_periodicity,
                        stretched_loudness,
                        stretched_length,
                        speakers)

                    # Get prosody features
                    (
                        predicted_pitch,
                        predicted_periodicity,
                        predicted_loudness,
                        predicted_voicing,
                        predicted_phones,
                        _
                    ) = pysodic.from_audio_and_text(
                        stretched[0],
                        promovits.SAMPLE_RATE,
                        text,
                        promovits.HOPSIZE / promovits.SAMPLE_RATE,
                        promovits.WINDOW_SIZE / promovits.SAMPLE_RATE,
                        gpu
                    )

                    # Log to tensorboard
                    key = f'stretched-{int(ratio * 100):03d}/{i:02d}'
                    metric_args = (
                        stretched_pitch,
                        stretched_periodicity,
                        stretched_loudness,
                        stretched_voicing,
                        predicted_pitch,
                        predicted_periodicity,
                        predicted_loudness,
                        predicted_voicing,
                        stretched_phones,
                        predicted_phones
                    )
                    log(stretched, key, waveforms, figures, metrics, metric_args)

            # Maybe log loudness-scaling
            if promovits.LOUDNESS_FEATURES:
                for ratio in [.5, 2.]:

                    # Generate loudness-scaled speech
                    scaled_loudness = loudness + 10 * math.log2(ratio)
                    scaled, *_ = generator(
                        phonemes,
                        pitch,
                        periodicity,
                        scaled_loudness,
                        lengths,
                        speakers)

                    # Get prosody features
                    (
                        predicted_pitch,
                        predicted_periodicity,
                        predicted_loudness,
                        predicted_voicing,
                        predicted_phones,
                        _
                    ) = pysodic.from_audio_and_text(
                        scaled[0],
                        promovits.SAMPLE_RATE,
                        text,
                        promovits.HOPSIZE / promovits.SAMPLE_RATE,
                        promovits.WINDOW_SIZE / promovits.SAMPLE_RATE,
                        gpu
                    )

                    # Log loudness-scaled audio
                    key = f'scaled-{int(ratio * 100):03d}/{i:02d}'
                    metric_args = (
                        pitch,
                        periodicity,
                        scaled_loudness,
                        voicing,
                        predicted_pitch,
                        predicted_periodicity,
                        predicted_loudness,
                        predicted_voicing,
                        phones,
                        predicted_phones)
                    log(scaled, key, waveforms, figures, metrics, metric_args)

            # Write audio and mels to Tensorboard
            promovits.write.audio(directory, step, waveforms)
            promovits.write.figures(directory, step, figures)

    # Format prosody metrics
    scalars = {}
    for condition in metrics:
        for name, value in metrics[condition]().items():

            if name == 'voicing':

                # Write precision, recall, and f1 metrics
                for subname, subvalue in value.items():
                    key = f'{condition}/{name}-{subname}'
                    scalars[key] = subvalue

            else:

                # Write metric
                key = f'{condition}/{name}'
                scalars[key] = value

    # Write prosody metrics to tensorboard
    promovits.write.scalars(directory, step, scalars)

    # Prepare generator for training
    generator.train()


###############################################################################
# Logging
###############################################################################


def log(audio, key, waveforms, figures, metrics, metric_args):
    """Update training logs"""
    audio = audio[0]

    # Log time-stretched audio
    waveforms[f'{key}-audio'] = audio

    # Log time-stretched melspectrogram
    figures[f'{key}-mels'] = promovits.plot.spectrogram_from_audio(audio)

    # Update metrics
    metrics[key.split('/')[0]].update(*metric_args)


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
