import contextlib
import functools
import math
import os

import matplotlib.pyplot as plt
import pysodic
import torch
import tqdm

import promonet


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
    return promonet.checkpoint.latest_path(output_directory)


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

    # Prevent matplotlib from warning us about all the figures we will have
    # open at once during evaluation. The figures are correctly closed.
    if not rank:
        plt.rcParams.update({'figure.max_open_warning': 100})

    # Get torch device
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')

    #######################
    # Create data loaders #
    #######################

    torch.manual_seed(promonet.RANDOM_SEED)
    train_loader = promonet.data.loader(dataset, train_partition, gpu)
    valid_loader = promonet.data.loader(dataset, valid_partition, gpu)

    #################
    # Create models #
    #################

    num_speakers = 109 #TODO: Handle magic number as config variable
    generator = promonet.model.Generator(num_speakers).to(device)
    discriminators = promonet.model.Discriminator().to(device)

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

    discriminator_optimizer = promonet.OPTIMIZER(discriminators.parameters())
    generator_optimizer = promonet.OPTIMIZER(generator.parameters())

    ##############################
    # Maybe load from checkpoint #
    ##############################

    generator_path = promonet.checkpoint.latest_path(
        checkpoint_directory,
        'generator-*.pt')
    discriminator_path = promonet.checkpoint.latest_path(
        checkpoint_directory,
        'discriminator-*.pt')

    if generator_path and discriminator_path:

        # Load generator
        (
            generator,
            generator_optimizer,
            step
        ) = promonet.checkpoint.load(
            generator_path,
            generator,
            generator_optimizer
        )

        # Load discriminator
        (
            discriminators,
            discriminator_optimizer,
            step
        ) = promonet.checkpoint.load(
            discriminator_path,
            discriminators,
            discriminator_optimizer
        )

    else:

        # Train from scratch
        step = 0

    #########
    # Train #
    #########

    # Automatic mixed precision (amp) gradient scaler
    scaler = torch.cuda.amp.GradScaler()

    # Get total number of steps
    if adapt:
        steps = promonet.NUM_STEPS + promonet.NUM_ADAPTATION_STEPS
    else:
        steps = promonet.NUM_STEPS

    # Determine which stage of two-stage model we are on
    if promonet.MODEL == 'two-stage':

        # Freeze weights
        for param in generator.parameters():
            param.requires_grad = False

        if (
            step < promonet.NUM_STEPS // 2 or
            (step >= promonet.NUM_STEPS and
             step - promonet.NUM_STEPS < promonet.NUM_ADAPTATION_STEPS // 2)
        ):
            promonet.TWO_STAGE_1 = True
            promonet.TWO_STAGE_2 = False

            # Unfreeze synthesizer
            for param in generator.speaker_embedding.parameters():
                param.requires_grad = True
            for param in generator.prior_encoder.parameters():
                param.requires_grad = True
        else:
            promonet.TWO_STAGE_1 = False
            promonet.TWO_STAGE_2 = True

            # Unfreeze vocoder
            for param in generator.speaker_embedding_vocoder.parameters():
                param.requires_grad = True
            for param in generator.vocoder.parameters():
                param.requires_grad = True

    # Setup progress bar
    if not rank:
        progress = tqdm.tqdm(
            initial=step,
            total=steps,
            dynamic_ncols=True,
            desc=f'{"Adapting" if adapt else "Training"} {promonet.CONFIG}')
    while step < steps:

        # Seed sampler
        train_loader.batch_sampler.set_epoch(step // len(train_loader.dataset))

        generator.train()
        discriminators.train()
        for batch in train_loader:

            # Unpack batch
            (
                _,
                phonemes,
                pitch,
                periodicity,
                loudness,
                lengths,
                speakers,
                ratios,
                spectrograms,
                spectrogram_lengths,
                audio,
                _
            ) = batch

            # Copy to device
            (
                phonemes,
                pitch,
                periodicity,
                loudness,
                lengths,
                speakers,
                ratios,
                spectrograms,
                spectrogram_lengths,
                audio
            ) = (
                item.to(device) for item in
                (
                    phonemes,
                    pitch,
                    periodicity,
                    loudness,
                    lengths,
                    speakers,
                    ratios,
                    spectrograms,
                    spectrogram_lengths,
                    audio
                )
            )

            # Bundle training input
            generator_input = (
                phonemes,
                pitch,
                periodicity,
                loudness,
                lengths,
                speakers,
                ratios,
                spectrograms,
                spectrogram_lengths)

            with torch.autocast(device.type):

                # Forward pass through generator
                (
                    generated,
                    latent_mask,
                    slice_indices,
                    predicted_mels,
                    durations,
                    attention,
                    prior,
                    predicted_mean,
                    predicted_logstd,
                    true_logstd
                ) = generator(*generator_input)

                # Convert to mels
                mels = promonet.data.preprocess.spectrogram.linear_to_mel(
                    spectrograms)

                # Slice segments for training discriminator
                segment_size = promonet.convert.samples_to_frames(
                    promonet.CHUNK_SIZE)

                # Slice spectral features
                mel_slices = promonet.model.slice_segments(
                    mels,
                    start_indices=slice_indices,
                    segment_size=segment_size)

                # Slice prosody
                indices, size = slice_indices, segment_size
                slice_fn = functools.partial(
                    promonet.model.slice_segments,
                    start_indices=indices,
                    segment_size=size)
                pitch_slices = slice_fn(pitch, fill_value=pitch.mean())
                periodicity_slices = slice_fn(periodicity)
                loudness_slices = slice_fn(loudness, fill_value=loudness.min())
                if promonet.PPG_FEATURES:
                    phoneme_slices = slice_fn(phonemes)
                else:
                    phoneme_slices = None

                # Compute mels of generated audio
                generated_mels = promonet.data.preprocess.spectrogram.from_audio(
                    generated,
                    True)

                # Slice ground truth audio
                audio = promonet.model.slice_segments(
                    audio,
                    slice_indices * promonet.HOPSIZE,
                    promonet.CHUNK_SIZE)

                #######################
                # Train discriminator #
                #######################

                if not promonet.TWO_STAGE_1:

                    real_logits, fake_logits, _, _ = discriminators(
                        audio,
                        generated.detach(),
                        pitch=pitch_slices,
                        periodicity=periodicity_slices,
                        loudness=loudness_slices,
                        phonemes=phoneme_slices)

                    with torch.autocast(device.type, enabled=False):

                        # Get discriminator loss
                        (
                            discriminator_losses,
                            real_discriminator_losses,
                            fake_discriminator_losses
                        ) = promonet.loss.discriminator(
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

            with torch.autocast(device.type):

                if not promonet.TWO_STAGE_1:
                    (
                        _,
                        fake_logits,
                        real_feature_maps,
                        fake_feature_maps
                    ) = discriminators(
                        audio,
                        generated,
                        pitch=pitch_slices,
                        periodicity=periodicity_slices,
                        loudness=loudness_slices,
                        phonemes=phoneme_slices)

                ####################
                # Generator losses #
                ####################

                with torch.autocast(device.type, enabled=False):

                    generator_losses = 0.

                    if promonet.TWO_STAGE_1:

                        # Get synthesizer loss
                        synthesizer_loss = torch.nn.functional.l1_loss(
                            mel_slices,
                            predicted_mels)
                        generator_losses += synthesizer_loss

                    else:

                        # Get duration loss
                        if durations is not None:
                            duration_loss = torch.sum(durations.float())
                            generator_losses += duration_loss
                        else:
                            duration_loss = 0

                        # Get melspectrogram loss
                        mel_loss = torch.nn.functional.l1_loss(mel_slices, generated_mels)
                        generator_losses +=  promonet.MEL_LOSS_WEIGHT * mel_loss

                        # Get KL divergence loss between features and prior
                        if promonet.MODEL in ['end-to-end', 'vits']:
                            kl_divergence_loss = promonet.loss.kl(
                                prior.float(),
                                true_logstd.float(),
                                predicted_mean.float(),
                                predicted_logstd.float(),
                                latent_mask)
                            generator_losses += promonet.KL_DIVERGENCE_LOSS_WEIGHT * kl_divergence_loss

                        # Get feature matching loss
                        feature_matching_loss = promonet.loss.feature_matching(
                            real_feature_maps,
                            fake_feature_maps)
                        generator_losses += promonet.FEATURE_MATCHING_LOSS_WEIGHT * feature_matching_loss

                        # Get adversarial loss
                        adversarial_loss, adversarial_losses = \
                            promonet.loss.generator(
                                [logit.float() for logit in fake_logits])
                        generator_losses += promonet.ADVERSARIAL_LOSS_WEIGHT * adversarial_loss

                ######################
                # Optimize generator #
                ######################

                generator_optimizer.zero_grad()

                # Backward pass
                scaler.scale(generator_losses).backward()

                # Maybe perform gradient clipping
                if promonet.GRADIENT_CLIP_GENERATOR is not None:

                    # Unscale gradients
                    scaler.unscale_(generator_optimizer)

                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(
                        generator.parameters(),
                        promonet.GRADIENT_CLIP_GENERATOR)

                # Update weights
                scaler.step(generator_optimizer)

                # Update gradient scaler
                scaler.update()

            ###########
            # Logging #
            ###########

            if not rank:

                if step % promonet.LOG_INTERVAL == 0:

                    # Log training losses
                    scalars = {
                        'loss/generator/total': generator_losses}
                    if promonet.TWO_STAGE_1:
                        scalars['loss/synthesizer'] = synthesizer_loss
                    else:
                        scalars.update({
                        'loss/discriminator/total': discriminator_losses,
                        'loss/generator/feature-matching':
                            feature_matching_loss,
                        'loss/generator/mels': mel_loss})
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
                        if promonet.MODEL not in ['hifigan', 'two-stage', 'vocoder']:
                            scalars['loss/generator/kl-divergence'] = kl_divergence_loss
                    promonet.write.scalars(log_directory, step, scalars)

                    # Evaluate on validation data
                    evaluate(
                        log_directory,
                        step,
                        generator,
                        valid_loader,
                        gpu)

                ###################
                # Save checkpoint #
                ###################

                if step and step % promonet.CHECKPOINT_INTERVAL == 0:
                    promonet.checkpoint.save(
                        generator,
                        generator_optimizer,
                        step,
                        output_directory / f'generator-{step:08d}.pt')
                    promonet.checkpoint.save(
                        discriminators,
                        discriminator_optimizer,
                        step,
                        output_directory / f'discriminator-{step:08d}.pt')

            # Maybe finish training
            if step >= steps:
                break

            # Two-stage model transition from training synthesizer to vocoder
            if promonet.MODEL == 'two-stage' and step == steps // 2:
                promonet.TWO_STAGE_1 = False
                promonet.TWO_STAGE_2 = True

                # Freeze all weights
                for param in generator.parameters():
                    param.requires_grad = False

                # Unfreeze vocoder
                for param in generator.speaker_embedding_vocoder.parameters():
                    param.requires_grad = True
                for param in generator.vocoder.parameters():
                    param.requires_grad = True

            if not rank:

                # Increment steps
                step += 1

                # Update progress bar
                progress.update()

    if not rank:

        # Close progress bar
        progress.close()

        # Save final model
        promonet.checkpoint.save(
            generator,
            generator_optimizer,
            step,
            output_directory / f'generator-{step:08d}.pt')
        promonet.checkpoint.save(
            discriminators,
            discriminator_optimizer,
            step,
            output_directory / f'discriminator-{step:08d}.pt')


###############################################################################
# Evaluation
###############################################################################


def evaluate(directory, step, generator, loader, gpu):
    """Perform model evaluation"""
    device = 'cpu' if gpu is None else f'cuda:{gpu}'

    if promonet.MODEL != 'vits':

        # Setup metrics
        metric_fn = functools.partial(
            promonet.evaluate.Metrics,
            gpu)

        # Reconstruction
        metrics = {'reconstruction': metric_fn()}

        # Editing
        if promonet.PITCH_FEATURES:
            metrics.update({
                'shifted-050': metric_fn(),
                'shifted-200': metric_fn()})
        if promonet.PPG_FEATURES:
            metrics.update({
                'stretched-050': metric_fn(),
                'stretched-200': metric_fn()})
        if promonet.LOUDNESS_FEATURES:
            metrics.update({
                'scaled-050': metric_fn(),
                'scaled-200': metric_fn()})

    # Audio, figures, and scalars for tensorboard
    waveforms, figures, scalars = {}, {}, {}

    # Setup model for inference
    with promonet.generation_context(generator):

        for i, batch in enumerate(loader):

            # Unpack
            (
                text,
                phonemes,
                pitch,
                periodicity,
                loudness,
                lengths,
                speakers,
                _,
                spectrogram,
                _,
                audio,
                _
            ) = batch
            text = text[0]

            # Copy to device
            (
                phonemes,
                pitch,
                periodicity,
                loudness,
                lengths,
                speakers,
                spectrogram,
                audio
            ) = (
                item.to(device) for item in (
                    phonemes,
                    pitch,
                    periodicity,
                    loudness,
                    lengths,
                    speakers,
                    spectrogram,
                    audio
                )
            )

            # Ensure audio and generated are same length
            trim = audio.shape[-1] % promonet.HOPSIZE
            if trim > 0:
                audio = audio[..., :-trim]

            # Log original audio on first evaluation
            if step == 0:
                waveforms[f'original/{i:02d}-audio'] = audio[0]

            # Extract prosody features
            (
                pitch,
                periodicity,
                loudness,
                voicing,
                phones,
                _
            ) = pysodic.from_audio_and_text(
                audio[0],
                promonet.SAMPLE_RATE,
                text,
                promonet.HOPSIZE / promonet.SAMPLE_RATE,
                promonet.WINDOW_SIZE / promonet.SAMPLE_RATE,
                gpu=gpu
            )

            ##################
            # Reconstruction #
            ##################

            # Generate
            generated, *_ = generator(
                phonemes,
                pitch,
                periodicity,
                loudness,
                lengths,
                speakers,
                spectrograms=spectrogram,
            )

            # Log generated audio
            key = f'reconstruction/{i:02d}'
            waveforms[f'{key}-audio'] = generated[0]

            # Get prosody features
            (
                predicted_pitch,
                predicted_periodicity,
                predicted_loudness,
                predicted_voicing,
                predicted_phones,
                predicted_alignment
            ) = pysodic.from_audio_and_text(
                generated[0],
                promonet.SAMPLE_RATE,
                text,
                promonet.HOPSIZE / promonet.SAMPLE_RATE,
                promonet.WINDOW_SIZE / promonet.SAMPLE_RATE,
                gpu=gpu
            )

            # Get ppgs
            predicted_phonemes = infer_ppgs(generated, phonemes.shape[-1], gpu)

            if promonet.MODEL != 'vits':

                # Plot target and generated prosody
                figures[key] = promonet.plot.from_features(
                    generated,
                    predicted_pitch,
                    predicted_periodicity,
                    predicted_loudness,
                    predicted_alignment,
                    pitch,
                    periodicity,
                    loudness)

                # Update metrics
                metrics[key.split('/')[0]].update(
                    (
                        pitch,
                        periodicity,
                        loudness,
                        voicing,
                        predicted_pitch,
                        predicted_periodicity,
                        predicted_loudness,
                        predicted_voicing,
                        phones,
                        predicted_phones
                    ),
                    (phonemes, predicted_phonemes),
                    (text, generated.squeeze()))
            else:

                # Plot generated prosody
                figures[key] = promonet.plot.from_features(
                    generated,
                    predicted_pitch,
                    predicted_periodicity,
                    predicted_loudness,
                    predicted_alignment)

            ##################
            # Pitch shifting #
            ##################

            if promonet.PITCH_FEATURES:
                for ratio in [.5, 2.]:

                    # Shift pitch
                    shifted_pitch = ratio * pitch

                    # Generate pitch-shifted speech
                    shifted, *_ = generator(
                        phonemes,
                        shifted_pitch,
                        periodicity,
                        loudness,
                        lengths,
                        speakers,
                    )

                    # Get prosody features
                    (
                        predicted_pitch,
                        predicted_periodicity,
                        predicted_loudness,
                        predicted_voicing,
                        predicted_phones,
                        predicted_alignment
                    ) = pysodic.from_audio_and_text(
                        shifted[0],
                        promonet.SAMPLE_RATE,
                        text,
                        promonet.HOPSIZE / promonet.SAMPLE_RATE,
                        promonet.WINDOW_SIZE / promonet.SAMPLE_RATE,
                        gpu=gpu
                    )

                    # Get ppgs
                    predicted_phonemes = infer_ppgs(shifted, phonemes.shape[2], gpu)

                    # Log pitch-shifted audio
                    key = f'shifted-{int(ratio * 100):03d}/{i:02d}'
                    waveforms[f'{key}-audio'] = shifted[0]

                    # Log prosody figure
                    figures[key] = promonet.plot.from_features(
                        shifted,
                        predicted_pitch,
                        predicted_periodicity,
                        predicted_loudness,
                        predicted_alignment,
                        shifted_pitch,
                        periodicity,
                        loudness)

                    # Update metrics
                    metrics[key.split('/')[0]].update(
                        (
                            shifted_pitch,
                            periodicity,
                            loudness,
                            voicing,
                            predicted_pitch,
                            predicted_periodicity,
                            predicted_loudness,
                            predicted_voicing,
                            phones,
                            predicted_phones
                        ),
                        (phonemes, predicted_phonemes),
                        (text, shifted.squeeze()))

            ###################
            # Time stretching #
            ###################

            if promonet.PPG_FEATURES:
                for ratio in [.5, 2.]:

                    # Create time-stretch grid
                    grid = promonet.interpolate.grid.constant(
                        phonemes,
                        ratio)

                    # Stretch phonetic posteriorgram
                    stretched_phonemes = promonet.interpolate.ppgs(
                        phonemes,
                        grid)

                    # Stretch prosody features
                    stretched_pitch = promonet.interpolate.pitch(pitch, grid)
                    stretched_periodicity = promonet.interpolate.grid_sample(
                        periodicity,
                        grid)
                    stretched_loudness = promonet.interpolate.grid_sample(
                        loudness,
                        grid)
                    stretched_voicing = pysodic.features.voicing(
                        stretched_periodicity)
                    stretched_phones = promonet.interpolate.grid_sample(
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
                        speakers,
                    )

                    # Get prosody features
                    (
                        predicted_pitch,
                        predicted_periodicity,
                        predicted_loudness,
                        predicted_voicing,
                        predicted_phones,
                        predicted_alignment
                    ) = pysodic.from_audio_and_text(
                        stretched[0],
                        promonet.SAMPLE_RATE,
                        text,
                        promonet.HOPSIZE / promonet.SAMPLE_RATE,
                        promonet.WINDOW_SIZE / promonet.SAMPLE_RATE,
                        gpu=gpu
                    )

                    # Get ppgs
                    predicted_phonemes = infer_ppgs(
                        stretched,
                        stretched_phonemes.shape[2],
                        gpu)

                    # Log time-stretched audio
                    key = f'stretched-{int(ratio * 100):03d}/{i:02d}'
                    waveforms[f'{key}-audio'] = stretched[0]

                    # Log prosody figure
                    figures[key] = promonet.plot.from_features(
                        stretched,
                        predicted_pitch,
                        predicted_periodicity,
                        predicted_loudness,
                        predicted_alignment,
                        stretched_pitch,
                        stretched_periodicity,
                        stretched_loudness)

                    # Update metrics
                    metrics[key.split('/')[0]].update(
                        (
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
                        ),
                        (stretched_phonemes, predicted_phonemes),
                        (text, stretched.squeeze()))

            ####################
            # Loudness scaling #
            ####################

            if promonet.LOUDNESS_FEATURES:
                for ratio in [.5, 2.]:

                    # Scale loudness
                    scaled_loudness = loudness + 10 * math.log2(ratio)

                    # Generate loudness-scaled speech
                    scaled, *_ = generator(
                        phonemes,
                        pitch,
                        periodicity,
                        scaled_loudness,
                        lengths,
                        speakers,
                    )

                    # Get prosody features
                    (
                        predicted_pitch,
                        predicted_periodicity,
                        predicted_loudness,
                        predicted_voicing,
                        predicted_phones,
                        predicted_alignment
                    ) = pysodic.from_audio_and_text(
                        scaled[0],
                        promonet.SAMPLE_RATE,
                        text,
                        promonet.HOPSIZE / promonet.SAMPLE_RATE,
                        promonet.WINDOW_SIZE / promonet.SAMPLE_RATE,
                        gpu=gpu
                    )

                    # Get ppgs
                    predicted_phonemes = infer_ppgs(scaled, phonemes.shape[2], gpu)

                    # Log loudness-scaled audio
                    key = f'scaled-{int(ratio * 100):03d}/{i:02d}'
                    waveforms[f'{key}-audio'] = scaled[0]

                    # Log prosody figure
                    figures[key] = promonet.plot.from_features(
                        scaled,
                        predicted_pitch,
                        predicted_periodicity,
                        predicted_loudness,
                        predicted_alignment,
                        pitch,
                        periodicity,
                        scaled_loudness)

                    # Update metrics
                    metrics[key.split('/')[0]].update(
                        (
                            pitch,
                            periodicity,
                            scaled_loudness,
                            voicing,
                            predicted_pitch,
                            predicted_periodicity,
                            predicted_loudness,
                            predicted_voicing,
                            phones,
                            predicted_phones
                        ),
                        (phonemes, predicted_phonemes),
                        (text, scaled.squeeze()))

    # Format prosody metrics
    if promonet.MODEL != 'vits':
        for condition in metrics:
            for key, value in metrics[condition]().items():
                scalars[f'{condition}/{key}'] = value

    # Write to Tensorboard
    promonet.write.audio(directory, step, waveforms)
    promonet.write.figures(directory, step, figures)
    promonet.write.scalars(directory, step, scalars)


def infer_ppgs(audio, size, gpu=None):
    """Extract aligned PPGs"""
    predicted_phonemes = promonet.data.preprocess.ppg.from_audio(
        audio[0],
        sample_rate=promonet.SAMPLE_RATE,
        gpu=gpu)
    # Maybe resample length
    mode = promonet.PPG_INTERP_METHOD
    return torch.nn.functional.interpolate(
        predicted_phonemes, #TODO check [none] ?
        size=size,
        mode=mode,
        align_corners=None if mode == 'nearest' else False)


###############################################################################
# Distributed data parallelism
###############################################################################


def train_ddp(
    rank,
    dataset,
    checkpoint_directory,
    output_directory,
    log_directory,
    train_partition,
    valid_partition,
    adapt,
    gpus):
    """Train with distributed data parallelism"""
    with ddp_context(rank, len(gpus)):
        train(
            dataset,
            checkpoint_directory,
            output_directory,
            log_directory,
            train_partition,
            valid_partition,
            adapt,
            gpus[rank])


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
