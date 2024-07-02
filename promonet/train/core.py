import math

import GPUtil
import matplotlib.pyplot as plt
import torch
import torchutil

import promonet


###############################################################################
# Training
###############################################################################


@torchutil.notify('train')
def train(
    directory,
    dataset=promonet.TRAINING_DATASET,
    train_partition='train',
    valid_partition='valid',
    adapt_from=None,
    gpu=None
):
    """Train a model"""
    # Prevent matplotlib from warning us about all the figures we will have
    # open at once during evaluation. The figures are correctly closed.
    plt.rcParams.update({'figure.max_open_warning': 100})

    # Get torch device
    device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')

    #######################
    # Create data loaders #
    #######################

    torch.manual_seed(promonet.RANDOM_SEED)
    train_loader = promonet.data.loader(
        dataset,
        train_partition,
        adapt_from is not None,
        gpu)
    valid_loader = promonet.data.loader(
        dataset,
        valid_partition,
        adapt_from is not None,
        gpu)

    #################
    # Create models #
    #################

    if promonet.SPECTROGRAM_ONLY:
        generator = promonet.model.MelGenerator().to(device)
    else:
        generator = promonet.model.Generator().to(device)
    discriminators = promonet.model.Discriminator().to(device)

    #####################
    # Create optimizers #
    #####################

    discriminator_optimizer = promonet.OPTIMIZER(discriminators.parameters())
    generator_optimizer = promonet.OPTIMIZER(generator.parameters())

    ##############################
    # Maybe load from checkpoint #
    ##############################

    generator_path = torchutil.checkpoint.latest_path(
        directory if adapt_from is None else adapt_from,
        'generator-*.pt')
    discriminator_path = torchutil.checkpoint.latest_path(
        directory if adapt_from is None else adapt_from,
        'discriminator-*.pt')

    if generator_path and discriminator_path:

        # Load generator
        (
            generator,
            generator_optimizer,
            state
        ) = torchutil.checkpoint.load(
            generator_path,
            generator,
            generator_optimizer
        )
        step, epoch = state['step'], state['epoch']

        # Load discriminator
        (
            discriminators,
            discriminator_optimizer,
            _
        ) = torchutil.checkpoint.load(
            discriminator_path,
            discriminators,
            discriminator_optimizer
        )

    else:

        # Train from scratch
        step, epoch = 0, 0

    #########
    # Train #
    #########

    # Get total number of steps
    if adapt_from:
        steps = promonet.STEPS + promonet.ADAPTATION_STEPS
    else:
        steps = promonet.STEPS

    # Automatic mixed precision (amp) gradient scaler
    scaler = torch.cuda.amp.GradScaler()

    # Maybe setup spectral convergence loss
    if promonet.SPECTRAL_CONVERGENCE_LOSS:
        spectral_convergence = \
            promonet.loss.MultiResolutionSpectralConvergence(device)

    # Setup progress bar
    progress = torchutil.iterator(
        range(step, steps),
        f'{"Train" if adapt_from is None else "Adapt"}ing {promonet.CONFIG}',
        initial=step,
        total=steps)
    while step < steps:

        # Seed sampler
        train_loader.batch_sampler.set_epoch(epoch)

        for batch in train_loader:

            # Unpack batch
            (
                _,
                loudness,
                pitch,
                periodicity,
                ppg,
                speakers,
                spectral_balance_ratios,
                loudness_ratios,
                spectrograms,
                audio,
                _
            ) = batch

            # Skip examples that are too short
            if audio.shape[-1] < promonet.CHUNK_SIZE:
                continue

            # Copy to device
            (
                loudness,
                pitch,
                periodicity,
                ppg,
                speakers,
                spectral_balance_ratios,
                loudness_ratios,
                spectrograms,
                audio
            ) = (
                item.to(device) for item in
                (
                    loudness,
                    pitch,
                    periodicity,
                    ppg,
                    speakers,
                    spectral_balance_ratios,
                    loudness_ratios,
                    spectrograms,
                    audio
                )
            )

            # Bundle training input
            if promonet.MODEL == 'cargan':
                previous_samples = audio[..., :promonet.CARGAN_INPUT_SIZE]
                slice_frames = promonet.CARGAN_INPUT_SIZE // promonet.HOPSIZE
            elif promonet.MODEL == 'fargan':
                previous_samples = audio[
                    ...,
                    :promonet.HOPSIZE * promonet.FARGAN_PREVIOUS_FRAMES]
                slice_frames = 0
            else:
                previous_samples = torch.zeros(
                    promonet.HOPSIZE,
                    dtype=audio.dtype,
                    device=audio.device)
                slice_frames = 0
            if promonet.SPECTROGRAM_ONLY:
                generator_input = (
                    spectrograms[..., slice_frames:],
                    speakers,
                    spectral_balance_ratios,
                    loudness_ratios,
                    previous_samples)
            else:
                generator_input = (
                    loudness[..., slice_frames:],
                    pitch[..., slice_frames:],
                    periodicity[..., slice_frames:],
                    ppg[..., slice_frames:],
                    speakers,
                    spectral_balance_ratios,
                    loudness_ratios,
                    previous_samples)

            #######################
            # Train discriminator #
            #######################

            with torch.autocast(device.type):

                # Forward pass through generator
                generated = generator(*generator_input)

                # Evaluate the boundary of autoregressive models
                if promonet.MODEL == 'cargan':
                    generated = torch.cat((previous_samples, generated), dim=1)
                elif promonet.MODEL == 'fargan':
                    generated = torch.cat(
                        (
                            previous_samples,
                            generated[..., previous_samples.shape[-1]:]
                        ),
                        dim=2)

                if step >= promonet.DISCRIMINATOR_START_STEP:

                    # Forward pass through discriminators
                    real_logits, fake_logits, _, _ = discriminators(
                        audio,
                        generated.detach())

                    # Get discriminator loss
                    (
                        discriminator_losses,
                        real_discriminator_losses,
                        fake_discriminator_losses
                    ) = promonet.loss.discriminator(
                        [logit.float() for logit in real_logits],
                        [logit.float() for logit in fake_logits])

            # Backward pass through discriminators
            if step >= promonet.DISCRIMINATOR_START_STEP:
                discriminator_optimizer.zero_grad()
                scaler.scale(discriminator_losses).backward()
                scaler.step(discriminator_optimizer)

            ###################
            # Train generator #
            ###################

            with torch.autocast(device.type):

                if step >= promonet.ADVERSARIAL_LOSS_START_STEP:

                    # Forward pass through discriminator
                    (
                        _,
                        fake_logits,
                        real_feature_maps,
                        fake_feature_maps
                    ) = discriminators(audio, generated)

                # Compute generator losses
                generator_losses = 0.

                if promonet.MEL_LOSS:

                    # Maybe use sparse Mel loss
                    log_dynamic_range_compression_threshold = (
                        promonet.LOG_DYNAMIC_RANGE_COMPRESSION_THRESHOLD
                        if promonet.SPARSE_MEL_LOSS else None)

                    # Compute target Mels
                    mels = promonet.preprocess.spectrogram.linear_to_mel(
                        spectrograms,
                        log_dynamic_range_compression_threshold)

                    # Compute predicted Mels
                    generated_mels = promonet.preprocess.spectrogram.from_audio(
                        generated,
                        True,
                        log_dynamic_range_compression_threshold)

                    # Maybe shift so clipping bound is zero
                    if promonet.SPARSE_MEL_LOSS:
                        mels += promonet.LOG_DYNAMIC_RANGE_COMPRESSION_THRESHOLD
                        generated_mels += \
                            promonet.LOG_DYNAMIC_RANGE_COMPRESSION_THRESHOLD

                    # Mel loss
                    mel_loss = torch.nn.functional.l1_loss(
                        mels,
                        generated_mels)
                    generator_losses += promonet.MEL_LOSS_WEIGHT * mel_loss

                # Spectral convergence loss
                if promonet.SPECTRAL_CONVERGENCE_LOSS:
                    spectral_loss = spectral_convergence(generated, audio)
                    generator_losses += spectral_loss

                # Waveform loss
                if promonet.SIGNAL_LOSS:
                    signal_loss = promonet.loss.signal(audio, generated)
                    generator_losses += promonet.SIGNAL_LOSS_WEIGHT * signal_loss

                if step >= promonet.ADVERSARIAL_LOSS_START_STEP:

                    # Get feature matching loss
                    feature_matching_loss = promonet.loss.feature_matching(
                        real_feature_maps,
                        fake_feature_maps)
                    generator_losses += (
                        promonet.FEATURE_MATCHING_LOSS_WEIGHT *
                        feature_matching_loss)

                    # Get adversarial loss
                    adversarial_loss, adversarial_losses = \
                        promonet.loss.generator(
                            [logit.float() for logit in fake_logits])
                    generator_losses += \
                        promonet.ADVERSARIAL_LOSS_WEIGHT * adversarial_loss

            # Zero gradients
            generator_optimizer.zero_grad()

            # Backward pass
            scaler.scale(generator_losses).backward()

            # Monitor gradient statistics
            gradient_statistics = torchutil.gradients.stats(generator)
            torchutil.tensorboard.update(
                directory,
                step,
                scalars=gradient_statistics)

            # Maybe perform gradient clipping
            if promonet.GRADIENT_CLIP_GENERATOR is not None:

                # Compare maximum gradient to threshold
                max_grad = max(
                    gradient_statistics['gradients/max'],
                    math.abs(gradient_statistics['gradients/min']))
                if max_grad > promonet.GRADIENT_CLIP_GENERATOR:

                    # Unscale gradients
                    scaler.unscale_(generator_optimizer)

                    # Clip
                    torch.nn.utils.clip_grad_norm_(
                        generator.parameters(),
                        promonet.GRADIENT_CLIP_GENERATOR,
                        norm_type='inf')

            # Update weights
            scaler.step(generator_optimizer)

            # Update gradient scaler
            scaler.update()

            ###########
            # Logging #
            ###########

            if step % promonet.EVALUATION_INTERVAL == 0:

                # Log VRAM utilization
                torchutil.tensorboard.update(
                    directory,
                    step,
                    scalars=torchutil.cuda.utilization(device, 'MB'))

                # Log training losses
                scalars = {
                    'loss/generator/total': generator_losses}
                if promonet.MEL_LOSS:
                    scalars.update({'loss/generator/mels': mel_loss})
                if promonet.SIGNAL_LOSS:
                    scalars.update({'loss/generator/signal': signal_loss})
                if promonet.SPECTRAL_CONVERGENCE_LOSS:
                    scalars.update({
                        'loss/generator/spectral-convergence': spectral_loss})
                if step >= promonet.ADVERSARIAL_LOSS_START_STEP:
                    scalars.update({
                    'loss/discriminator/total': discriminator_losses,
                    'loss/generator/feature-matching':
                        feature_matching_loss})
                    scalars.update(
                        {f'loss/generator/adversarial-{i:02d}': value
                        for i, value in enumerate(adversarial_losses)})
                    scalars.update(
                        {f'loss/discriminator/real-{i:02d}': value
                        for i, value in enumerate(real_discriminator_losses)})
                    scalars.update(
                        {f'loss/discriminator/fake-{i:02d}': value
                        for i, value in enumerate(fake_discriminator_losses)})
                torchutil.tensorboard.update(directory, step, scalars=scalars)

                # Evaluate on validation data
                with torchutil.inference.context(generator):
                    evaluation_steps = (
                        None if step == steps
                        else promonet.DEFAULT_EVALUATION_STEPS)
                    evaluate(
                        directory,
                        step,
                        generator,
                        valid_loader,
                        gpu,
                        evaluation_steps)

            ###################
            # Save checkpoint #
            ###################

            if step and step % promonet.CHECKPOINT_INTERVAL == 0:
                torchutil.checkpoint.save(
                    directory / f'generator-{step:08d}.pt',
                    generator,
                    generator_optimizer,
                    step=step,
                    epoch=epoch)
                torchutil.checkpoint.save(
                    directory / f'discriminator-{step:08d}.pt',
                    discriminators,
                    discriminator_optimizer,
                    step=step,
                    epoch=epoch)

            ########################
            # Termination criteria #
            ########################

            # Finished training
            if step >= steps:
                break

            # Raise if GPU tempurature exceeds 80 C
            if any(gpu.temperature > 80. for gpu in GPUtil.getGPUs()):
                raise RuntimeError(
                    f'GPU is overheating. Terminating training.')

            ###########
            # Updates #
            ###########

            # Increment steps
            step += 1

            # Update progress bar
            progress.update()
        epoch += 1

    # Close progress bar
    progress.close()

    # Save final model
    torchutil.checkpoint.save(
        directory / f'generator-{step:08d}.pt',
        generator,
        generator_optimizer,
        step=step,
        epoch=epoch)
    torchutil.checkpoint.save(
        directory / f'discriminator-{step:08d}.pt',
        discriminators,
        discriminator_optimizer,
        step=step,
        epoch=epoch)


###############################################################################
# Evaluation
###############################################################################


def evaluate(directory, step, generator, loader, gpu, evaluation_steps=None):
    """Perform model evaluation"""
    device = 'cpu' if gpu is None else f'cuda:{gpu}'

    # Setup evaluation metrics
    metrics = {'reconstruction': promonet.evaluate.Metrics()}
    ratios = [
        f'{int(ratio * 100):03d}' for ratio in promonet.EVALUATION_RATIOS]
    if 'pitch' in promonet.INPUT_FEATURES:
        metrics.update({
            f'shifted-{ratio}': promonet.evaluate.Metrics()
            for ratio in ratios})
    if 'ppg' in promonet.INPUT_FEATURES:
        metrics.update({
            f'stretched-{ratio}': promonet.evaluate.Metrics()
            for ratio in ratios})
    if 'loudness' in promonet.INPUT_FEATURES:
        metrics.update({
            f'scaled-{ratio}': promonet.evaluate.Metrics()
            for ratio in ratios})

    # Audio, figures, and scalars for tensorboard
    waveforms, figures, scalars = {}, {}, {}

    # Use default values for augmentation ratios
    spectral_balance_ratios = torch.ones(1, dtype=torch.float, device=device)
    loudness_ratios = torch.ones(1, dtype=torch.float, device=device)

    for i, batch in enumerate(loader):

        # Unpack
        (
            _,
            loudness,
            pitch,
            periodicity,
            ppg,
            speakers,
            _,
            _,
            spectrogram,
            audio,
            _
        ) = batch

        # Copy to device
        (
            loudness,
            pitch,
            periodicity,
            ppg,
            speakers,
            spectrogram,
            audio
        ) = (
            item.to(device) for item in (
                loudness,
                pitch,
                periodicity,
                ppg,
                speakers,
                spectrogram,
                audio
            )
        )

        # Pack global features
        global_features = (
            speakers,
            spectral_balance_ratios,
            loudness_ratios,
            generator.default_previous_samples)

        # Ensure audio and generated are same length
        trim = audio.shape[-1] % promonet.HOPSIZE
        if trim > 0:
            audio = audio[..., :-trim]

        # Log original audio on first evaluation
        if step == 0:
            waveforms[f'original/{i:02d}-audio'] = audio[0]

        ##################
        # Reconstruction #
        ##################

        # Generate
        if promonet.SPECTROGRAM_ONLY:
            generator_input = (spectrogram, *global_features)
        else:
            generator_input = (
                loudness,
                pitch,
                periodicity,
                ppg,
                *global_features)
        generated = generator(*generator_input)

        # Log generated audio
        key = f'reconstruction/{i:02d}'
        waveforms[f'{key}-audio'] = generated[0]

        # Get prosody features
        (
            predicted_loudness,
            predicted_pitch,
            predicted_periodicity,
            predicted_ppg
        ) = promonet.preprocess.from_audio(generated[0], gpu=gpu)

        # Plot target and generated prosody
        if i < promonet.PLOT_EXAMPLES:
            figures[key] = promonet.plot.from_features(
                generated,
                promonet.preprocess.loudness.band_average(predicted_loudness, 1),
                predicted_pitch,
                predicted_periodicity,
                predicted_ppg,
                promonet.preprocess.loudness.band_average(loudness, 1),
                pitch,
                periodicity,
                ppg)

        # Update metrics
        metrics[key.split('/')[0]].update(
            loudness,
            pitch,
            periodicity,
            ppg,
            predicted_loudness,
            predicted_pitch,
            predicted_periodicity,
            predicted_ppg)

        ##################
        # Pitch shifting #
        ##################

        if 'pitch' in promonet.INPUT_FEATURES:
            for ratio in promonet.EVALUATION_RATIOS:

                # Shift pitch
                shifted_pitch = ratio * pitch

                # Generate pitch-shifted speech
                generator_input = (
                    loudness,
                    shifted_pitch,
                    periodicity,
                    ppg,
                    *global_features)
                shifted = generator(*generator_input)

                # Get prosody features
                (
                    predicted_loudness,
                    predicted_pitch,
                    predicted_periodicity,
                    predicted_ppg
                ) = promonet.preprocess.from_audio(shifted[0], gpu=gpu)

                # Log pitch-shifted audio
                key = f'shifted-{int(100 * ratio):03d}/{i:02d}'
                waveforms[f'{key}-audio'] = shifted[0]

                # Log prosody figure
                if i < promonet.PLOT_EXAMPLES:
                    figures[key] = promonet.plot.from_features(
                        shifted,
                        promonet.preprocess.loudness.band_average(predicted_loudness, 1),
                        predicted_pitch,
                        predicted_periodicity,
                        predicted_ppg,
                        promonet.preprocess.loudness.band_average(loudness, 1),
                        shifted_pitch,
                        periodicity,
                        ppg)

                # Update metrics
                metrics[key.split('/')[0]].update(
                    loudness,
                    shifted_pitch,
                    periodicity,
                    ppg,
                    predicted_loudness,
                    predicted_pitch,
                    predicted_periodicity,
                    predicted_ppg)

        ###################
        # Time stretching #
        ###################

        if 'ppg' in promonet.INPUT_FEATURES:
            for ratio in promonet.EVALUATION_RATIOS:

                # Stretch representation
                (
                    stretched_loudness,
                    stretched_pitch,
                    stretched_periodicity,
                    stretched_ppg
                ) = promonet.edit.from_features(
                    loudness,
                    pitch,
                    periodicity,
                    ppg,
                    time_stretch_ratio=ratio)

                # Generate
                generator_input = (
                    stretched_loudness,
                    stretched_pitch,
                    stretched_periodicity,
                    stretched_ppg,
                    *global_features)
                stretched = generator(*generator_input)

                # Get prosody features
                (
                    predicted_loudness,
                    predicted_pitch,
                    predicted_periodicity,
                    predicted_ppg
                ) = promonet.preprocess.from_audio(stretched[0], gpu=gpu)

                # Log time-stretched audio
                key = f'stretched-{int(ratio * 100):03d}/{i:02d}'
                waveforms[f'{key}-audio'] = stretched[0]

                # Log prosody figure
                if i < promonet.PLOT_EXAMPLES:
                    figures[key] = promonet.plot.from_features(
                        stretched,
                        promonet.preprocess.loudness.band_average(predicted_loudness, 1),
                        predicted_pitch,
                        predicted_periodicity,
                        predicted_ppg,
                        promonet.preprocess.loudness.band_average(stretched_loudness, 1),
                        stretched_pitch,
                        stretched_periodicity,
                        stretched_ppg)

                # Update metrics
                metrics[key.split('/')[0]].update(
                    stretched_loudness,
                    stretched_pitch,
                    stretched_periodicity,
                    stretched_ppg,
                    predicted_loudness,
                    predicted_pitch,
                    predicted_periodicity,
                    predicted_ppg)

        ####################
        # Loudness scaling #
        ####################

        if 'loudness' in promonet.INPUT_FEATURES:
            for ratio in promonet.EVALUATION_RATIOS:

                # Scale loudness
                scaled_loudness = \
                    loudness + promonet.convert.ratio_to_db(ratio)

                # Generate loudness-scaled speech
                generator_input = (
                    scaled_loudness,
                    pitch,
                    periodicity,
                    ppg,
                    *global_features)
                scaled = generator(*generator_input)

                # Get prosody features
                (
                    predicted_loudness,
                    predicted_pitch,
                    predicted_periodicity,
                    predicted_ppg
                ) = promonet.preprocess.from_audio(scaled[0], gpu=gpu)

                # Log loudness-scaled audio
                key = f'scaled-{int(ratio * 100):03d}/{i:02d}'
                waveforms[f'{key}-audio'] = scaled[0]

                # Log prosody figure
                if i < promonet.PLOT_EXAMPLES:
                    figures[key] = promonet.plot.from_features(
                        scaled,
                        promonet.preprocess.loudness.band_average(predicted_loudness, 1),
                        predicted_pitch,
                        predicted_periodicity,
                        predicted_ppg,
                        promonet.preprocess.loudness.band_average(scaled_loudness, 1),
                        pitch,
                        periodicity,
                        ppg)

                # Update metrics
                metrics[key.split('/')[0]].update(
                    scaled_loudness,
                    pitch,
                    periodicity,
                    ppg,
                    predicted_loudness,
                    predicted_pitch,
                    predicted_periodicity,
                    predicted_ppg)

        # Stop when we exceed some number of batches
        if evaluation_steps is not None and i + 1 == evaluation_steps:
            break

    # Format prosody metrics
    for condition in metrics:
        for key, value in metrics[condition]().items():
            scalars[f'{condition}/{key}'] = value

    # Write to Tensorboard
    torchutil.tensorboard.update(
        directory,
        step,
        figures=figures,
        scalars=scalars,
        audio=waveforms,
        sample_rate=promonet.SAMPLE_RATE)
