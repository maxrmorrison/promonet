import functools

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
    dataset,
    directory,
    train_partition='train',
    valid_partition='valid',
    adapt_from=None,
    gpu=None):
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
    train_loader = promonet.data.loader(dataset, train_partition, gpu)
    valid_loader = promonet.data.loader(dataset, valid_partition, gpu)

    #################
    # Create models #
    #################

    num_speakers = promonet.NUM_SPEAKERS
    generator = promonet.model.Generator(num_speakers).to(device)
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

    # Automatic mixed precision (amp) gradient scaler
    scaler = torch.cuda.amp.GradScaler()

    # Get total number of steps
    if adapt_from:
        steps = promonet.STEPS + promonet.ADAPTATION_STEPS
    else:
        steps = promonet.STEPS

    # Determine which stage of two-stage model we are on
    if promonet.MODEL == 'two-stage':

        # Freeze weights
        for param in generator.parameters():
            param.requires_grad = False

        if (
            step < promonet.STEPS // 2 or
            (step >= promonet.STEPS and
             step - promonet.STEPS < promonet.ADAPTATION_STEPS // 2)
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
    progress = torchutil.iterator(
        range(step, steps),
        f'{"Train" if adapt_from is None else "Adapt"}ing {promonet.CONFIG}',
        initial=step,
        total=steps)
    while step < steps:

        # Seed sampler
        train_loader.batch_sampler.set_epoch(epoch)

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


            # Skip examples that are too short
            if audio.shape[-1] < promonet.CHUNK_SIZE:
                continue

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
                mels = promonet.preprocess.spectrogram.linear_to_mel(
                    spectrograms)

                # Compute mels of generated audio
                generated_mels = promonet.preprocess.spectrogram.from_audio(
                    generated,
                    True)

                if promonet.SLICING:

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
                    if 'ppg' in promonet.INPUT_FEATURES:
                        phoneme_slices = slice_fn(phonemes)
                    else:
                        phoneme_slices = None

                    # Slice ground truth audio
                    audio = promonet.model.slice_segments(
                        audio,
                        slice_indices * promonet.HOPSIZE,
                        promonet.CHUNK_SIZE)

                else:
                    print(f"Input shape: {audio.shape}")
                    print(f"Generated shape: {generated.shape}")
                    print(f"Pitch shape: {pitch.shape}")
                    mel_slices = mels
                    pitch_slices = pitch
                    periodicity_slices = periodicity
                    loudness_slices = loudness
                    if 'ppg' in promonet.INPUT_FEATURES:
                        phoneme_slices = phonemes
                    else:
                        phoneme_slices = None

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

                        if promonet.MULTI_MEL_LOSS:
                            mel_loss = promonet.loss.multimel(audio, generated)
                            generator_losses += promonet.MEL_LOSS_WEIGHT / len(promonet.MULTI_MEL_LOSS_WINDOWS) * mel_loss
                        else:
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

            if step % promonet.EVALUATION_INTERVAL == 0:

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
                torchutil.tensorboard.update(directory, step, scalars=scalars)

                # Evaluate on validation data
                with promonet.generation_context(generator):
                    evaluation_steps = (
                        None if step == promonet.STEPS
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
            if step >= promonet.STEPS:
                break

            # Raise if GPU tempurature exceeds 80 C
            if any(gpu.temperature > 80. for gpu in GPUtil.getGPUs()):
                raise RuntimeError(f'GPU is overheating. Terminating training.')

            ###########
            # Updates #
            ###########

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

    if promonet.MODEL != 'vits':

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
                phonemes.squeeze(0),
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
            spectrograms=spectrogram)

        # Log generated audio
        key = f'reconstruction/{i:02d}'
        waveforms[f'{key}-audio'] = generated[0]

        # Get prosody features
        (
            predicted_pitch,
            predicted_periodicity,
            predicted_loudness,
            predicted_phonemes
        ) = promonet.preprocess.from_audio(generated[0], gpu=gpu)

        if promonet.MODEL != 'vits':

            # Plot target and generated prosody
            figures[key] = promonet.plot.from_features(
                generated,
                predicted_pitch,
                predicted_periodicity,
                predicted_loudness,
                predicted_phonemes,
                pitch,
                periodicity,
                loudness,
                phonemes)

            # Update metrics
            metrics[key.split('/')[0]].update(
                pitch,
                periodicity,
                loudness,
                phonemes,
                predicted_pitch,
                predicted_periodicity,
                predicted_loudness,
                predicted_phonemes,
                (text, generated.squeeze()))
        else:

            # Plot generated prosody
            figures[key] = promonet.plot.from_features(
                generated,
                predicted_pitch,
                predicted_periodicity,
                predicted_loudness,
                predicted_phonemes,
                pitch,
                periodicity,
                loudness,
                phonemes)

        ##################
        # Pitch shifting #
        ##################

        if 'pitch' in promonet.INPUT_FEATURES:
            for ratio in promonet.EVALUATION_RATIOS:

                # Shift pitch
                shifted_pitch = ratio * pitch

                # Generate pitch-shifted speech
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
                    predicted_phonemes
                ) = promonet.preprocess.from_audio(shifted[0], gpu=gpu)

                # Log pitch-shifted audio
                key = f'shifted-{int(100 * ratio):03d}/{i:02d}'
                waveforms[f'{key}-audio'] = shifted[0]

                # Log prosody figure
                figures[key] = promonet.plot.from_features(
                    shifted,
                    predicted_pitch,
                    predicted_periodicity,
                    predicted_loudness,
                    predicted_phonemes,
                    shifted_pitch,
                    periodicity,
                    loudness,
                    phonemes)

                # Update metrics
                metrics[key.split('/')[0]].update(
                    shifted_pitch,
                    periodicity,
                    loudness,
                    phonemes,
                    predicted_pitch,
                    predicted_periodicity,
                    predicted_loudness,
                    predicted_phonemes,
                    (text, shifted.squeeze()))

        ###################
        # Time stretching #
        ###################

        if 'ppg' in promonet.INPUT_FEATURES:
            for ratio in promonet.EVALUATION_RATIOS:

                # Stretch representation
                (
                    stretched_pitch,
                    stretched_periodicity,
                    stretched_loudness,
                    stretched_phonemes
                ) = promonet.edit.from_features(
                    pitch,
                    periodicity,
                    loudness,
                    phonemes,
                    time_stretch_ratio=ratio)

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
                    predicted_phonemes
                ) = promonet.preprocess.from_audio(stretched[0], gpu=gpu)

                # Log time-stretched audio
                key = f'stretched-{int(ratio * 100):03d}/{i:02d}'
                waveforms[f'{key}-audio'] = stretched[0]

                # Log prosody figure
                figures[key] = promonet.plot.from_features(
                    stretched,
                    predicted_pitch,
                    predicted_periodicity,
                    predicted_loudness,
                    predicted_phonemes,
                    stretched_pitch,
                    stretched_periodicity,
                    stretched_loudness,
                    stretched_phonemes)

                # Update metrics
                metrics[key.split('/')[0]].update(
                    stretched_pitch,
                    stretched_periodicity,
                    stretched_loudness,
                    stretched_phonemes,
                    predicted_pitch,
                    predicted_periodicity,
                    predicted_loudness,
                    predicted_phonemes,
                    (text, stretched.squeeze()))

        ####################
        # Loudness scaling #
        ####################

        if 'loudness' in promonet.INPUT_FEATURES:
            for ratio in promonet.EVALUATION_RATIOS:

                # Scale loudness
                scaled_loudness = \
                    loudness + promonet.convert.ratio_to_db(ratio)

                # Generate loudness-scaled speech
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
                    predicted_phonemes,
                ) = promonet.preprocess.from_audio(scaled[0], gpu=gpu)

                # Log loudness-scaled audio
                key = f'scaled-{int(ratio * 100):03d}/{i:02d}'
                waveforms[f'{key}-audio'] = scaled[0]

                # Log prosody figure
                figures[key] = promonet.plot.from_features(
                    scaled,
                    predicted_pitch,
                    predicted_periodicity,
                    predicted_loudness,
                    predicted_phonemes,
                    pitch,
                    periodicity,
                    scaled_loudness,
                    phonemes)

                # Update metrics
                metrics[key.split('/')[0]].update(
                    pitch,
                    periodicity,
                    scaled_loudness,
                    phonemes,
                    predicted_pitch,
                    predicted_periodicity,
                    predicted_loudness,
                    predicted_phonemes,
                    (text, scaled.squeeze()))

        # Stop when we exceed some number of batches
        if evaluation_steps is not None and i + 1 == evaluation_steps:
            break

    # Format prosody metrics
    if promonet.MODEL != 'vits':
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
