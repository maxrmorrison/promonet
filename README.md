# ProMoVITS
<!-- [![PyPI](https://img.shields.io/pypi/v/promovits.svg)](https://pypi.python.org/pypi/promovits)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://pepy.tech/badge/promovits)](https://pepy.tech/project/promovits) -->

Official code for the paper _Adaptive End-to-End Voice Modification_
[[paper]](https://www.maxrmorrison.com/pdfs/morrison2022adaptive.pdf)
[[companion website]](https://www.maxrmorrison.com/sites/promovits/)


TODO - sphinx documentation

## Installation

```
# Create a new conda environment with MFA
conda create -n promovits -c conda-forge montreal-forced-aligner -y

# Activate conda environment
conda activate promovits

# Option 1) Install from pypi
pip install promovits

# Option 2) Install from source
git clone git@github.com:maxrmorrison/promovits
cd promovits
pip install -e .
```

If you would like to perform text-to-speech, you must also install the `espeak`
backend for the grapheme-to-phoneme module as well compile the code for
monotonic alignment search (MAS).

```
# Install espeak
apt-get install espeak
```

```
# Compile MAS
cd promovits/monotonic_align
python setup.py build_ext --inplace
```


## Configuration

We use [`yapecs`](https://github.com/maxrmorrison/yapecs) for experiment
configuration. Configuration files for experiments described in our paper
can be found in `config/`.


## Usage

To use `promovits` for speech prosody editing, you must first perform speaker
adaptation. To do so, create a directory of `.wav` audio files that will be
used for adaptation. Then, use either the CLI or API to perform adaptation
and generation.


### Command-line interface (CLI)

TODO - give comparable example as API section

```
python -m promovits
    [-h]
    [--config CONFIG]
    --audio_files AUDIO_FILES [AUDIO_FILES ...]
    --output_files OUTPUT_FILES [OUTPUT_FILES ...]
    [--target_alignment_files TARGET_ALIGNMENT_FILES [TARGET_ALIGNMENT_FILES ...]]
    [--target_loudness_files TARGET_LOUDNESS_FILES [TARGET_LOUDNESS_FILES ...]]
    [--target_periodicity_files TARGET_PERIODICITY_FILES [TARGET_PERIODICITY_FILES ...]]
    [--target_pitch_files TARGET_PITCH_FILES [TARGET_PITCH_FILES ...]]
    [--checkpoint CHECKPOINT]
    [--gpu GPU]

Perform prosody editing

required arguments:
  --audio_files AUDIO_FILES [AUDIO_FILES ...]
                        The audio files to process
  --output_files OUTPUT_FILES [OUTPUT_FILES ...]
                        The files to save the output audio

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       The configuration file
  --target_alignment_files TARGET_ALIGNMENT_FILES [TARGET_ALIGNMENT_FILES ...]
                        The files with the target phoneme alignment
  --target_loudness_files TARGET_LOUDNESS_FILES [TARGET_LOUDNESS_FILES ...]
                        The files with the per-frame target loudness
  --target_periodicity_files TARGET_PERIODICITY_FILES [TARGET_PERIODICITY_FILES ...]
                        The files with the per-frame target periodicity
  --target_pitch_files TARGET_PITCH_FILES [TARGET_PITCH_FILES ...]
                        The files with the per-frame target pitch
  --checkpoint CHECKPOINT
                        The generator checkpoint
  --gpu GPU             The index of the gpu to use for generation
```


### Application programming interface (API)

The following is an example of performing speaker adaptation followed by
prosody editing.

```
import promovits


###############################################################################
# Speaker adaptation
###############################################################################


# Speaker's name
name = 'max'

# Directory containing wav files for adaptation
directory = promovits.DATA_DIR / name

# GPU indices to use for training
gpus = [0]

# Perform speaker adaptation
checkpoint = promovits.adapt.speaker(name, directory, gpus=gpus)


###############################################################################
# Prosody editing
###############################################################################


# Load audio for prosody editing
audio = promovits.load.audio('test.wav')

# Get prosody features to edit
pitch, loudness = promovits.preprocess.prosody(audio, gpu=gpus[0])

# (Optional) If you have the text transcript, you can edit phoneme
#            durations using a forced alignment representation
pitch, loudness, alignment = promovits.preprocess.prosody(
    audio,
    text=promovits.load.text('test.txt'),
    gpu=gpus[0])

# We'll use a ratio of 2.0 for all prosody editing examples
ratio = 2.0

# Perform pitch-shifting
target_pitch = ratio * pitch
shifted = promovits.from_audio(
    audio,
    target_pitch=target_pitch,
    checkpoint=checkpoint,
    gpu=gpus[0])

# Perform time-stretching
target_alignment = copy.deepcopy(alignment)
durations = [ratio * p.duration() for p in target_alignment.phonemes()]
target_alignment.update(durations=durations)
stretched = promovits.from_audio(
    audio,
    target_alignment=target_alignment,
    checkpoint=checkpoint,
    gpu=gpus[0])

# Perform loudness-scaling
target_loudness = 10 * math.log2(ratio) + loudness
scaled = promovits.from_audio(
    audio,
    target_loudness=target_loudness,
    checkpoint=checkpoint,
    gpu=gpus[0])
```


## Reproducing results

For the following subsections, the arguments are as follows
- `config` - The configuration file
- `checkpoint` - Path to a checkpoint on disk
- `dataset` - The name of the dataset to use. One of [`vctk` or `daps`].
- `datasets` - A list of datasets to use
- `gpu` - The index of the gpu to use
- `gpus` - A list of indices of gpus to use for distributed data parallelism
  (DDP)
- `num` - The number of samples to evaluate


### Download

Downloads, unzips, and formats datasets. Stores datasets in `data/datasets/`.
Stores formatted datasets in `data/cache/`.

```
python -m promovits.data.download --datasets <datasets>
```


### Preprocess

Prepares features for training. Features are stored in `data/cache/`.

```
python -m promovits.preprocess --datasets <datasets> --gpu <gpu>
```


### Partition

Partitions a dataset. You should not need to run this, as the partitions
used in our work are provided for each dataset in
`promovits/assets/partitions/`.

```
python -m promovits.partition --datasets <datasets>
```

The optional `--overwrite` flag forces the existing partition to be
overwritten.


### Train

Trains a model. Checkpoints and logs are stored in `runs/`.

```
python -m promovits.train \
    --config <config> \
    --dataset <dataset> \
    --gpus <gpus>
```

If the config file has been previously run, the most recent checkpoint will
automatically be loaded and training will resume from that checkpoint.


### Adapt

Adapt a model to a speaker. The files corresponding to a speaker are specified
via disjoint data partitions. Checkpoints and logs are stored in `runs/`.

```
python -m promovits.train \
    --config <config> \
    --dataset <dataset> \
    --train_partition <train_partition> \
    --valid_partition <valid_partition> \
    --adapt \
    --gpus <gpus>
```

If the config file has been previously run, the most recent checkpoint will
automatically be loaded and training will resume from that checkpoint.


### Monitor

You can monitor training via `tensorboard` as follows.

```
tensorboard --logdir runs/train/ --port <port>
```


### Evaluate

Performs objective evaluation and generates examples for subjective evaluation.
Also performs benchmarking of generation speed. Results are stored in `eval/`.

```
python -m promovits.evaluate \
    --config <name> \
    --datasets <datasets> \
    --gpus <gpus>
```


## Running tests

```
pip install pytest
pytest
```


## Citation

### IEEE
M. Morrison and B. Pardo, "Adaptive End-to-End Voice Modification," Submitted
to ICML 2022, July 2022.


### BibTex

```
@inproceedings{morrison2022adaptive,
    title={Adaptive End-to-End Voice Modification},
    author={Morrison, Max and Pardo, Bryan},
    booktitle={Submitted to ICML 2022},
    month={July},
    year={2022}
}
```
