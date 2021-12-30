# ProMoVITS
<!-- [![PyPI](https://img.shields.io/pypi/v/promovits.svg)](https://pypi.python.org/pypi/promovits)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://pepy.tech/badge/promovits)](https://pepy.tech/project/promovits) -->

Official code for the paper _Adaptive End-to-End Voice Modification_
[[paper]](https://www.maxrmorrison.com/pdfs/morrison2022adaptive.pdf)
[[companion website]](https://www.maxrmorrison.com/sites/promovits/)


TODO - sphinx documentation

## Installation

`pip install promovits`

TODO - espeak, pyfoal, MAS


## Configuration

TODO - add link
We use `yapem` for experiment configuration. Configuration files for
experiments described in our paper can be found in `config/`.


## Generation

### CLI

TODO
Generate from audio files on disk. `audio_files` and `output_files` can be
lists of files to perform batch generation.

```
python -m promovits
    [-h]
    --config CONFIG
    --audio_files AUDIO_FILES [AUDIO_FILES ...]
    --output_files OUTPUT_FILES [OUTPUT_FILES ...]
    [--target_alignment_files TARGET_ALIGNMENT_FILES [TARGET_ALIGNMENT_FILES ...]]
    [--target_loudness_files TARGET_LOUDNESS_FILES [TARGET_LOUDNESS_FILES ...]]
    [--target_periodicity_files TARGET_PERIODICITY_FILES [TARGET_PERIODICITY_FILES ...]]
    [--target_pitch_files TARGET_PITCH_FILES [TARGET_PITCH_FILES ...]]
    [--checkpoint CHECKPOINT]
    [--gpu GPU]

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       The configuration file
  --audio_files AUDIO_FILES [AUDIO_FILES ...]
                        The audio files to process
  --output_files OUTPUT_FILES [OUTPUT_FILES ...]
                        The files to save the output audio
  --target_alignment_files TARGET_ALIGNMENT_FILES [TARGET_ALIGNMENT_FILES ...]
                        The files with the target phoneme alignment
  --target_loudness_files TARGET_LOUDNESS_FILES [TARGET_LOUDNESS_FILES ...]
                        The files with the per-frame target loudness
  --target_periodicity_files TARGET_PERIODICITY_FILES [TARGET_PERIODICITY_FILES ...]
                        The files with the per-frame target periodicity
  --target_pitch_files TARGET_PITCH_FILES [TARGET_PITCH_FILES ...]
                        The files with the per-frame target pitch
  --checkpoint CHECKPOINT
  --gpu GPU             The index of the gpu to use for generation
```


### API

TODO - example usage

```
import promovits
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


### Montoring

You can monitor training via `tensorboard` as follows.

```
tensorboard --logdir runs/train/ --port <port>
```


### Evaluate


Performs objective evaluation and generates examples for subjective evaluation.
Also performs benchmarking of generation speed. Results are stored in `eval/`.

```
python -m promovits.evaluate.subjective \
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
M. Morrison and B. Pardo, "Adaptive End-to-End Voice Modification," Submitted to ICML 2022, July 2022.


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
