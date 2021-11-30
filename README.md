# ProMoVITS
<!-- [![PyPI](https://img.shields.io/pypi/v/promovits.svg)](https://pypi.python.org/pypi/promovits)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://pepy.tech/badge/promovits)](https://pepy.tech/project/promovits) -->

Official code for the paper _Adaptive End-to-End Speech Prosody Modification_ [[paper]](https://www.maxrmorrison.com/pdfs/morrison2022adaptive.pdf) [[companion website]](https://www.maxrmorrison.com/sites/promovits/)


## Table of contents

- [Installation](#installation)
- [Configuration](#configuration)
- [Generation](#generation)
    * [CLI](#cli)
    * [API](#api)
        * [`promovits.from_audio`](#promovitsfrom_audio)
        * [`promovits.from_file`](#promovitsfrom_file)
        * [`promovits.from_file_to_file`](#promovitsfrom_file_to_file)
- [Reproducing results](#reproducing-results)
    * [Download](#download)
    * [Partition](#partition)
    * [Preprocess](#preprocess)
    * [Train](#train)
    * [Evaluate](#evaluate)
        * [Objective](#objective)
        * [Subjective](#subjective)
- [Running tests](#running-tests)
- [Citation](#citation)


## Installation

`pip install promovits`

TODO - espeak, pyfoal, MAS


## Configuration

TODO

Additional configuration files for experiments described in our paper
can be found in `config/`.


## Generation

### CLI

TODO
Generate from an audio files on disk. `audio_files` and `output_files` can be
lists of files to perform batch generation.

```
python -m promovits \
    --audio_files <audio_files> \
    --output_files <output_files> \
    --checkpoint <checkpoint> \
    --gpu <gpu>
```


### API

#### `promovits.from_audio`

```
"""Perform prosody editing

Arguments
    TODO

Returns
    TODO
"""
```

#### `promovits.from_file`

```
"""Perform prosody editing from files on disk

Arguments
    TODO

Returns
    TODO
"""
```


#### `promovits.from_file_to_file`

```
"""Perform prosody editing from files on disk and save to disk

Arguments
    TODO

Returns
    TODO
"""
```


## Reproducing results

For the following subsections, the arguments are as follows
- `checkpoint` - Path to an existing checkpoint on disk
- `datasets` - A list of datasets to use. Supported datasets are
  `vctk`, `daps`, and `ravdess`.
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

Partitions a dataset into training, validation, and testing partitions. You
should not need to run this, as the partitions used in our work are provided
for each dataset in `promovits/assets/partitions/`.

```
python -m promovits.partition --datasets <datasets>
```

The optional `--overwrite` flag forces the existing partition to be overwritten.


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

You can monitor training via `tensorboard` as follows.

```
tensorboard --logdir runs/train/ --port <port>
```


### Evaluate

#### Objective

TODO
Reports the pitch RMSE (in cents), periodicity RMSE, and voiced/unvoiced F1
score. Results are both printed and stored in `eval/objective/`.

```
python -m promovits.evaluate.objective \
    --name <name> \
    --datasets <datasets> \
    --checkpoint <checkpoint> \
    --num <num> \
    --gpu <gpu>
```


#### Subjective

TODO
Generates samples for subjective evaluation. Also performs benchmarking
of generation speed. Results are stored in `eval/subjective/`.

```
python -m promovits.evaluate.subjective \
    --name <name> \
    --datasets <datasets> \
    --checkpoint <checkpoint> \
    --num <num> \
    --gpu <gpu>
```


## Running tests

```
pip install pytest
pytest
```


## Citation

### IEEE
M. Morrison and B. Pardo, "Adaptive End-to-End Speech Prosody Modification," Submitted to ICML 2022, July 2022.


### BibTex

```
@inproceedings{morrison2022adaptive,
    title={Adaptive End-to-End Speech Prosody Modification},
    author={Morrison, Max and Pardo, Bryan},
    booktitle={Submitted to ICML 2022},
    month={July},
    year={2022}
}
```
