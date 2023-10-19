<h1 align="center">Prosody Modification Network (ProMoNet)</h1>
<div align="center">

[![PyPI](https://img.shields.io/pypi/v/promonet.svg)](https://pypi.python.org/pypi/promonet)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://pepy.tech/badge/promonet)](https://pepy.tech/project/promonet)

</div>

Official code for the paper _Adaptive Neural Speech Editing_
[[paper]](https://www.maxrmorrison.com/pdfs/morrison2024adaptive.pdf)
[[companion website]](https://www.maxrmorrison.com/sites/promonet/)


## Table of contents

- [Installation](#installation)
- [Usage](#usage)
- [Application programming interface (API)](#application-programming-interface-api)
    * [Adaptation API](#adaptation-api)
        * [`promonet.adapt.speaker`](#promonetadaptspeaker)
    * [Generation API](#generation-api)
        * [`promonet.from_features`](#promonetfrom_features)
        * [`promonet.from_file`](#promonetfrom_file)
        * [`promonet.from_file_to_file`](#promonetfrom_file_to_file)
        * [`promonet.from_files_to_files`](#promonetfrom_files_to_files)
- [Command-line interface (CLI)](#command-line-interface-cli)
    * [Adaptation CLI](#adaptation-cli)
        * [`promonet.adapt`](#promonetadapt)
    * [Generation CLI](#generation-cli)
        * [`promonet`](#promonet)
- [Training](#training)
    * [Download](#download)
    * [Preprocess](#preprocess)
    * [Partition](#partition)
    * [Train](#train)
    * [Monitor](#monitor)
    * [Evaluate](#evaluate)
- [Citation](#citation)


## Installation

`pip install promonet`


## Usage

To use `promonet` for speech editing, you must first perform speaker
adaptation on a dataset of recordings of the target speaker. You can then use
the resulting model checkpoint to perform speech editing in the target
speaker's voice. All of this can be done using either the API or CLI.

```python
import promonet


###############################################################################
# Speaker adaptation
###############################################################################


# Speaker's name
name = 'max'

# Audio files for adaptation
files = [...]

# Perform speaker adaptation
checkpoint = promonet.adapt.speaker(name, files)


###############################################################################
# Prosody editing
###############################################################################


# Load audio for prosody editing
audio = promonet.load.audio('test.wav')

# GPU index to perform editing on
gpu = 0

# Get prosody features to edit
pitch, periodicity, loudness, ppg = promonet.preprocess(
    audio.to(f'cuda:{gpu}'))

# We'll use a ratio of 2.0 for all prosody editing examples
ratio = 2.0

# Perform pitch-shifting
shifted = promonet.from_features(
    *promonet.edit.from_features(
        pitch,
        periodicity,
        loudness,
        ppg,
        pitch_shift_cents=promonet.convert.ratio_to_cents(ratio)),
    checkpoint=checkpoint,
    gpu=gpu)

# Perform time-stretching
stretched = promonet.from_features(
    *promonet.edit.from_features(
        pitch,
        periodicity,
        loudness,
        ppg,
        time_stretch_ratio=ratio),
    checkpoint=checkpoint,
    gpu=gpu)

# Perform loudness-scaling
scaled = promonet.from_audio(
    *promonet.edit.from_features(
        pitch,
        periodicity,
        loudness,
        ppg,
        loudness_scale_db=promonet.convert.ratio_to_db(ratio)),
    checkpoint=checkpoint,
    gpu=gpu)
```


## Application programming interface (API)

### Adaptation API

#### `promonet.adapt.speaker`

```python
def speaker(
    name: str,
    files: List[Path],
    checkpoint: Path = promonet.DEFAULT_CHECKPOINT,
    gpu: Optional[int] = None
) -> Path:
    """Perform speaker adaptation

    Args:
        name: The name of the speaker
        files: The audio files to use for adaptation
        checkpoint: The model checkpoint
        gpu: The gpu to run adaptation on

    Returns:
        checkpoint: The file containing the trained generator checkpoint
    """
```


### Generation API

##### `promonet.from_features`

```python
def from_features(
    pitch: torch.Tensor,
    periodicity: torch.Tensor,
    loudness: torch.Tensor,
    ppg: torch.Tensor,
    speaker: Optional[Union[int, torch.Tensor]] = 0,
    checkpoint: Union[str, os.PathLike] = promonet.DEFAULT_CHECKPOINT,
    gpu: Optional[int] = None) -> torch.Tensor:
    """Perform speech editing

    Args:
        pitch: The pitch contour
        periodicity: The periodicity contour
        loudness: The loudness contour
        ppg: The phonetic posteriorgram
        speaker: The speaker index
        checkpoint: The generator checkpoint
        gpu: The GPU index

    Returns
        generated: The generated speech
    """
```


##### `promonet.from_file`

```python
def from_file(
    pitch_file: Union[str, os.PathLike],
    periodicity_file: Union[str, os.PathLike],
    loudness_file: Union[str, os.PathLike],
    ppg_file: Union[str, os.PathLike],
    speaker: Optional[Union[int, torch.Tensor]] = 0,
    checkpoint: Union[str, os.PathLike] = promonet.DEFAULT_CHECKPOINT,
    gpu: Optional[int] = None
) -> torch.Tensor:
    """Edit speech on disk

    Args:
        pitch_file: The pitch file
        periodicity_file: The periodicity file
        loudness_file: The loudness file
        ppg_file: The phonetic posteriorgram file
        speaker: The speaker index
        checkpoint: The generator checkpoint
        gpu: The GPU index

    Returns
        generated: The generated speech
    """
```


##### `promonet.from_file_to_file`

```python
def from_file_to_file(
    pitch_file: Union[str, os.PathLike],
    periodicity_file: Union[str, os.PathLike],
    loudness_file: Union[str, os.PathLike],
    ppg_file: Union[str, os.PathLike],
    output_file: Union[str, os.PathLike],
    speaker: Optional[Union[int, torch.Tensor]] = 0,
    checkpoint: Union[str, os.PathLike] = promonet.DEFAULT_CHECKPOINT,
    gpu: Optional[int] = None
) -> None:
    """Edit speech on disk and save to disk

    Args:
        pitch_file: The pitch file
        periodicity_file: The periodicity file
        loudness_file: The loudness file
        ppg_file: The phonetic posteriorgram file
        output_file: The file to save generated speech audio
        speaker: The speaker index
        checkpoint: The generator checkpoint
        gpu: The GPU index
    """
```


##### `promonet.from_files_to_files`

```python
def from_files_to_files(
    pitch_files: List[Union[str, os.PathLike]],
    periodicity_files: List[Union[str, os.PathLike]],
    loudness_files: List[Union[str, os.PathLike]],
    ppg_files: List[Union[str, os.PathLike]],
    output_files: List[Union[str, os.PathLike]],
    speakers: Optional[Union[List[int], torch.Tensor]] = None,
    checkpoint: Union[str, os.PathLike] = promonet.DEFAULT_CHECKPOINT,
    gpu: Optional[int] = None
) -> None:
    """Edit speech on disk and save to disk

    Args:
        pitch_files: The pitch files
        periodicity_files: The periodicity files
        loudness_files: The loudness files
        ppg_files: The phonetic posteriorgram files
        output_files: The files to save generated speech audio
        speakers: The speaker indices
        checkpoint: The generator checkpoint
        gpu: The GPU index
    """
```


## Command-line interface (CLI)

### Adaptation CLI

#### `promonet.adapt`

```
python -m promonet.adapt \
    --name NAME \
    --files FILES \
    [--checkpoint CHECKPOINT] \
    [--gpu GPU]

Perform speaker adaptation

arguments:
  --name NAME
    The name of the speaker
  --files FILES
    The audio files to use for adaptation

optional arguments:
  -h, --help
    show this help message and exit
  --checkpoint CHECKPOINT
    The model checkpoint
  --gpu GPU
    The gpu to run adaptation on
```


### Generation CLI

#### `promonet`

```
python -m promonet \
    --pitch_files PITCH_FILES [PITCH_FILES ...] \
    --periodicity_files PERIODICITY_FILES [PERIODICITY_FILES ...] \
    --loudness_files LOUDNESS_FILES [LOUDNESS_FILES ...] \
    --ppg_files PPG_FILES [PPG_FILES ...] \
    --output_files OUTPUT_FILES [OUTPUT_FILES ...] \
    [--speakers SPEAKERS [SPEAKERS ...]] \
    [--checkpoint CHECKPOINT] \
    [--gpu GPU]

Perform speech editing

arguments:
  --pitch_files PITCH_FILES [PITCH_FILES ...]
    The pitch files
  --periodicity_files PERIODICITY_FILES [PERIODICITY_FILES ...]
    The periodicity files
  --loudness_files LOUDNESS_FILES [LOUDNESS_FILES ...]
    The loudness files
  --ppg_files PPG_FILES [PPG_FILES ...]
    The phonetic posteriorgram files
  --output_files OUTPUT_FILES [OUTPUT_FILES ...]
    The files to save the edited audio

optional arguments:
  -h, --help
    show this help message and exit
  --speakers SPEAKERS [SPEAKERS ...]
    The IDs of the speakers for voice conversion
  --checkpoint CHECKPOINT
    The generator checkpoint
  --gpu GPU
    The GPU index
```


## Training

### Download

Downloads, unzips, and formats datasets. Stores datasets in `data/datasets/`.
Stores formatted datasets in `data/cache/`.

```
python -m promonet.data.download --datasets <datasets>
```


### Preprocess

Prepares features for training. Features are stored in `data/cache/`.

```
python -m promonet.data.preprocess \
    --datasets <datasets> \
    --features <features> \
    --gpu <gpu>
```


### Partition

Partitions a dataset. You should not need to run this, as the partitions
used in our work are provided for each dataset in
`promonet/assets/partitions/`.

```
python -m promonet.partition --datasets <datasets>
```


### Train

Trains a model. Checkpoints and logs are stored in `runs/`.

```
python -m promonet.train \
    --config <config> \
    --dataset <dataset> \
    --gpu <gpu>
```

If the config file has been previously run, the most recent checkpoint will
automatically be loaded and training will resume from that checkpoint.


### Monitor

You can monitor training via `tensorboard`.

```
tensorboard --logdir runs/ --port <port>
```

To use the `torchutil` notification system to receive notifications for long
jobs (download, preprocess, train, and evaluate), set the
`PYTORCH_NOTIFICATION_URL` environment variable to a supported webhook as
explained in [the Apprise documentation](https://pypi.org/project/apprise/).


### Evaluate

Performs objective evaluation and generates examples for subjective evaluation.
Also performs benchmarking of generation speed. Results are stored in `eval/`.

```
python -m promonet.evaluate \
    --config <name> \
    --datasets <datasets> \
    --gpus <gpus>
```


## Citation

### IEEE
M. Morrison, C. Churchwell, N. Pruyne, and B. Pardo, "Adaptive Neural Speech Prosody Editing," Submitted
to ICML 2024, July 2024.


### BibTex

```
@inproceedings{morrison2024adaptive,
    title={Adaptive Neural Speech Prosody Editing},
    author={Morrison, Max and Churchwell, Cameron and Pruyne, Nathan and Pardo, Bryan},
    booktitle={Submitted to ICML 2024},
    month={July},
    year={2024}
}
```
