<h1 align="center">Prosody Modification Network (ProMoNet)</h1>
<div align="center">

[![PyPI](https://img.shields.io/pypi/v/promonet.svg)](https://pypi.python.org/pypi/promonet)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://pepy.tech/badge/promonet)](https://pepy.tech/project/promonet)

</div>

Official code for the paper _Adaptive Neural Speech Editing_
[[paper]](https://www.maxrmorrison.com/pdfs/morrison2023adaptive.pdf)
[[companion website]](https://www.maxrmorrison.com/sites/promonet/)


## Table of contents

- [Installation](#installation)
- [Usage](#usage)
    * [Example](#example)
    * [Adaptation](#adaptation)
    * [Generation](#generation)
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

### Example

To use `promonet` for speech editing, you must first perform speaker
adaptation on a dataset of recordings of the target speaker. You can then use
the resulting model checkpoint to perform speech editing in the target
speaker's voice. All of this can be done using either the API or CLI.


#### Application programming interface (API)

```
import promonet


###############################################################################
# Speaker adaptation
###############################################################################


# Speaker's name
name = 'max'

# Audio files for adaptation
files = [...]

# GPU indices to use for training
gpus = [0]

# Perform speaker adaptation
checkpoint = promonet.adapt.speaker(name, files, gpus=gpus)


###############################################################################
# Prosody editing
###############################################################################


# Load audio for prosody editing
audio = promonet.load.audio('test.wav')

# Get prosody features to edit
pitch, loudness = promonet.preprocess(audio, gpu=gpus[0])

# (Optional) If you have the text transcript, you can edit phoneme
#            durations using a forced alignment representation
pitch, loudness, alignment = promonet.preprocess(
    audio,
    text=promonet.load.text('test.txt'),
    gpu=gpus[0])

# We'll use a ratio of 2.0 for all prosody editing examples
ratio = 2.0

# Perform pitch-shifting
shifted = promonet.from_audio(
    audio,
    target_pitch=ratio * pitch,
    checkpoint=checkpoint,
    gpu=gpus[0])

# Perform time-stretching
stretched = promonet.from_audio(
    audio,
    grid=promonet.interpolate.grid.constant(pitch, ratio),
    checkpoint=checkpoint,
    gpu=gpus[0])

# Perform loudness-scaling
scaled = promonet.from_audio(
    audio,
    target_loudness=10 * math.log2(ratio) + loudness,
    checkpoint=checkpoint,
    gpu=gpus[0])
```


#### Command-line interface (CLI)

```
###############################################################################
# Speaker adaptation
###############################################################################


# Adapt the base model defined by the configuration and checkpoint to speaker
# <name> using a list of audio files and the indicated GPUs
python -m promonet.adapt \
    --config <config> \
    --name <name> \
    --files <files> \
    --checkpoint <checkpoint> \
    --gpus <gpus>


###############################################################################
# Speech editing
###############################################################################


# Perform reconstruction
python -m promonet \
    --config <config> \
    --audio_files <audio_files> \
    --output_files <output_files> \
    --checkpoint <checkpoint> \
    --gpu <gpu>

# Perform pitch-shifting
python -m promonet \
    --config <config> \
    --audio_files <audio_files> \
    --output_files <output_files> \
    --target_pitch_files <target_pitch_files> \
    --checkpoint <checkpoint> \
    --gpu <gpu>

# Perform time-stretching
python -m promonet \
    --config <config> \
    --audio_files <audio_files> \
    --output_files <output_files> \
    --grid_files <grid_files> \
    --checkpoint <checkpoint> \
    --gpu <gpu>

# Perform loudness-scaling
python -m promonet \
    --config <config> \
    --audio_files <audio_files> \
    --output_files <output_files> \
    --target_loudness_files <target_loudness_files> \
    --checkpoint <checkpoint> \
    --gpu <gpu>
```


### Adaptation

#### Application programming interface (API)

##### `promonet.adapt.speaker`

```
def speaker(
    name: str,
    files: List[Path],
    checkpoint: Path = promonet.DEFAULT_CHECKPOINT,
    gpus: Optional[int] = None) -> Path:
    """Perform speaker adaptation

    Args:
        name: The name of the speaker
        files: The audio files to use for adaptation
        checkpoint: The model checkpoint
        gpus: The gpus to run adaptation on

    Returns:
        checkpoint: The file containing the trained generator checkpoint
    """
```


#### Command-line interface (CLI)

```
python -m promonet.adapt \
    [-h] \
    [--config CONFIG] \
    --name NAME \
    --files FILES \
    [--checkpoint CHECKPOINT] \
    [--gpus GPUS [GPUS ...]]

Perform speaker adaptation

  -h, --help               show this help message and exit
  --config CONFIG          The configuration file
  --name NAME              The name of the speaker
  --files FILES            The audio files to use for adaptation
  --checkpoint CHECKPOINT  The model checkpoint
  --gpus GPUS [GPUS ...]   The gpus to run adaptation on
```


### Generation

#### Application programming interface (API)

##### `promonet.from_audio`

```
def from_audio(
    audio: torch.Tensor,
    sample_rate: int = promonet.SAMPLE_RATE,
    text: Optional[str] = None,
    grid: Optional[torch.Tensor] = None,
    target_loudness: Optional[torch.Tensor] = None,
    target_pitch: Optional[torch.Tensor] = None,
    checkpoint: Union[str, os.PathLike]=promonet.DEFAULT_CHECKPOINT,
    gpu: Optional[int] = None) -> torch.Tensor:
    """Perform speech editing

    Args:
        audio: The audio to edit
        sample_rate: The audio sample rate
        text: The speech transcript for editing phoneme durations
        grid: The interpolation grid for editing phoneme durations
        target_loudness: The loudness contour for editing loudness
        target_pitch: The pitch contour for shifting pitch
        checkpoint: The model checkpoint
        gpu: The GPU index

    Returns
        edited: The edited audio
    """
```


##### `promonet.from_file`

```
def from_file(
    audio_file: Union[str, os.PathLike],
    text_file: Optional[Union[str, os.PathLike]] = None,
    grid_file: Optional[Union[str, os.PathLike]] = None,
    target_loudness_file: Optional[Union[str, os.PathLike]] = None,
    target_pitch_file: Optional[Union[str, os.PathLike]] = None,
    checkpoint: Union[str, os.PathLike] = promonet.DEFAULT_CHECKPOINT,
    gpu: Optional[int] = None) -> torch.Tensor:
    """Edit speech on disk

    Args:
        audio_file: The audio to edit
        text_file: The speech transcript for editing phoneme durations
        grid_file: The interpolation grid for editing phoneme durations
        target_loudness_file: The loudness contour for editing loudness
        target_pitch_file: The pitch contour for shifting pitch
        checkpoint: The model checkpoint
        gpu: The GPU index

    Returns
        edited: The edited audio
    """
```


##### `promonet.from_file_to_file`

```
def from_file_to_file(
    audio_file: Union[str, os.PathLike],
    output_file: Union[str, os.PathLike],
    text_file: Optional[Union[str, os.PathLike]] = None,
    grid_file: Optional[Union[str, os.PathLike]] = None,
    target_loudness_file: Optional[Union[str, os.PathLike]] = None,
    target_pitch_file: Optional[Union[str, os.PathLike]] = None,
    checkpoint: Union[str, os.PathLike] = promonet.DEFAULT_CHECKPOINT,
    gpu: Optional[int] = None):
    """Edit speech on disk and save to disk

    Args:
        audio_file: The audio to edit
        output_file: The file to save the edited audio
        text_file: The speech transcript for editing phoneme durations
        grid_file: The interpolation grid for editing phoneme durations
        target_loudness_file: The loudness contour for editing loudness
        target_pitch_file: The pitch contour for shifting pitch
        checkpoint: The model checkpoint
        gpu: The GPU index
    """
```


##### `promonet.from_files_to_files`

```
def from_files_to_files(
    audio_files: List[Union[str, os.PathLike]],
    output_files: List[Union[str, os.PathLike]],
    text_files: Optional[List[Union[str, os.PathLike]]] = None,
    grid_files: Optional[List[Union[str, os.PathLike]]] = None,
    target_loudness_files: Optional[List[Union[str, os.PathLike]]] = None,
    target_pitch_files: Optional[List[Union[str, os.PathLike]]] = None,
    checkpoint: Union[str, os.PathLike] = promonet.DEFAULT_CHECKPOINT,
    gpu: Optional[int] = None):
    """Edit speech on disk and save to disk

    Args:
        audio_files: The audio to edit
        output_files: The files to save the edited audio
        text_files: The speech transcripts for editing phoneme durations
        grid_files: The interpolation grids for editing phoneme durations
        target_loudness_files: The loudness contours for editing loudness
        target_pitch_files: The pitch contours for shifting pitch
        checkpoint: The model checkpoint
        gpu: The GPU index
    """
```


#### Command-line interface (CLI)

```
python -m promonet \
    [-h] \
    [--config CONFIG] \
    --audio_files AUDIO_FILES [AUDIO_FILES ...] \
    --output_files OUTPUT_FILES [OUTPUT_FILES ...] \
    [--grid_files GRID_FILES [GRID_FILES ...]] \
    [--target_loudness_files TARGET_LOUDNESS_FILES [TARGET_LOUDNESS_FILES ...]] \
    [--target_pitch_files TARGET_PITCH_FILES [TARGET_PITCH_FILES ...]] \
    [--checkpoint CHECKPOINT] \
    [--gpu GPU]

Perform speech editing

    -h, --help
        show this help message and exit
    --config CONFIG
        The configuration file
    --audio_files AUDIO_FILES [AUDIO_FILES ...]
        The audio to edit
    --output_files OUTPUT_FILES [OUTPUT_FILES ...]
        The files to save the edited audio
    --grid_files GRID_FILES [GRID_FILES ...]
        The interpolation grids for editing phoneme durations
    --target_loudness_files TARGET_LOUDNESS_FILES [TARGET_LOUDNESS_FILES ...]
        The loudness contours for editing loudness
    --target_pitch_files TARGET_PITCH_FILES [TARGET_PITCH_FILES ...]
        The pitch contours for shifting pitch
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
python -m promonet.data.preprocess --datasets <datasets> --gpu <gpu>
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
    --gpus <gpus>
```

If the config file has been previously run, the most recent checkpoint will
automatically be loaded and training will resume from that checkpoint.


### Monitor

You can monitor training via `tensorboard` as follows.

```
tensorboard --logdir runs/ --port <port>
```


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
M. Morrison and B. Pardo, "Adaptive Neural Speech Prosody Editing," Submitted
to ICML 2023, July 2023.


### BibTex

```
@inproceedings{morrison2023adaptive,
    title={Adaptive Neural Speech Prosody Editing},
    author={Morrison, Max and Pardo, Bryan},
    booktitle={Submitted to ICML 2023},
    month={July},
    year={2023}
}
```
