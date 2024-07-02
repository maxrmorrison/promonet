<h1 align="center">Prosody and Pronunciation Modification Network (ProMoNet)</h1>
<div align="center">

[![PyPI](https://img.shields.io/pypi/v/promonet.svg)](https://pypi.python.org/pypi/promonet)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://pepy.tech/badge/promonet)](https://pepy.tech/project/promonet)

Official code for the paper _Fine-Grained and Interpretable Neural Speech Editing_

[[paper]](https://www.maxrmorrison.com/pdfs/morrison2024fine.pdf)
[[website]](https://www.maxrmorrison.com/sites/promonet/)

</div>


## Table of contents

- [Installation](#installation)
- [Usage](#usage)
- [Application programming interface (API)](#application-programming-interface-api)
    * [Adaptation API](#adaptation-api)
        * [`promonet.adapt.speaker`](#promonetadaptspeaker)
    * [Preprocessing API](#preprocessing-api)
        * [`promonet.preprocess.from_audio`](#promonetpreprocessfrom_audio)
        * [`promonet.preprocess.from_file`](#promonetpreprocessfrom_file)
        * [`promonet.preprocess.from_file_to_file`](#promonetpreprocessfrom_file_to_file)
        * [`promonet.preprocess.from_files_to_files`](#promonetpreprocessfrom_files_to_files)
    * [Editing API](#editing-api)
        * [`promonet.edit.from_features`](#promoneteditfrom_features)
        * [`promonet.edit.from_file`](#promoneteditfrom_file)
        * [`promonet.edit.from_file_to_file`](#promoneteditfrom_file_to_file)
        * [`promonet.edit.from_files_to_files`](#promoneteditfrom_files_to_files)
    * [Synthesis API](#synthesis-api)
        * [`promonet.synthesize.from_features`](#promonetsynthesizefrom_features)
        * [`promonet.synthesize.from_file`](#promonetsynthesizefrom_file)
        * [`promonet.synthesize.from_file_to_file`](#promonetsynthesizefrom_file_to_file)
        * [`promonet.synthesize.from_files_to_files`](#promonetsynthesizefrom_files_to_files)
- [Command-line interface (CLI)](#command-line-interface-cli)
    * [Adaptation CLI](#adaptation-cli)
        * [`promonet.adapt`](#promonetadapt)
    * [Preprocessing CLI](#preprocessing-cli)
        * [`promonet.preprocess`](#promonetpreprocess)
    * [Editing CLI](#editing-cli)
        * [`promonet.edit`](#promonetedit)
    * [Synthesis CLI](#synthesis-cli)
        * [`promonet.synthesize`](#promonetsynthesize)
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

We are working on adding [`torbi`, our fast Viterbi decoding implementation](https://github.com/maxrmorrison/torbi) to PyTorch. Until then, you must manually download and install `torbi`. You can track the progress of incorporation into PyTorch [here](https://github.com/pytorch/pytorch/issues/121160).


## Usage

Our included model checkpoint allows speech editing and synthesis for VCTK speakers.
To use `promonet` with other speakers, you must first perform speaker
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

# GPU index to perform adaptation and editing on
gpu = 0

# Perform speaker adaptation
checkpoint = promonet.adapt.speaker(name, files, gpu=gpu)


###############################################################################
# Speech editing
###############################################################################


# Load speech to edit
audio = promonet.load.audio('test.wav')

# Get features to edit
loudness, pitch, periodicity, ppg = promonet.preprocess.from_audio(
    audio,
    promonet.SAMPLE_RATE,
    gpu)

# We'll use a ratio of 2.0 for all editing examples
ratio = 2.0

# Perform pitch-shifting
shifted = promonet.synthesize.from_features(
    *promonet.edit.from_features(
        loudness,
        pitch,
        periodicity,
        ppg,
        pitch_shift_cents=promonet.convert.ratio_to_cents(ratio)),
    checkpoint=checkpoint,
    gpu=gpu)

# Perform time-stretching
stretched = promonet.synthesize.from_features(
    *promonet.edit.from_features(
        loudness,
        pitch,
        periodicity,
        ppg,
        time_stretch_ratio=ratio),
    checkpoint=checkpoint,
    gpu=gpu)

# Perform loudness editing
scaled = promonet.synthesize.from_features(
    *promonet.edit.from_features(
        loudness,
        pitch,
        periodicity,
        ppg,
        loudness_scale_db=promonet.convert.ratio_to_db(ratio)),
    checkpoint=checkpoint,
    gpu=gpu)

# Edit spectral balance (> 1 for Alvin and the Chipmunks; < 1 for Patrick Star)
alvin = promonet.synthesize.from_features(
    loudness,
    pitch,
    periodicity,
    ppg,
    spectral_balance_ratio=ratio,
    checkpoint=checkpoint,
    gpu=gpu)
```

See the [`ppgs.edit`](https://github.com/interactiveaudiolab/ppgs#ppgsedit) submodule documentation for the pronunciation (PPG) editing API.


## Application programming interface (API)

### Adaptation API

#### `promonet.adapt.speaker`

```python
def speaker(
    name: str,
    files: List[Path],
    checkpoint: Path = None,
    gpu: Optional[int] = None
) -> Path:
    """Perform speaker adaptation

    Args:
        name: The name of the speaker
        files: The audio files to use for adaptation
        checkpoint: The model checkpoint directory
        gpu: The gpu to run adaptation on

    Returns:
        checkpoint: The file containing the trained generator checkpoint
    """
```


### Preprocessing API

#### `promonet.preprocess.from_audio`

```python
def from_audio(
    audio: torch.Tensor,
    sample_rate: int = promonet.SAMPLE_RATE,
    gpu: Optional[int] = None,
    features: list = ['loudness', 'pitch', 'periodicity', 'ppg']
) -> Union[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]
]:
    """Preprocess audio

    Arguments
        audio: Audio to preprocess
        sample_rate: Audio sample rate
        gpu: The GPU index
        features: The features to preprocess.
            Options: ['loudness', 'pitch', 'periodicity', 'ppg', 'text'].

    Returns
        loudness: The loudness contour
        periodicity: The periodicity contour
        pitch: The pitch contour
        ppg: The phonetic posteriorgram
        text: The text transcript
    """
```


#### `promonet.preprocess.from_file`

```python
def from_file(
    file: Union[str, bytes, os.PathLike],
    gpu: Optional[int] = None,
    features: list = ['loudness', 'pitch', 'periodicity', 'ppg']
) -> Union[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]
]:
    """Preprocess audio on disk

    Arguments
        file: Audio file to preprocess
        gpu: The GPU index
        features: The features to preprocess.
            Options: ['loudness', 'pitch', 'periodicity', 'ppg', 'text'].

    Returns
        loudness: The loudness contour
        pitch: The pitch contour
        periodicity: The periodicity contour
        ppg: The phonetic posteriorgram
        text: The text transcript
    """
```


#### `promonet.preprocess.from_file_to_file`

```python
def from_file_to_file(
    file: Union[str, bytes, os.PathLike],
    output_prefix: Optional[Union[str, os.PathLike]] = None,
    gpu: Optional[int] = None,
    features: list = ['loudness', 'pitch', 'periodicity', 'ppg']
) -> None:
    """Preprocess audio on disk and save

    Arguments
        file: Audio file to preprocess
        output_prefix: File to save features, minus extension
        gpu: The GPU index
        features: The features to preprocess.
            Options: ['loudness', 'pitch', 'periodicity', 'ppg', 'text'].
    """
```


#### `promonet.preprocess.from_files_to_files`

```python
def from_files_to_files(
    files: List[Union[str, bytes, os.PathLike]],
    output_prefixes: Optional[List[Union[str, os.PathLike]]] = None,
    gpu: Optional[int] = None,
    features: list = ['loudness', 'pitch', 'periodicity', 'ppg']
) -> None:
    """Preprocess multiple audio files on disk and save

    Arguments
        files: Audio files to preprocess
        output_prefixes: Files to save features, minus extension
        gpu: The GPU index
        features: The features to preprocess.
            Options: ['loudness', 'pitch', 'periodicity', 'ppg', 'text'].
    """
```


### Editing API

##### `promonet.edit.from_features`

```python
def from_features(
    loudness: torch.Tensor,
    pitch: torch.Tensor,
    periodicity: torch.Tensor,
    ppg: torch.Tensor,
    pitch_shift_cents: Optional[float] = None,
    time_stretch_ratio: Optional[float] = None,
    loudness_scale_db: Optional[float] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Edit speech representation

    Arguments
        loudness: Loudness contour to edit
        pitch: Pitch contour to edit
        periodicity: Periodicity contour to edit
        ppg: PPG to edit
        pitch_shift_cents: Amount of pitch-shifting in cents
        time_stretch_ratio: Amount of time-stretching. Faster when above one.
        loudness_scale_db: Loudness ratio editing in dB (not recommended; use loudness)

    Returns
        edited_loudness, edited_pitch, edited_periodicity, edited_ppg
    """
```


##### `promonet.edit.from_file`

```python
def from_file(
    loudness_file: Union[str, bytes, os.PathLike],
    pitch_file: Union[str, bytes, os.PathLike],
    periodicity_file: Union[str, bytes, os.PathLike],
    ppg_file: Union[str, bytes, os.PathLike],
    pitch_shift_cents: Optional[float] = None,
    time_stretch_ratio: Optional[float] = None,
    loudness_scale_db: Optional[float] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Edit speech representation on disk

    Arguments
        loudness_file: Loudness file to edit
        pitch_file: Pitch file to edit
        periodicity_file: Periodicity file to edit
        ppg_file: PPG file to edit
        pitch_shift_cents: Amount of pitch-shifting in cents
        time_stretch_ratio: Amount of time-stretching. Faster when above one.
        loudness_scale_db: Loudness ratio editing in dB (not recommended; use loudness)

    Returns
        edited_loudness, edited_pitch, edited_periodicity, edited_ppg
    """
```


##### `promonet.edit.from_file_to_file`

```python
def from_file_to_file(
    loudness_file: Union[str, bytes, os.PathLike],
    pitch_file: Union[str, bytes, os.PathLike],
    periodicity_file: Union[str, bytes, os.PathLike],
    ppg_file: Union[str, bytes, os.PathLike],
    output_prefix: Union[str, bytes, os.PathLike],
    pitch_shift_cents: Optional[float] = None,
    time_stretch_ratio: Optional[float] = None,
    loudness_scale_db: Optional[float] = None
) -> None:
    """Edit speech representation on disk and save to disk

    Arguments
        loudness_file: Loudness file to edit
        pitch_file: Pitch file to edit
        periodicity_file: Periodicity file to edit
        ppg_file: PPG file to edit
        output_prefix: File to save output, minus extension
        pitch_shift_cents: Amount of pitch-shifting in cents
        time_stretch_ratio: Amount of time-stretching. Faster when above one.
        loudness_scale_db: Loudness ratio editing in dB (not recommended; use loudness)
    """
```


##### `promonet.edit.from_files_to_files`

```python
def from_files_to_files(
    loudness_files: List[Union[str, bytes, os.PathLike]],
    pitch_files: List[Union[str, bytes, os.PathLike]],
    periodicity_files: List[Union[str, bytes, os.PathLike]],
    ppg_files: List[Union[str, bytes, os.PathLike]],
    output_prefixes: List[Union[str, bytes, os.PathLike]],
    pitch_shift_cents: Optional[float] = None,
    time_stretch_ratio: Optional[float] = None,
    loudness_scale_db: Optional[float] = None
) -> None:
    """Edit speech representations on disk and save to disk

    Arguments
        loudness_files: Loudness files to edit
        pitch_files: Pitch files to edit
        periodicity_files: Periodicity files to edit
        ppg_files: Phonetic posteriorgram files to edit
        output_prefixes: Files to save output, minus extension
        pitch_shift_cents: Amount of pitch-shifting in cents
        time_stretch_ratio: Amount of time-stretching. Faster when above one.
        loudness_scale_db: Loudness ratio editing in dB (not recommended; use loudness)
    """
```


### Synthesis API

##### `promonet.synthesize.from_features`

```python
def from_features(
    loudness: torch.Tensor,
    pitch: torch.Tensor,
    periodicity: torch.Tensor,
    ppg: torch.Tensor,
    speaker: Union[int, torch.Tensor] = 0,
    spectral_balance_ratio: float = 1.,
    checkpoint: Optional[Union[str, os.PathLike]] = None,
    gpu: Optional[int] = None) -> torch.Tensor:
    """Perform speech synthesis

    Args:
        loudness: The loudness contour
        pitch: The pitch contour
        periodicity: The periodicity contour
        ppg: The phonetic posteriorgram
        speaker: The speaker index
        spectral_balance_ratio: > 1 for Alvin and the Chipmunks; < 1 for Patrick Star
        checkpoint: The generator checkpoint
        gpu: The GPU index

    Returns
        generated: The generated speech
    """
```


##### `promonet.synthesize.from_file`

```python
def from_file(
    loudness_file: Union[str, os.PathLike],
    pitch_file: Union[str, os.PathLike],
    periodicity_file: Union[str, os.PathLike],
    ppg_file: Union[str, os.PathLike],
    speaker: Union[int, torch.Tensor] = 0,
    checkpoint: Optional[Union[str, os.PathLike]] = None,
    gpu: Optional[int] = None
) -> torch.Tensor:
    """Perform speech synthesis from features on disk

    Args:
        loudness_file: The loudness file
        pitch_file: The pitch file
        periodicity_file: The periodicity file
        ppg_file: The phonetic posteriorgram file
        speaker: The speaker index
        checkpoint: The generator checkpoint
        gpu: The GPU index

    Returns
        generated: The generated speech
    """
```


##### `promonet.synthesize.from_file_to_file`

```python
def from_file_to_file(
    loudness_file: Union[str, os.PathLike],
    pitch_file: Union[str, os.PathLike],
    periodicity_file: Union[str, os.PathLike],
    ppg_file: Union[str, os.PathLike],
    output_file: Union[str, os.PathLike],
    speaker: Union[int, torch.Tensor] = 0,
    checkpoint: Optional[Union[str, os.PathLike]] = None,
    gpu: Optional[int] = None
) -> None:
    """Perform speech synthesis from features on disk and save

    Args:
        loudness_file: The loudness file
        pitch_file: The pitch file
        periodicity_file: The periodicity file
        ppg_file: The phonetic posteriorgram file
        output_file: The file to save generated speech audio
        speaker: The speaker index
        checkpoint: The generator checkpoint
        gpu: The GPU index
    """
```


##### `promonet.synthesize.from_files_to_files`

```python
def from_files_to_files(
    loudness_files: List[Union[str, os.PathLike]],
    pitch_files: List[Union[str, os.PathLike]],
    periodicity_files: List[Union[str, os.PathLike]],
    ppg_files: List[Union[str, os.PathLike]],
    output_files: List[Union[str, os.PathLike]],
    speakers: Optional[Union[List[int], torch.Tensor]] = None,
    checkpoint: Optional[Union[str, os.PathLike]] = None,
    gpu: Optional[int] = None
) -> None:
    """Perform batched speech synthesis from features on disk and save

    Args:
        loudness_files: The loudness files
        pitch_files: The pitch files
        periodicity_files: The periodicity files
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
    --files FILES [FILES ...] \
    [--checkpoint CHECKPOINT] \
    [--gpu GPU]

Perform speaker adaptation

optional arguments:
  -h, --help
    show this help message and exit
  --name NAME
    The name of the speaker
  --files FILES [FILES ...]
    The audio files to use for adaptation
  --checkpoint CHECKPOINT
    The model checkpoint directory
  --gpu GPU
    The gpu to run adaptation on
```


### Preprocessing CLI

#### `promonet.preprocess`

```
python -m promonet.preprocess \
    [-h] \
    --files FILES [FILES ...] \
    [--output_prefixes OUTPUT_PREFIXES [OUTPUT_PREFIXES ...]] \
    [--features {loudness,pitch,periodicity,ppg} [{loudness,pitch,periodicity,ppg} ...]] \
    [--gpu GPU]

Preprocess

arguments:
  --files FILES [FILES ...]
    Audio files to preprocess

optional arguments:
  -h, --help
    show this help message and exit
  --output_prefixes OUTPUT_PREFIXES [OUTPUT_PREFIXES ...]
    Files to save features, minus extension
  --features {loudness,pitch,periodicity,ppg} [{loudness,pitch,periodicity,ppg} ...]
    The features to preprocess
  --gpu GPU
    The index of the gpu to use
```


### Editing CLI

#### `promonet.edit`

```
python -m promonet.edit \
    [-h] \
    --loudness_files LOUDNESS_FILES [LOUDNESS_FILES ...] \
    --pitch_files PITCH_FILES [PITCH_FILES ...] \
    --periodicity_files PERIODICITY_FILES [PERIODICITY_FILES ...] \
    --ppg_files PPG_FILES [PPG_FILES ...] \
    --output_prefixes OUTPUT_PREFIXES [OUTPUT_PREFIXES ...] \
    [--pitch_shift_cents PITCH_SHIFT_CENTS] \
    [--time_stretch_ratio TIME_STRETCH_RATIO] \
    [--loudness_scale_db LOUDNESS_SCALE_DB]

Edit speech representation

arguments:
  --loudness_files LOUDNESS_FILES [LOUDNESS_FILES ...]
    The loudness files to edit
  --pitch_files PITCH_FILES [PITCH_FILES ...]
    The pitch files to edit
  --periodicity_files PERIODICITY_FILES [PERIODICITY_FILES ...]
    The periodicity files to edit
  --ppg_files PPG_FILES [PPG_FILES ...]
    The ppg files to edit
  --output_prefixes OUTPUT_PREFIXES [OUTPUT_PREFIXES ...]
    The locations to save output files, minus extension

optional arguments:
  -h, --help
    show this help message and exit
  --pitch_shift_cents PITCH_SHIFT_CENTS
    Amount of pitch-shifting in cents
  --time_stretch_ratio TIME_STRETCH_RATIO
    Amount of time-stretching. Faster when above one.
  --loudness_scale_db LOUDNESS_SCALE_DB
    Loudness ratio editing in dB (not recommended; use loudness)
```


### Synthesis CLI

#### `promonet.synthesize`

```
python -m promonet.synthesize \
    --loudness_files LOUDNESS_FILES [LOUDNESS_FILES ...] \
    --pitch_files PITCH_FILES [PITCH_FILES ...] \
    --periodicity_files PERIODICITY_FILES [PERIODICITY_FILES ...] \
    --ppg_files PPG_FILES [PPG_FILES ...] \
    --output_files OUTPUT_FILES [OUTPUT_FILES ...] \
    [--speakers SPEAKERS [SPEAKERS ...]] \
    [--checkpoint CHECKPOINT] \
    [--gpu GPU]

Synthesize speech from features

arguments:
  --loudness_files LOUDNESS_FILES [LOUDNESS_FILES ...]
    The loudness files
  --pitch_files PITCH_FILES [PITCH_FILES ...]
    The pitch files
  --periodicity_files PERIODICITY_FILES [PERIODICITY_FILES ...]
    The periodicity files
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
tensorboard --logdir runs/ --port <port> --load_fast true
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
    --gpu <gpu>
```


## Citation

### IEEE
M. Morrison, C. Churchwell, N. Pruyne, and B. Pardo, "Fine-Grained and Interpretable Neural Speech Editing," Interspeech, September 2024.


### BibTex

```
@inproceedings{morrison2024adaptive,
    title={Fine-Grained and Interpretable Neural Speech Editing},
    author={Morrison, Max and Churchwell, Cameron and Pruyne, Nathan and Pardo, Bryan},
    booktitle={Interspeech},
    month={September},
    year={2024}
}
```
