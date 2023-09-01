import os
from json import load
import promonet

all_representations = [
    'w2v2fb',
    'w2v2fc',
    'bottleneck',
    'mel',
    'encodec'
]

all_models = [f'{rep}-ppg' for rep in all_representations] + [f'{rep}-latents' for rep in all_representations]

with open(promonet.PARTITION_DIR / 'vctk.json', 'r') as f:
    partitions = load(f)

stems = partitions['test']

cache_dir = promonet.CACHE_DIR / 'vctk'
output_dir = promonet.CACHE_DIR / 'ppgs-subjective'

ratio = 100

stems = [f'{stem}-100' for stem in stems]

audio_files = [cache_dir / f'{stem}.wav' for stem in stems]
for audio_file in audio_files:
    assert audio_file.exists(), f'{audio_file} does not exist'
audio_files = [str(f) for f in audio_files]


pitch_files = [cache_dir / f'{stem}-pitch.pt' for stem in stems]
for pitch_file in pitch_files:
    assert pitch_file.exists(), f'{pitch_file} does not exist'
pitch_files = [str(f) for f in pitch_files]

speaker_ids = [stem.split('/')[0] for stem in stems]

for model in all_models:
    ppg_files = [cache_dir / f'{stem}-{model}.pt' for stem in stems]
    for ppg_file in ppg_files:
        assert ppg_file.exists(), f'{ppg_file} does not exist'
    ppg_files = [str(f) for f in ppg_files]


    output_files = [output_dir / model / f'{stem.replace("/", "-")}.wav' for stem in stems]
    for f in output_files:
        f.parent.mkdir(exist_ok=True, parents=True)
    output_files = [str(f) for f in output_files]


    command = "python -m promonet " 
    command += f"--audio_files {' '.join(audio_files)} "
    command += f"--output_files {' '.join(output_files)} "
    command += f"--target_ppg_files {' '.join(ppg_files)} "
    command += f"--speaker_ids {' '.join(speaker_ids)} "
    command += f'--checkpoint /repos/promonet/runs/{model}/generator-00100000.pt '
    command += f'--config "/repos/promonet/config/ppgs-experiments/ppgs/{model}.py" "/repos/promonet/config/ppgs-experiments/promonet/{model}.py" '
    command += f'--gpu 0'

    print(model)

    os.system(command)