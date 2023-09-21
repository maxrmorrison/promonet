import os
from json import load
import promonet
from shutil import copy as shcp
import torch
import re
from pathlib import Path

all_representations = [
    'w2v2fb',
    'w2v2fc',
    'bottleneck',
    'mel',
    'encodec'
]

SHIFT_CENTS = [-200, 200]
# SHIFT_CENTS = []

all_models = []
all_models += [f'{rep}-ppg' for rep in all_representations]
all_models += [f'{rep}-latents' for rep in all_representations]

with open(promonet.PARTITION_DIR / 'vctk.json', 'r') as f:
    partitions = load(f)

stems = partitions['test']

cache_dir = promonet.CACHE_DIR / 'vctk'
output_dir = promonet.CACHE_DIR / 'ppgs-subjective'

stems = [f'{stem}-100' for stem in stems]

audio_files = [cache_dir / f'{stem}.wav' for stem in stems]
for audio_file in audio_files:
    assert audio_file.exists(), f'{audio_file} does not exist'

#copy originals
originals_dir = output_dir / 'original'
originals_dir.mkdir(exist_ok=True, parents=True)
for idx, audio_file in enumerate(audio_files):
    shcp(audio_file, originals_dir / (audio_file.parent.name + '-' + audio_file.name))

command = f'python -m ppgs --sources {originals_dir} --gpu 0'

os.system(command)

audio_files = [str(f) for f in audio_files]
print('original', flush=True)

pitch_files = [cache_dir / f'{stem}-pitch.pt' for stem in stems]
for pitch_file in pitch_files:
    assert pitch_file.exists(), f'{pitch_file} does not exist'
print('finished checking pitch files', flush=True)
# pitch_files = [str(f) for f in pitch_files]

#create shifted pitch files

ratios = [2 ** (cents / 1200) for cents in SHIFT_CENTS]
print(ratios, flush=True)

shifted_pitch_files = []
for ratio in ratios:
    print(ratio, flush=True)
    for pfi, pitch_file in enumerate(pitch_files):
        print(ratio, float(pfi) / len(pitch_files), flush=True)
        filename = (re.sub(r'-100-', f'-{int(ratio * 100):03d}-', str(pitch_file)))
        shifted_pitch_files.append(filename)
        # if not Path(filename).exists():
        print(pitch_file, ratio)
        contour = torch.load(pitch_file)
        # import pdb; pdb.set_trace()
        shifted = contour * ratio
        torch.save(shifted, filename)
# pitch_files += shifted_pitch_files

print('finished creating pitch shift files', flush=True)

speaker_ids = [stem.split('/')[0] for stem in stems]

audio_files *= len(ratios) + 1
speaker_ids *= len(ratios) + 1

print('starting models', flush=True)

pitch_files = [str(pitch_file) for pitch_file in pitch_files]

for model in all_models:
    print('starting', model, flush=True)
    ppg_files = [cache_dir / f'{stem}-{model}.pt' for stem in stems]
    for ppg_file in ppg_files:
        assert ppg_file.exists(), f'{ppg_file} does not exist'
    ppg_files = [str(f) for f in ppg_files]

    ppg_files *= (len(ratios) + 1)


    output_files = [output_dir / model / f'{stem.replace("/", "-")}.wav' for stem in stems]
    for f in output_files:
        f.parent.mkdir(exist_ok=True, parents=True)
    output_files = [str(f) for f in output_files]

    shifted_outputs = []
    for ratio in ratios:
        for output_file in output_files:
            shifted_outputs.append(re.sub(r'-100\.wav', f'-{int(ratio * 100):03d}.wav', str(output_file)))
        
    shifted_pitch_files = []
    for ratio in ratios:
        for pitch_file in pitch_files:
            shifted_pitch_files.append(re.sub(r'-100-pitch.pt', f'-{int(ratio * 100):03d}-pitch.pt', str(pitch_file)))

    target_pitch_files = pitch_files[:] + shifted_pitch_files
    
    output_files += shifted_outputs

    # raise ValueError(list(zip(pitch_files, output_files)))

    # import pdb; pdb.set_trace()


    command = "python -m promonet " 
    command += f"--audio_files {' '.join(audio_files)} "
    command += f"--output_files {' '.join(output_files)} "
    command += f"--target_ppg_files {' '.join(ppg_files)} "
    command += f"--speaker_ids {' '.join(speaker_ids)} "
    command += f"--target_pitch_files {' '.join(target_pitch_files)} "
    command += f'--checkpoint /repos/promonet/runs/{model}/generator-00250000.pt '
    command += f'--config "/repos/promonet/config/ppgs-experiments/ppgs/{model}.py" "/repos/promonet/config/ppgs-experiments/promonet/{model}.py" '
    command += f'--gpu 0'

    print(command)

    os.system(command)

    print('creating ppg files')

    command = f'python -m ppgs --sources {output_dir / model} --gpu 0'
    os.system(command)
    
    print(model, flush=True)



# for model in all_models:
#     print('starting', model, flush=True)
#     for ratio in ratios:
#         ppg_files = [cache_dir / f'{stem}-{model}.pt' for stem in stems]

#         output_files = [output_dir / model / f'{stem.replace('/', '-')}']
        

#         command = "python -m promonet " 
#         command += f"--audio_files {' '.join(audio_files)} "
#         command += f"--output_files {' '.join(output_files)} "
#         command += f"--target_ppg_files {' '.join(ppg_files)} "
#         command += f"--speaker_ids {' '.join(speaker_ids)} "
#         command += f"--target_pitch_files {' '.join(pitch_files)} "
#         command += f'--checkpoint /repos/promonet/runs/{model}/generator-00400000.pt '
#         command += f'--config "/repos/promonet/config/ppgs-experiments/ppgs/{model}.py" "/repos/promonet/config/ppgs-experiments/promonet/{model}.py" '
#         command += f'--gpu 0'

#         print(os.system(command))
    
#     print(model, flush=True)

print('done!', flush=True)