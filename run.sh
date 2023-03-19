# Runs all experiments in the paper
# "Adaptive Neural Speech Prosody Editing"

# Args
# $1 - list of indices of GPUs to use

# Download datasets
python -m promonet.data.download

# Setup experiments
python -m promonet.data.augment
python -m promonet.data.preprocess --features spectrogram ppg prosody --gpu $1
python -m promonet.partition

# First pass experiments trainings and evaluations
# python -m promonet.train --config config/base-small.py --gpus $1
# python -m promonet.train --config config/conddisc-small.py --gpus $1
# python -m promonet.train --config config/conddisc-condgen-small.py --gpus $1
# python -m promonet.train --config config/conddisc-condgen-augment-small.py --gpus $1
python -m promonet.train --config config/conddisc-condgen-augment-snake-small.py --gpus $1

# Baseline trainings and evaluations
# python -m promonet.train --config config/promovoco-small.py --gpus $1
# python -m promonet.train --config config/spectral-small.py --gpus $1
# python -m promonet.train --config config/spectral-voco-small.py --gpus $1
# python -m promonet.train --config config/two-stage-small.py --gpus $1
# python -m promonet.train --config config/vits.py --gpus $1

# DSP-based baseline evaluations
python -m promonet.evaluate --config config/psola.py
python -m promonet.evaluate --config config/world.py
