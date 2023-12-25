# Runs all experiments in the paper
# "Adaptive Neural Speech Prosody Editing"

# Args
# $1 - index of GPU to use

# Download datasets
python -m promonet.data.download

# Setup experiments
python -m promonet.data.augment
python -m promonet.data.preprocess --gpu $1
python -m promonet.partition

# First pass experiments trainings and evaluations
python -m promonet.train --config config/augment-multiband-varpitch-256.py --gpu $1
python -m promonet.evaluate --config config/augment-multiband-varpitch-256.py --gpu $1

# DSP-based baseline evaluations
# python -m promonet.evaluate --config config/baselines/psola.py --gpu $1
# python -m promonet.evaluate --config config/baselines/world.py --gpu $1
