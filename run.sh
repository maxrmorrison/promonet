# Runs all experiments in the paper
# "Fine-Grained and Interpretable Neural Speech Editing"

# Args
# $1 - index of GPU to use

# Download datasets
# python -m promonet.data.download --datasets daps vctk

# Setup experiments
# python -m promonet.data.augment --datasets daps vctk
# python -m promonet.data.preprocess --gpu $1 --datasets daps vctk
# python -m promonet.partition --datasets daps vctk

# First pass experiments trainings and evaluations
python -m promonet.train --gpu $1
python -m promonet.evaluate --gpu $1 --datasets vctk

# DSP-based baseline evaluations
python -m promonet.evaluate --config config/baselines/world.py --gpu $1 --datasets vctk
