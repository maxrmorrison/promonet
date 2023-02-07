# Runs all experiments in the paper
# "Neural Speech Prosody Editing"

# Args
# $1 - list of indices of GPUs to use

# Download datasets
python -m promonet.data.download

# Setup experiments
python -m promonet.data.preprocess --gpu $1
python -m promonet.data.augment --gpu $1
python -m promonet.partition

# Train
python -m promonet.train --config config/test-ablate-mrd-small.py --gpus $1

# Evaluate
python -m promonet.evaluate --config config/psola.py
python -m promonet.evaluate --config config/world.py
python -m promonet.evaluate \
    --config config/test-ablate-mrd-small.py \
    --checkpoint runs/test-ablate-mrd-small/00250000.pt \
    --gpu $1
