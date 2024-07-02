# Runs experiments from the paper
# "Fine-Grained and Interpretable Neural Speech Editing"

# Args
# $1 - index of GPU to use


###############################################################################
# Best model
###############################################################################


# Data pipeline
python -m promonet.data.download --datasets vctk
python -m promonet.data.augment --datasets vctk
python -m promonet.data.preprocess --datasets vctk --gpu $1
python -m promonet.partition --datasets vctk

# Train
python -m promonet.train --gpu $1

# Evaluate
python -m promonet.evaluate --datasets vctk --gpu $1


###############################################################################
# Ablations
###############################################################################


# Data pipeline
python -m promonet.data.preprocess \
    --config config/ablations/ablate-viterbi.py \
    --features pitch \
    --datasets vctk \
    --gpu $1

# Train
python -m promonet.train --config config/ablations/ablate-all.py --gpu $1
python -m promonet.train --config config/ablations/ablate-augment.py --gpu $1
python -m promonet.train --config config/ablations/ablate-multiloud.py --gpu $1
python -m promonet.train --config config/ablations/ablate-sppg.py --gpu $1
python -m promonet.train --config config/ablations/ablate-variable-pitch.py --gpu $1
python -m promonet.train --config config/ablations/ablate-viterbi.py --gpu $1

# Evaluate
python -m promonet.evaluate \
    --config config/ablations/ablate-all.py \
    --datasets vctk \
    --gpu $1
python -m promonet.evaluate \
    --config config/ablations/ablate-augment.py \
    --datasets vctk \
    --gpu $1
python -m promonet.evaluate \
    --config config/ablations/ablate-multiloud.py \
    --datasets vctk \
    --gpu $1
python -m promonet.evaluate \
    --config config/ablations/ablate-sppg.py \
    --datasets vctk \
    --gpu $1
python -m promonet.evaluate \
    --config config/ablations/ablate-variable-pitch.py \
    --datasets vctk \
    --gpu $1
python -m promonet.evaluate \
    --config config/ablations/ablate-viterbi.py \
    --datasets vctk \
    --gpu $1


###############################################################################
# Baselines
###############################################################################


# Train
python -m promonet.train --config config/baselines/mels.py --gpu $1

# Evaluate
python -m promonet.evaluate \
    --config config/baselines/mels.py \
    --datasets vctk \
    --gpu $1
python -m promonet.evaluate \
    --config config/baselines/world.py \
    --datasets vctk \
    --gpu $1
