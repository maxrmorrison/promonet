# Runs all experiments in the paper
# "Adaptive Neural Speech Prosody Editing"

# Args
# $1 - index of GPU to use

# Download datasets
# python -m promonet.data.download --datasets daps vctk

# Setup experiments
# python -m promonet.data.augment --datasets daps vctk
# python -m promonet.preprocess --gpu $1 --datasets daps vctk
# python -m promonet.partition --datasets daps vctk

# First pass experiments trainings and evaluations
python -m promonet.train --config config/base.py --gpu $1
# python -m promonet.train --config config/conddisc.py --gpu $1
# python -m promonet.train --config config/conddisc-condgen.py --gpu $1
# python -m promonet.train --config config/conddisc-condgen-augment.py --gpu $1
# python -m promonet.train --config config/conddisc-condgen-augment-snake.py --gpu $1

# Ablate architecture
# python -m promonet.train --config config/ablate-architecture/vocoder.py --gpu $1
# python -m promonet.train --config config/ablate-architecture/two-stage.py --gpu $1

# Neural baselines training and evaluation
# python -m promonet.train --config config/baselines/hifigan.py --gpu $1
# python -m promonet.train --config config/baselines/vits.py --gpu $1

# DSP-based baseline evaluations
# python -m promonet.evaluate --config config/baselines/psola.py --gpu $1
# python -m promonet.evaluate --config config/baselines/world.py --gpu $1
