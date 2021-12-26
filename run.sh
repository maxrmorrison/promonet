python -m promovits.data.download --datasets vctk && \
python -m promovits.preprocess --datasets vctk --gpu 0 && \
python -m promovits.train --config config/test.py --datasets <datasets> --gpus 0
