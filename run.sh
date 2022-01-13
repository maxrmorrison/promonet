python -m promovits.data.download --datasets daps && \
python -m promovits.preprocess --datasets daps --gpu 0 && \
python -m promovits.partition --datasets daps --overwrite
# python -m promovits.partition --datasets daps --overwrite && \
# python -m promovits.train --config config/test.py --dataset vctk --gpus 0
