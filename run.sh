python -m promonet.data.download --datasets daps && \
python -m promonet.preprocess --datasets daps --gpu 0 && \
python -m promonet.partition --datasets daps --overwrite
# python -m promonet.partition --datasets daps --overwrite && \
# python -m promonet.train --config config/test.py --dataset vctk --gpus 0
