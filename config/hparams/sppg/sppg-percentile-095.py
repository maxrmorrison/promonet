MODULE = 'promonet'

# Configuration name
CONFIG = 'sppg-percentile-095'

# Type of sparsification used for ppgs
# One of ['constant', 'percentile', 'topk', None]
SPARSE_PPG_METHOD = 'percentile'

# Threshold for ppg sparsification.
# In [0, 1] for 'contant' and 'percentile'; integer > 0 for 'topk'.
SPARSE_PPG_THRESHOLD = 0.95
