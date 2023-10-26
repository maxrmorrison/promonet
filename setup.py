from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

import numpy as np


with open('README.md', encoding='utf8') as file:
    long_description = file.read()


setup(
    name='promonet',
    description='Prosody Modification Network',
    version='0.0.1',
    author='Interactive Audio Lab',
    author_email='interactiveaudiolab@gmail.com',
    url='https://github.com/maxrmorrison/promonet',
    ext_modules=cythonize(
        Extension(
            'promonet.model.align.mas',
            sources=['promonet/model/align/align.pyx'],
            include_dirs=[np.get_include()]
        ),
        compiler_directives={'language_level': '3'}
    ),
    include_dirs=[np.get_include(), 'promonet/model/align'],
    setup_requires=['numpy', 'cython'],
    install_requires=[
        'alias-free-torch',
        'espnet',
        'jiwer',
        'librosa',
        'matplotlib',
        'numpy<1.24',
        'openai-whisper',
        # 'ppgs',  # TEMPORARY - install manually until release
        'psola',  # TEMPORARY - GPL dependency
        'pypar',
        'pyworld',
        'pyyaml',
        'resemblyzer',
        'scipy',
        'tensorboard',
        'torch<2.0.0',
        'torchutil',
        'torchaudio<2.0.0',
        'tqdm',
        'umap-learn',
        'vocos[train]',
        'yapecs',
    ],
    packages=find_packages(),
    package_data={'promonet': ['assets/*', 'assets/*/*']},
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=['speech', 'prosody', 'editing', 'synthesis', 'pronunciation'],
    license='MIT')
