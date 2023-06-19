from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

import numpy as np


with open('README.md', encoding='utf8') as file:
    long_description = file.read()


setup(
    name='promonet',
    description='Prosody Modification Network',
    version='0.0.1',
    author='Max Morrison',
    author_email='maxrmorrison@gmail.com',
    url='https://github.com/maxrmorrison/promonet',
    ext_modules=cythonize(
        Extension(
            'promonet.model.align.mas',
            sources=['promonet/model/align/align.pyx'],
            include_dirs=[np.get_include()]
        ),
        compiler_directives={'language_level': '3'}),
    include_dirs=[np.get_include(), 'promonet/model/align'],
    setup_requires=['numpy', 'cython'],
    install_requires=[
        'alias-free-torch',
        'espnet',
        'librosa',
        'matplotlib',
        'numpy<1.24',
        # 'pysodic',  # TEMPORARY - install manually until release of pysodic
        'psola',  # TEMPORARY - GPL dependency
        'pyworld',
        'pyyaml',
        'scipy',
        'tensorboard',
        'torch<2.0.0',
        'torchaudio<2.0.0',
        'tqdm',
        'yapecs'],
    packages=find_packages(),
    package_data={'promonet': ['assets/*', 'assets/*/*']},
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=['audio', 'speech', 'prosody', 'pytorch', 'vits'],
    license='MIT')
