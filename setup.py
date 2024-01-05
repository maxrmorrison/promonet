from setuptools import find_packages, setup


with open('README.md', encoding='utf8') as file:
    long_description = file.read()


setup(
    name='promonet',
    description='Prosody Modification Network',
    version='0.0.1',
    author='Interactive Audio Lab',
    author_email='interactiveaudiolab@gmail.com',
    url='https://github.com/maxrmorrison/promonet',
    install_requires=[
        'alias-free-torch',
        'GPUtil',
        'jiwer',
        'librosa',
        'matplotlib',
        'numpy',
        'openai-whisper',
        'penn',
        # 'ppgs',  # TEMPORARY - install manually until release
        'psola',  # TEMPORARY - GPL dependency
        'pypar',
        'pyworld',
        'resemblyzer',
        'scipy',
        'torch<2.0.0',
        'torchaudio<2.0.0',
        'torchutil',
        'umap-learn',
        'vocos[train]',
        'yapecs',
    ],
    packages=find_packages(),
    package_data={'promonet': ['assets/*', 'assets/*/*', 'assets/*/*/*']},
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=['speech', 'prosody', 'editing', 'synthesis', 'pronunciation'],
    license='MIT')
