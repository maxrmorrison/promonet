from setuptools import find_packages, setup


with open('README.md', encoding='utf8') as file:
    long_description = file.read()


setup(
    name='promovits',
    description='Adaptive End-to-End Speech Prosody Modification',
    version='0.0.1',
    author='Max Morrison',
    author_email='maxrmorrison@gmail.com',
    url='https://github.com/maxrmorrison/promovits',
    install_requires=[
        'Cython',
        'espnet',
        'librosa',
        'matplotlib',
        'numpy',
        'phonemizer',
        'pypar',
        'pyyaml',
        'scipy',
        'tensorboard',
        'torch',
        'torchaudio',
        'torchcrepe',
        'torchinfo',
        'torchvision',
        'tqdm',
        'Unidecode'],
    packages=find_packages(),
    package_data={'promovits': ['assets/*', 'assets/*/*']},
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=['audio', 'speech', 'prosody', 'pytorch', 'vits'])
