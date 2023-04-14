from setuptools import find_packages, setup


with open('README.md', encoding='utf8') as file:
    long_description = file.read()


setup(
    name='promonet',
    description='Prosody Modification Network',
    version='0.0.1',
    author='Max Morrison',
    author_email='maxrmorrison@gmail.com',
    url='https://github.com/maxrmorrison/promonet',
    install_requires=[
        'Cython',
        'espnet',
        'librosa',
        'matplotlib',
        'numpy',
        # TEMPORARY - exclude until release
        # 'pysodic',
        # TEMPORARY - GPL dependency
        'psola',
        'pyworld',
        'pyyaml',
        'scipy',
        'tensorboard',
        'torch',
        'torchaudio',
        'tqdm',
        'yapecs'],
    packages=find_packages(),
    package_data={'promonet': ['assets/*', 'assets/*/*']},
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=['audio', 'speech', 'prosody', 'pytorch', 'vits'],
    license='MIT')
