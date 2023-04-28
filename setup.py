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
        'espnet',
        'librosa',
        'matplotlib',
        'monotonic_align @ git+ssh://git@github.com/resemble-ai/monotonic_align.git',
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
