from setuptools import setup, find_packages

setup(
    name='vitookit',
    author='Gent',
    author_email='clouderow@gmail.com',
    decscription='A toolkit for evaluating and analyzing vision models',
    keywords=[
    'artificial intelligence',
    'deep learning',
    'evaluation'
    ],
    version='0.1',
    packages=find_packages('.',include=['vitookit']),
    include_package_data=True,
    scripts=[
        'bin/vitrun',
        'bin/submitit',
    ],
    install_requires = [
        'grad-cam',
        'timm>=1.0.12',
        'torch>=1.7.1',
        'torchvision>=0.8.2',
        'tqdm',
        'wandb',
        'gin-config',
        'einops',
    ]
)