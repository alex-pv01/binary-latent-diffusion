from setuptools import setup, find_packages

setup(
    name='binary latent diffusion',
    version='0.0.1',
    description='Binary Latent Diffusion',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'tqdm',
    ],
)

