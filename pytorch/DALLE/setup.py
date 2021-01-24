from setuptools import setup, find_packages

setup(
    name='dalle-pytorch',
    packages=find_packages(),
    version='0.0.48',
    licence='MIT',
    description='DALL-E Pytorch',
    author='Olalekan Taofeek',
    url='https://github.com/Harphies/AI/tree/master/pytorch/DALLE',
    keywords=[
        'artificial intelligence',
        'attention mechanism',
        'transformers',
        'text-to-image'
    ],
    install_requires=[
        'axial_positional_embedding',
        'einops>=0.3',
        'torch>=1.6'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6'
    ],
)
