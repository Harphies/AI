from setuptools import setup, find_packages

setup(
    name='vit-pytorch',
    packages=find_packages(exclude=['examples']),
    version='0.25.6',
    licence='MIT',
    description='Vision Transformer (ViT) - Pytorch',
    author='Olalekan Taofeek',
    author_email="a@gmail.com",
    url='https://github.com/harphies/vit-pytorch',
    keywords=[
        'artificial intelligence',
        'attention mechanism',
        'image recognition'
    ],
    install_requires=[
        'eniops>=0.3',
        'torch>=1.6',
        'torchvision'
    ],
    setup_requires=[
        'pytest-runner',
    ],
    test_requires=[
        'pytest'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Licence :: OSI Approved :: MIT Licence',
        'Programming Langiage :: Python :: 3.8'
    ]
)
