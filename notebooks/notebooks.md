# Jupiter Notebooks

## Jupiter Servers

- Google Colab
- Kaggle Notebook Servers

## Utilities

- Reusing notebook cells from another notebook - `%run "/path/to/code-snippets.ipynb"`

## Manage Python versions

- pyenv versions

## Running notebooks Locally

### pip
```sh
~/.pyenv/versions/3.10.0
pyenv global 3.10.0
python --version
cd notebooks

pyenv virtualenv 3.10.0 venv
pyenv activate venv
pyenv deactivate venv
python --version
pyenv uninstall venv
pyenv uninstall 3.10.0
pyenv global
pyenv local
pyenv shell
pyenv version
which python

# Activate virtual environment
python -m venv venv
source venv/bin/activate
echo $VIRTUAL_ENV

pip install --upgrade pip
pip install jupyter
pip install ipykernel
python -m ipykernel install --user --name=venv-kernel
which jupyter  # Verify jupyter installation 
jupyter notebook  # 
pip install -r requirements.txt                                                     🐍 notebooks ⬢ system 06:54:52


# PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# TensorFlow
pip install tensorflow

# Save dependencies
pip freeze > requirements.txt
```

### Conda