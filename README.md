# GeoAlert FineTuning

## Pre-requirements
#### `python >= 3.9`
#### `pip`
#### `pipenv`

## Install

#### 1. Create `.venv` folder at root path of the project
#### 2. Run `pipenv shell` to activate virtual environment
#### 3. Run 
- `pipenv install` to install the latest versions of dependencies
- `pipenv install --ignore-pipfile` to install used versions of dependencies at the moment of development

## Configs
#### `finetune.yaml` is a main file of configuration, where dataset path and training params are set

## Run
#### All scripts are called from `main.py` file with args

Example:
```
pipenv run python3 main.py
```

Testing model:
```
pipenv run python3 test.py --model=PATH
```
