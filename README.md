# Production Data Science
This package contains a demonstration in several 
components of how to create and test a production 
machine learning model, from notebook to completed package.


## What is this package? 
This package is a simple demonstration of a production machine learning package, 
from notebook to training functions with unit tests and implemented invoke service. 
This invoke service is meant to be adapted into a
[SageMaker custom inference container](https://docs.aws.amazon.com/sagemaker/latest/dg/adapt-inference-container.html).

### Package Components
* `pyproject.toml`: Build instructions 
* `setup.cfg`: Package metadata and PyPi dependencies
* `.gitignore`: Ignore certain files when committing to git
* `notebooks/`: Development notebooks for modeling experiments 
* `src/`: Production code with train and evaluate scripts, and invoke function
* `doc/`: Sphinx documentation example; built during `tox` testing
* `test/`: python unit tests 

## How do I install and run this package?

### Installation
1. `git clone https://github.com/bshurick/production-demo.git`  
2. `cd production-demo`  
3. `python3 -m venv .`  
4. `source bin/activate`  
5. `pip install . tox`  

### Build testing 
Run build and unit tests with `tox`. After running, 
check test coverage HTML in `build/coverage/index.html`. 

### Training 
To run train, copy [House Prices dataset from Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
and unzip into data/ folder, then run `bin/HouseTrain`.

### Evaluation 
Run `bin/HouseEval` to run cross validation and emit results as logs. 
