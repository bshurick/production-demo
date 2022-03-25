# Production Data Science
A demonstration for Data Scientists of how to create and test a production 
machine learning model, from notebook to completed python package and hosted inference service.

## What is this package? 
This package is a simple demonstration of a production machine learning package, 
from notebook to training functions with unit tests and implemented invoke service. 
This invoke service is meant to be adapted into a
[SageMaker custom inference container](https://docs.aws.amazon.com/sagemaker/latest/dg/adapt-inference-container.html).

The code is basic and meant to be used as a starting point for larger projects. Unit 
tests cover both logic and scripting with mocks, which should be helpful for Data Scientists 
looking to improve their unit testing skills. 

### Project Highlights
1. Notebooks folder for parameter tuning and data analysis
1. Buildable python package with train, evaluate, and inference modules
1. Unit tests with >99% code coverage, automated with [tox](https://tox.wiki/en/latest/)
1. Auto-generated documentation via [Sphinx](https://www.sphinx-doc.org/en/master/)

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
5. `pip install .`  

### Build testing 
Install tox with `pip install tox`; 
run build and unit tests using the `tox` command.  
After running, 
check the test coverage HTML table in `build/coverage/index.html`. 

### Training 
To run train after building, copy the
[House Prices dataset from Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
and unzip into a data/ folder in the package root directory;
then, run `bin/HouseTrain`.

### Evaluation 
Run `bin/HouseEval` to run cross validation and emit results as logs. 
