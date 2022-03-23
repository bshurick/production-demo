# Production Data Science
This package contains a demonstration in several 
components of how to create and test a production 
machine learning model, from notebook to completed package.

## Package Components
* `pyproject.toml`: this tells python how to build your package. 
* `setup.cfg`: this includes package metadata and requirements
* `.gitignore`: ignore certain files when committing to git
* `notebooks/`: development notebooks 
* `src/`: actual production code with invoke service
* `docs/`: sphinx docs
* `test/`: python package tests 

## Installing 
1. `git clone https://github.com/bshurick/production-demo.git`  
2. `cd production-demo`  
3. `python3 -m venv .`  
4. `source bin/activate`  
5. `pip install . tox`  

Build test with `tox`.

To run train, copy [House Prices dataset from Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
and unzip into data/ folder, run `bin/HouseTrain`.
