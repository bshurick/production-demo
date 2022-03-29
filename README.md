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

### Demo Highlights
1. Notebooks folder for parameter tuning and data analysis
1. Buildable python package with train, evaluate, and inference modules
1. Unit tests with code coverage reporting, automated with [tox](https://tox.wiki/en/latest/)
1. Auto-generated [package documentation](https://bshurick.github.io/production-demo/) 
via [Sphinx](https://www.sphinx-doc.org/en/master/)
1. Pipeline as code via [GoCD](https://www.gocd.org/)

### Package Components
* `pyproject.toml`: Build instructions 
* `setup.cfg`: Package metadata and PyPi dependencies
* `.gitignore`: Ignore some local files, such as data, when committing to git
* `notebooks/`: Development notebooks for modeling experiments 
* `src/`: Production code with train and evaluate scripts, and invoke function
* `doc/`: Sphinx documentation example; built during `tox` testing
* `test/`: Python unit tests 
* `launch`: Build the inference docker container and launch service locally or publish to ECR

## How do I install and run this package?

You may first need to install [Make](https://www.gnu.org/software/make/) on your machine, 
e.g. via [brew](https://formulae.brew.sh/formula/make) on MacOS. If the `make -version` command 
works already, then it's already installed. 

### Installation
1. `git clone https://github.com/bshurick/production-demo.git`  
2. `cd production-demo`  
3. `python3 -m venv .`  
4. `source bin/activate`  
5. `make`  

### Build testing 
Install tox with `pip install tox`; 
run build and unit tests using the `tox` command.  
After running, 
view the test coverage report in `build/coverage/index.html`. 

### Dataset 
Copy the
[House Prices dataset from Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
and unzip into a data/ folder in the package root directory for use with 
model training and evaluation. 

### Training 
To run train after building, [copy the dataset](#dataset) locally, then run `bin/HouseTrain`.

### Evaluation 
Run `bin/HouseEval` after copying the dataset locally to run cross validation and emit results as logs.  
Alternatively, run `make evaluate` to save results as a CSV in the `eval/` folder.

### Launch inference service
1. Install [Docker](https://docs.docker.com/get-docker/)
1. Install this package via [instructions](#installation)
1. Run [training](#training)
1. Run the deploy make script (`make deploy`) to build a docker container and run locally

### Local integration test
After [launching](#launch-inference-service) the inference service docker container locally,
run `pytest test_integ` to run integration tests on the running service.  

## How do I set up a continuous integration pipeline? 

Pipeline yaml definition files for GoCD are provided in the `pipeline/` folder.  
Follow these steps to start with the pipelines provided:
1. Follow the install steps for [GoCD](https://docs.gocd.org/current/installation/)
to install the server and agent locally 
1. Follow the [pipeline as code](https://docs.gocd.org/current/advanced_usage/pipelines_as_code.html)
step using `pipeline/*.yaml` files from this repo.

**CI / CD build and test pipeline:**
<img src="https://github.com/bshurick/production-demo/blob/doc-page/doc/images/pipeline.png?raw=true" />

## What is the best way to compare model results using this template? 

If you create an eval script like in the example `make evaluate`, results 
are output to stdout then redirected into a `results.csv` file. The results are then committed 
to the git repository. If you use git tagging, you can use `git diff` to show differences
in the results file across different tags, e.g. `git diff tag1..tag2 -- eval/results.csv`.
