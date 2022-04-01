# Production Data Science
A demonstration for Data Scientists of how to create and test a production 
machine learning model, from notebook to completed python package, hosted inference service, 
unit and integration tests, and continuous integration and deployment pipelines. 

## Table of Contents
* [What is this package?](#what-is-this-package)
  * [Demo Highlights](#demo-highlights)
  * [Package Components](#package-components)
* [How do I install and run this package?](#how-do-i-install-and-run-this-package)
  * [Installation](#installation)
  * [Dataset](#dataset)
  * [Build testing](#build-testing)
  * [Training](#training)
  * [Evaluation](#evaluation)
  * [Launch inference service](#launch-inference-service)
  * [Integration test](#local-integration-test)
* [What is the best way to compare model results using this template?](#what-is-the-best-way-to-compare-model-results-using-this-template)  
* [What steps can be automated with continuous integration and deployment pipelines?](#what-steps-can-be-automated-with-continuous-integration-and-deployment-pipelines)
* [How can I set up a continuous integration pipeline?](#how-can-i-set-up-a-continuous-integration-pipeline)

## What is this package? 
This package is a simple demonstration of a production machine learning package, 
from notebook to training functions with unit tests and implemented invoke service. 
This invoke service is meant to be adapted into a
[SageMaker custom inference container](https://docs.aws.amazon.com/sagemaker/latest/dg/adapt-inference-container.html).

The code is basic and meant to be used as a starting point for larger projects. Unit 
tests cover both logic and scripting with mocked resources, which should be helpful for Data Scientists 
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
* `test_integ/` Integration tests for a live inference service
* `Makefile`: Contains commands to build and test this package

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

### Dataset 
Copy the
[House Prices dataset from Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
and unzip into the /tmp folder for use with model training and evaluation. 

### Build testing 
Install tox with `pip install tox`; 
run build and unit tests using the `tox` command.  
After running, 
view the test coverage report in `build/coverage/index.html`. 

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

## What is the best way to compare model results using this template? 

If you create an eval script like in the example `make evaluate`, results 
are output to stdout then redirected into a `results.csv` file. The results are then committed 
to the git repository.  

If you use git tagging, you can use `git diff` to show differences
in the results file across different tags, e.g. `git diff tag1..tag2 -- eval/results.csv`.  

Model results should be added to pull requests to quickly view how each change impacts the results.

## What steps can be automated with continuous integration and deployment pipelines?

The [GoCD](https://docs.gocd.org) CI/CD pipeline automates all steps in the 
[How do I install...](#how-do-i-install-and-run-this-package) section above, where 
each step must succeed before the next step will execute, and any git push will 
automatically start the pipeline at step 1. 

The example CI/CD pipeline includes:
1. Code build and unit tests
1. Automated training and evaluation 
1. Bundling of trained model artifacts 
1. Deployment of hosted model service 
1. Integration tests of model service

**CI / CD build and test pipeline:**
<img src="https://raw.githubusercontent.com/bshurick/production-demo/main/doc/images/pipeline.png" />

## How can I set up a continuous integration pipeline? 

Pipeline yaml definition files for [GoCD](https://docs.gocd.org) 
are provided in the `pipeline/` folder.  
Follow these steps to start with the pipelines provided:
1. Launch a GoCD server docker container, e.g. `docker run -d -p8153:8153 --name "gocd" gocd/gocd-server:v22.1.0`
2. Launch a GoCD agent docker container using the provided Dockerfile
`configuration/Dockerfile.gocd-agent`, e.g.: 
```
docker build -t gocd-agent --file configuration/Dockerfile.gocd-agent .
docker run -d \
  -e GO_SERVER_URL=http://$(docker inspect --format='{{(index (index .NetworkSettings.IPAddress))}}' gocd):8153/go \
  --mount type=bind,source=/tmp,target=/tmp \
  gocd-agent
```

3. Launch interface in web browser, at `http://localhost:8153`
4. In Admin -> Config Repositories add this repository (url "https://github.com/bshurick/production-demo.git", branch 'main') with `pipeline/*.yaml` in "GoCD YAML files pattern" 
5. In Rules set Allow -> Pipeline Group -> prod-demo
6. In 'Agents' tab select a running agent and hit 'Enable'

The `docker run` command shown above assumes the [source data](#dataset) exists in the /tmp directory where the agent is launched.

Note that docker containers should be easy to deploy in cloud environments, with agents installed 
on each production environment and the GoCD server running on an independent "Pipelines" environment. 

It may be necessary to edit the GoCD Agent dockerfile if extra prereqs are added for your environment. 
