.PHONY: test build
all: test build

# Run unit tests in sandboxed test environment
test:
	pip3 install -q -U tox && tox

# Build docker environment and save as tarfile 
build:
	docker build -t prod-demo --no-cache \
	--file configuration/Dockerfile \
	--output configuration/prod-demo-build . \
	&& tar -cvzf configuration/prod-demo-build.tar.gz configuration/prod-demo-build \
	&& rm -rf configuration/prod-demo-build

# Generate documentation and upload to git documentation page
update-docs: test
	git fetch origin doc-page && git checkout doc-page \
	&& rm -rf ./docs && mkdir ./docs && cp -r build/docs ./ \
	&& git add --all ./docs/* && git commit -m 'Update docs' \
	&& git push origin doc-page \
	&& git checkout main

# Clean up build artifacts and remove docker container 
clean:
	rm -rf build && rm -rf docs && rm -rf bin \
	&& docker stop prod-demo \
	&& docker rm prod-demo

# Run 'black' to format code and export notebook as markdown 
format:
	pip3 install -q -U black jupyter black[jupyter] \
	&& black . \
	&& jupyter nbconvert --to markdown notebooks/*.ipynb

# Install package in a virtual environment and train 
train:
	python3 -m venv . \
	&& source bin/activate \
	&& pip3 install . \
	&& bin/HouseTrain

# Install package in a virtual environment and update eval results file 
evaluate:
	python3 -m venv . \
	&& source bin/activate \
	&& pip3 install . \
	&& bin/HouseEval > eval/results.csv

# Run a docker image from tarfile 
deploy:
	docker import prod-demo-build.tar.gz \
	&& docker run -d -p 8000:8000 --name prod-demo prod-demo

# Run integration tests on running service
integ-test:
	pip3 install -q -U pytest requests \
	&& pytest -vv test_integ
