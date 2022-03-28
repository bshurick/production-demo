.PHONY: test build
all: test build

test:
	pip3 install -q -U tox && tox

build:
	pip3 install -e .

update-docs: test
	git fetch origin doc-page && git checkout doc-page \
	&& rm -rf ./docs && mkdir ./docs && cp -r build/docs ./ \
	&& git add --all ./docs/* && git commit -m 'Update docs' \
	&& git push origin doc-page \
	&& git checkout main

clean:
	rm -rf build && rm -rf docs && rm -rf bin

format:
	pip3 install -q -U black && black .

train:
	bin/HouseTrain

evaluate:
	bin/HouseEval > eval/results.csv

deploy:
	./launch

integ-test:
	sleep 2 && pytest -vv test_integ
