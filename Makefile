.PHONY: test build
all: test build

test:
	pip install -q -U tox && tox

build:
	pip install -e .

update-docs:
	git fetch origin doc-page && git checkout doc-page \
	&& rm -rf ./docs && mkdir ./docs && cp -r build/docs ./ \
	&& git add --all ./docs/* && git commit -m 'Update docs' \
	&& git push origin doc-page \
	&& git checkout main

clean:
	rm -rf build && rm -rf docs && rm -rf bin

format:
	pip install -q -U black && black .

evaluate:
	bin/HouseEval > eval/results.csv

integ-test:
	./launch && sleep 2 && pytest -vv test_integ
