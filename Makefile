.DEFAULT_GOAL := build

test:
	tox

build:
	pip install .

update-docs:
	git fetch origin doc-page && git checkout doc-page \
	&& rm -rf ./docs && mkdir ./docs && cp -r build/docs/* ./docs \
	&& git add -A ./docs/* && git commit -m 'Update docs' \
	&& git push origin doc-page \
	&& git checkout main

clean:
	rm -rf build && rm -rf docs && rm -rf bin
