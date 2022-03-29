.PHONY: test build
all: test build

test:
	pip3 install -q -U tox && tox

build:
	docker build -t prod-demo --no-cache \
	--file configuration/Dockerfile \
	--output configuration/prod-demo-build . \
	&& tar -cvzf \
	configuration/prod-demo-build.tar.gz \
	configuration/prod-demo-build \
	2>/dev/null \
	&& rm -rf configuration/prod-demo-build

update-docs: test
	git fetch origin doc-page && git checkout doc-page \
	&& rm -rf ./docs && mkdir ./docs && cp -r build/docs ./ \
	&& git add --all ./docs/* && git commit -m 'Update docs' \
	&& git push origin doc-page \
	&& git checkout main

clean:
	rm -rf build && rm -rf docs && rm -rf bin \
	&& docker stop prod-demo \
	&& docker rm prod-demo

format:
	pip3 install -q -U black && black .

train:
	bin/HouseTrain

evaluate:
	bin/HouseEval > eval/results.csv

deploy: build
	docker load < prod-demo-build.tar.gz \
	&& docker run -d -p 8000:8000 --name prod-demo prod-demo-build.tar.gz

integ-test:
	sleep 2 && pytest -vv test_integ
