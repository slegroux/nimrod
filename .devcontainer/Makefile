.PHONY: nimrod-dev nimrod

IMAGE_TAG = nimrod-dev

nimrod-dev:
	docker build --network host -t nimrod-dev -f Dockerfile .

nimrod:
	docker build --network host -t nimrod -f Dockerfile .

paperspace:
	docker build -t nimrod-paperspace -f Dockerfile.paperspace . #--platform linux/amd64

test:
	docker build -t nimrod-test -f Dockerfile.test .

base:
	docker build -t base -f Dockerfile.base .

push:
	docker image tag nimrod-paperspace slegroux/nimrod-paperspace
	docker image push slegroux/nimrod-paperspace