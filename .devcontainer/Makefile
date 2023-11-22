.PHONY: nimrod-dev nimrod

IMAGE_NAME ?= nvidia

nimrod-dev:
	docker build --network host -t nimrod-dev -f Dockerfile .

nimrod:
	docker build --network host -t nimrod -f Dockerfile.nimrod .

paperspace:
	docker build --platform linux/amd64 -t paperspace -f Dockerfile.paperspace . #--platform linux/amd64

test:
	docker build -t test -f Dockerfile.test .

nvidia:
	docker build --platform=linux/amd64 -t nvidia -f Dockerfile.nvidia .

base:
	docker build -t base -f Dockerfile.base .

nbdev:
	docker build -t nbdev -f Dockerfile.paperspace.base .

tag:
	docker image tag ${IMAGE_NAME} slegroux/${IMAGE_NAME}

push:
	docker image push slegroux/${IMAGE_NAME}

run:
	docker run -p 8888:8888 -it --rm ${IMAGE_NAME} 