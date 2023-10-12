.PHONY: nimrod-dev nimrod

IMAGE_TAG = nimrod-dev

nimrod-dev:
	docker build --network host -t nimrod-dev -f Dockerfile .

nimrod:
	docker build --network host -t nimrod -f Dockerfile .

nimrod-paperspace:
	docker build -t nimrod-paperspace -f Dockerfile.paperspace .

push:
	docker image tag nimrod-paperspace slegroux/nimrod-paperspace
	docker image push slegroux/nimrod-paperspace