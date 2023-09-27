.PHONY: nimrod-dev nimrod

IMAGE_TAG = nimrod-dev

nimrod-dev:
	docker build --network host -t nimrod-dev -f Dockerfile .

nimrod:
	docker build --network host -t nimrod -f Dockerfile .
