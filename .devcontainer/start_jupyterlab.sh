#!/bin/env bash

jupyter lab --allow-root --ip=0.0.0.0 --port=8888 --no-browser \
    --NotebookApp.token='' --NotebookApp.password='' \
    --LabApp.trust_xheaders=True --LabApp.disable_check_xsrf=False --LabApp.allow_remote_access=True --LabApp.allow_origin='*'