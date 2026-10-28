#!/usr/bin/env bash
# Set JUPYTER_TOKEN before launching; the server refuses to start without one.
: "${JUPYTER_TOKEN:?set JUPYTER_TOKEN to a secret token before starting JupyterLab}"

jupyter lab --allow-root --ip=0.0.0.0 --port=8888 --no-browser \
    --IdentityProvider.token="$JUPYTER_TOKEN" \
    --LabApp.trust_xheaders=True --LabApp.disable_check_xsrf=False --LabApp.allow_remote_access=True --LabApp.allow_origin='*'
