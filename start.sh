#!/bin/sh

# Start FaceFusion in the background
python facefusion.py run --execution-providers cuda &

# Start BentoML in the foreground
bentoml serve faceswap_bento_service:AIToolsAPI --port 3000
