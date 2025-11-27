#!/usr/bin/env bash
# call_pph_api.sh — cURL caller for image prediction
# Usage:
#   ./call_pph_api.sh http://<EC2_PUBLIC_IP>:8000 /predict /path/to/image.jpg pph
#   ./call_pph_api.sh http://<EC2_PUBLIC_IP>:8000 "/{model}/predict" /path/to/image.jpg pph

set -euo pipefail

BASE="${1:-}"
ENDPOINT="${2:-/predict}"
IMAGE="${3:-}"
MODEL="${4:-pph}"
FIELD="${FIELD:-file}"

if [[ -z "$BASE" || -z "$IMAGE" ]]; then
  echo "Usage: $0 <base-url> <endpoint> <image-path> <model=pph>"
  exit 2
fi

if [[ "$ENDPOINT" == *"{model}"* ]]; then
  EP="${ENDPOINT//\{model\}/$MODEL}"
  URL="${BASE%/}${EP}"
  echo "POST $URL (multipart form ${FIELD}=@${IMAGE})"
  curl -sS -X POST "$URL" -F "${FIELD}=@${IMAGE}" | python -m json.tool
else
  URL="${BASE%/}${ENDPOINT}"
  echo "POST $URL?model=$MODEL (multipart form ${FIELD}=@${IMAGE})"
  curl -sS -G -X POST --data-urlencode "model=$MODEL" "$URL" -F "${FIELD}=@${IMAGE}" | python -m json.tool
fi
