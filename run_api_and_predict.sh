#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="redlight-api:cpu"
CONTAINER_NAME="redlight-api"
PORT="${PORT:-8000}"
API_FILE="${API_FILE:-api_detectron2.py}"

if [[ ! -f "$API_FILE" ]]; then
  echo "API file '$API_FILE' not found in $(pwd). Set API_FILE=<file.py> or save your API as that name."
  exit 1
fi

if [[ ! -d "models" ]]; then
  echo "'models/' not found. Create it with: models/config.yaml, models/model_final.pth, models/classes.json"
  exit 1
fi
for f in config.yaml model_final.pth classes.json; do
  [[ -f "models/$f" ]] || { echo "Missing models/$f"; exit 1; }
done

if ! command -v docker &>/dev/null; then
  echo "Installing Docker (Ubuntu)..."
  curl -fsSL https://get.docker.com | sh
  sudo usermod -aG docker "$USER" || true
fi

echo "Building image $IMAGE_NAME ..."
docker build -t "$IMAGE_NAME" -f Dockerfile.api .

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
fi

echo "Starting container on port $PORT ..."
docker run -d --name "$CONTAINER_NAME" -p "$PORT:8000" \
  -v "$(pwd)/models:/app/models:ro" \
  -v "$(pwd)/$API_FILE:/app/api_detectron2.py:ro" \
  "$IMAGE_NAME"

echo -n "Waiting for API"
for i in {1..30}; do
  sleep 1
  if curl -s "http://127.0.0.1:${PORT}/" >/dev/null; then
    echo -e "\nAPI is up."; break
  fi
  echo -n "."
  if [[ "$i" -eq 30 ]]; then
    echo -e "\nAPI did not start in time."; docker logs "$CONTAINER_NAME" | tail -n 100; exit 1
  fi
done

read -rp "Enter image path to predict: " IMG
[[ -f "$IMG" ]] || { echo "File not found: $IMG"; exit 2; }

curl -sS -X POST "http://127.0.0.1:${PORT}/predict" -F "file=@${IMG}" | python3 -m json.tool || true
echo "Done. Stop with: docker rm -f $CONTAINER_NAME"
