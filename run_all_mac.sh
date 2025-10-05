#!/usr/bin/env bash
set -euo pipefail

# ------------------------------
# Config
# ------------------------------
IMAGE_NAME="detectron2-cpu"        # your CPU image name
DOCKERFILE_PATH="./DockerfileMac"  # your CPU Dockerfile
HOST_WORKDIR="$(pwd)"
CONTAINER_WORKDIR="/workspace"

# ------------------------------
# Build if missing (Mac M-series => arm64)
# ------------------------------
if [[ -z "$(docker images -q "$IMAGE_NAME" 2>/dev/null)" ]]; then
  echo "Docker image '$IMAGE_NAME' not found. Building..."
  docker buildx build --platform=linux/arm64 -t "$IMAGE_NAME" -f "$DOCKERFILE_PATH" .
else
  echo "Docker image '$IMAGE_NAME' found. Skipping build."
fi

# ------------------------------
# Commands to run inside the container
# ------------------------------
CONTAINER_COMMANDS=$(cat <<'EOF'
set -e
echo "[container] Running all commands..."

# Safe (CPU-only) and fast environment
export FORCE_CPU=1
export CODECARBON_DISABLED=1
export MLOPS_DISABLE_DAGSHUB=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MAX_ITER=5       # quick training
export MAX_TEST=10     # inference on 100 images
export PYTHONPATH="/workspace:${PYTHONPATH:-}"

# Ensure pytest (remove if already baked into the image)
python3 -m pip install -q --no-input pytest

echo ">> pytest (unit)"
pytest -q -m "not integration" -vv

echo ">> pytest (integration smoke)"
pytest -q -m integration -vv || echo "(integration tests skipped/failed — continuing pipeline)"

echo ">> training"
python3 ./mlops_greenlight/modeling/train.py

echo ">> inference"
python3 ./mlops_greenlight/modeling/test.py

echo ">> render video"
bash mlops_greenlight/predictions2mp4.sh ./models/predictions/ ./models/predictions/output.mp4 || echo "(video step skipped)"

echo "[container] Finished!"
EOF
)

# ------------------------------
# Run (NO GPU on Mac)
# ------------------------------
docker run --rm \
  -v "$HOST_WORKDIR":"$CONTAINER_WORKDIR" \
  -w "$CONTAINER_WORKDIR" \
  "$IMAGE_NAME" \
  bash -lc "$CONTAINER_COMMANDS"
