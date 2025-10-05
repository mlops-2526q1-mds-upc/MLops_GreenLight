#!/usr/bin/env bash
set -euo pipefail

# ------------------------------
# Config
# ------------------------------
IMAGE_NAME="detectron2-cpu"        # usa el tag de la imagen que ya construiste
DOCKERFILE_PATH="./DockerfileMac"  # pon aquí tu Dockerfile CPU (no CUDA)
HOST_WORKDIR="$(pwd)"
CONTAINER_WORKDIR="/workspace"

# ------------------------------
# Build si no existe (Mac M-series => arm64)
# ------------------------------
if [[ -z "$(docker images -q "$IMAGE_NAME" 2>/dev/null)" ]]; then
  echo "Docker image '$IMAGE_NAME' not found. Building..."
  docker buildx build --platform=linux/arm64 -t "$IMAGE_NAME" -f "$DOCKERFILE_PATH" .
else
  echo "Docker image '$IMAGE_NAME' found. Skipping build."
fi

# ------------------------------
# Comandos a ejecutar dentro del contenedor
# ------------------------------
CONTAINER_COMMANDS=$(cat <<'EOF'
set -e
echo "[container] Running all commands..."
python3 ./mlops_greenlight/modeling/train.py
python3 ./mlops_greenlight/modeling/test.py
bash mlops_greenlight/predictions2mp4.sh ./models/predictions/ ./models/predictions/output.mp4
echo "[container] Finished!"
EOF
)

# ------------------------------
# Run (SIN GPU en Mac)
# ------------------------------
docker run --rm \
  -v "$HOST_WORKDIR":"$CONTAINER_WORKDIR" \
  -w "$CONTAINER_WORKDIR" \
  "$IMAGE_NAME" \
  bash -lc "$CONTAINER_COMMANDS"
