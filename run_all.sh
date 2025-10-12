#!/bin/bash
set -e  # exit on first error

# ------------------------------
# Configuration
# ------------------------------
IMAGE_NAME="my-python-env"
DOCKERFILE_PATH="./Dockerfile"
HOST_WORKDIR="$(pwd)"
CONTAINER_WORKDIR="/workspace"

# ------------------------------
# Build image if it doesn't exist
# ------------------------------
if [[ "$(docker images -q $IMAGE_NAME 2> /dev/null)" == "" ]]; then
    echo "Docker image '$IMAGE_NAME' not found. Building..."
    docker build -t "$IMAGE_NAME" -f "$DOCKERFILE_PATH" .
else
    echo "Docker image '$IMAGE_NAME' found. Skipping build."
fi

# ------------------------------
# Commands to run inside container
# ------------------------------
CONTAINER_COMMANDS=$(cat <<'EOF'
echo "Running all commands..."
# Example commands:
python3 ./mlops_greenlight/dataset.py
python3 ./mlops_greenlight/convert.py
python3 ./mlops_greenlight/modeling/train.py
python3 ./mlops_greenlight/modeling/test.py
bash mlops_greenlight/predictions2mp4.sh ./models/predictions/ ./models/predictions/output.mp4
echo "Finished!"
EOF
)

# ------------------------------
# Run the commands in container with GPU access
# ------------------------------
docker run --rm --gpus all \
    -v "$HOST_WORKDIR":"$CONTAINER_WORKDIR" \
    -w "$CONTAINER_WORKDIR" \
    "$IMAGE_NAME" \
    bash -c "$CONTAINER_COMMANDS"

