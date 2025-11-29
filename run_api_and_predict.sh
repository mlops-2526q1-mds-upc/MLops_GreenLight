
#!/usr/bin/env bash
set -euo pipefail

DOCKERFILE="Dockerfile_cpu_vm"
IMAGE_NAME="redlight-api:cpu"
CONTAINER_NAME="redlight-api"
PORT=8000

# 0) Sanity: fastapi.py must exist in project root (or adjust CMD in Dockerfile)
if [[ ! -f ".\tests\fastapi.py" ]]; then
  echo "❌ fastapi.py not found in $(pwd). Either create fastapi.py (with 'app = FastAPI(...)')"
  echo "   or change Dockerfile CMD to match your filename (e.g., uvicorn my_api:app)."
  exit 1
fi

# 1) Docker
if ! command -v docker &>/dev/null; then
  echo "🔧 Installing Docker (Ubuntu)..."
  curl -fsSL https://get.docker.com | sh
  sudo usermod -aG docker "$USER" || true
fi
DOCKER="docker"; $DOCKER ps >/dev/null 2>&1 || DOCKER="sudo docker"

# 2) Build
echo "🔨 Building image '$IMAGE_NAME' from '$DOCKERFILE' ..."
$DOCKER build -t "$IMAGE_NAME" -f "$DOCKERFILE" .

# 3) Run (mount project → /workspace so CMD 'uvicorn api:app' can see api.py)
if $DOCKER ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  $DOCKER rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
fi
echo "▶️  Starting container '$CONTAINER_NAME' on :$PORT ..."
$DOCKER run -d --name "$CONTAINER_NAME" -p "$PORT:8000" \
  -v "$(pwd):/workspace:ro" \
  "$IMAGE_NAME"

# 4) Give Uvicorn a moment to boot
echo -n "⏳ Waiting for API to start"
for i in {1..30}; do
  sleep 1; echo -n "."
  code=$(
    curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${PORT}/docs" || echo 000
  )
  [[ "$code" == "200" || "$code" == "404" ]] && break
done
echo

# 5) Ask for an image & predict
read -rp "📸 Enter image path to predict: " IMG
if [[ ! -f "$IMG" ]]; then
  echo "❌ File not found: $IMG"; exit 2
fi
echo "📤 POST http://127.0.0.1:${PORT}/predict"
if python3 -c "import sys" 2>/dev/null; then
  curl -sS -X POST "http://127.0.0.1:${PORT}/predict" -F "file=@${IMG}" | python3 -m json.tool
else
  curl -sS -X POST "http://127.0.0.1:${PORT}/predict" -F "file=@${IMG}"
fi

echo "✅ Done. Stop the API with: $DOCKER rm -f $CONTAINER_NAME"
