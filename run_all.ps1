# run_all.ps1
$ErrorActionPreference = "Stop"

# ------------------------------
# Configuration
# ------------------------------
$IMAGE_NAME        = "my-python-env"
$DOCKERFILE_PATH   = "./Dockerfile_cpu"
$HOST_WORKDIR      = (Get-Location).Path
$CONTAINER_WORKDIR = "/workspace"
$TMP_SCRIPT        = "container_commands.sh"

# ------------------------------
# Build image if it doesn't exist
# ------------------------------
$IMAGE_EXISTS = docker images -q $IMAGE_NAME
if ([string]::IsNullOrEmpty($IMAGE_EXISTS)) {
    Write-Output "Docker image '$IMAGE_NAME' not found. Building..."
    docker build -t $IMAGE_NAME -f $DOCKERFILE_PATH .
} else {
    Write-Output "Docker image '$IMAGE_NAME' found. Skipping build."
}

# ------------------------------
# Commands to run inside container
# ------------------------------
$CONTAINER_COMMANDS = @"
set -e

echo "Running all commands..."

# --- DVC auth & data pull (reads env from Docker --env-file) ---
# Expected in .env: DAGSHUB_USER=..., DAGSHUB_TOKEN=..., DVC_REMOTE=origin|localstore
: "\${DVC_REMOTE:=origin}"   # default if not set
mkdir -p .dvcstore || true   # harmless if not used

# Configure DVC remote credentials into .dvc/config.local (never committed)
dvc remote modify "\$DVC_REMOTE" auth basic       || true
dvc remote modify "\$DVC_REMOTE" user "\$DAGSHUB_USER" --local    || true
dvc remote modify "\$DVC_REMOTE" password "\$DAGSHUB_TOKEN" --local || true

# Pull data for this Git commit (ok if remote is localstore and no creds)
dvc pull -r "\$DVC_REMOTE" -q || true

# --- (Optional) MLflow env; uncomment if you use MLflow locally ---
# export MLFLOW_TRACKING_URI=file:/workspace/mlruns
# export MLFLOW_EXPERIMENT_NAME=GreenLight

# --- Your pipeline ---
python3 ./mlops_greenlight/dataset.py
python3 ./mlops_greenlight/modeling/train.py
python3 ./mlops_greenlight/modeling/test.py
bash   ./mlops_greenlight/predictions2mp4.sh ./models/predictions/ ./models/predictions/output.mp4

echo "Finished!"
"@

# --- write script with LF line endings (no CR) ---
if (Test-Path $TMP_SCRIPT) { Remove-Item $TMP_SCRIPT -Force }
$scriptLF  = $CONTAINER_COMMANDS -replace "`r`n", "`n"    # CRLF -> LF
$utf8NoBom = New-Object System.Text.UTF8Encoding($false) # UTF-8 without BOM
[System.IO.File]::WriteAllText($TMP_SCRIPT, $scriptLF, $utf8NoBom)

# ------------------------------
# Run the commands in container (CPU only)
# ------------------------------
$envFileArg = ""
if (Test-Path ".env") { $envFileArg = "--env-file .env" }

docker run --rm `
  $envFileArg `
  -v "${HOST_WORKDIR}:${CONTAINER_WORKDIR}" `
  -w $CONTAINER_WORKDIR `
  $IMAGE_NAME `
  bash $TMP_SCRIPT


