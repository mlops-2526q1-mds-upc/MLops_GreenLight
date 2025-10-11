# run_all.ps1
$ErrorActionPreference = "Stop"

# ------------------------------
# Configuration
# ------------------------------
$IMAGE_NAME = "my-python-env"
$DOCKERFILE_PATH = "./DockerfileMac"
$HOST_WORKDIR = (Get-Location).Path
$CONTAINER_WORKDIR = "/workspace"
$TMP_SCRIPT = "container_commands.sh"

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
echo "Running all commands..."
python3 ./mlops_greenlight/dataset.py
python3 ./mlops_greenlight/modeling/train.py
python3 ./mlops_greenlight/modeling/test.py
bash ./mlops_greenlight/predictions2mp4.sh ./models/predictions/ ./models/predictions/output.mp4
echo "Finished!"
"@

# --- write script with LF line endings (no CR) ---
if (Test-Path $TMP_SCRIPT) { Remove-Item $TMP_SCRIPT -Force }
$scriptLF = $CONTAINER_COMMANDS -replace "`r`n", "`n"       # CRLF -> LF
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)   # UTF-8 sin BOM
[System.IO.File]::WriteAllText($TMP_SCRIPT, $scriptLF, $utf8NoBom)

# ------------------------------
# Run the commands in container (CPU only)
# ------------------------------
docker run --rm `
  -v "${HOST_WORKDIR}:${CONTAINER_WORKDIR}" `
  -w $CONTAINER_WORKDIR `
  $IMAGE_NAME `
  bash $TMP_SCRIPT
