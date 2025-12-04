import os
import json
import subprocess
from pathlib import Path
import pytest

HOST = os.environ.get("ML_PUBLIC_HOST", "56.228.22.182")
PORT = os.environ.get("ML_PUBLIC_PORT", "8000")
ENDPOINT = os.environ.get("ML_PUBLIC_ENDPOINT", f"http://{HOST}:{PORT}/predict")
IMG = os.environ.get("ML_LOCAL_IMAGE", "35112.png")

def _curl_bin():
    # A PowerShell, fem servir curl.exe per evitar l'alias Invoke-WebRequest
    return "curl.exe" if os.name == "nt" else "curl"

@pytest.mark.integration
def test_public_predict_and_print_response():
    img = Path(IMG)
    if not img.exists():
        pytest.skip(f"Image not found: {img.resolve()} (set ML_LOCAL_IMAGE)")

    curl = _curl_bin()
    file_arg = f"file=@{str(img)}"

    proc = subprocess.run(
        [curl, "-sS", "-X", "POST", ENDPOINT, "-F", file_arg],
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 0, f"curl failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    body = proc.stdout.strip()
    assert body, "Empty response from API"

    # --- imprimeix la resposta a consola ---
    print("\n=== API RESPONSE (raw) ===")
    print(body)

    # --- intenta pretty-print JSON i desa-ho a fitxer ---
    artifacts_dir = Path("tests") / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    try:
        payload = json.loads(body)
        pretty = json.dumps(payload, indent=2, ensure_ascii=False)
        (artifacts_dir / "last_response.json").write_text(pretty, encoding="utf-8")
        print("\n=== API RESPONSE (pretty JSON) saved to: tests/artifacts/last_response.json ===")
        print(pretty)
    except json.JSONDecodeError:
        (artifacts_dir / "last_response.txt").write_text(body, encoding="utf-8")
        print("\n(Response is not JSON; saved raw body to tests/artifacts/last_response.txt)")

