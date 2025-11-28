#!/usr/bin/env python3
"""
call_pph_api.py — client for sending an image to endpoint on EC2.
"""
import argparse
import json
import mimetypes
from pathlib import Path
import requests

def build_url(base: str, endpoint: str, model: str):
    base = base.rstrip("/")
    ep = endpoint if endpoint.startswith("/") else "/" + endpoint
    if "{model}" in ep:
        ep = ep.replace("{model}", model)
    return base + ep

def main():
    ap = argparse.ArgumentParser(description="Send an image to a FastAPI prediction endpoint.")
    ap.add_argument("--base", required=True, help="Base URL (e.g., http://<EC2_PUBLIC_IP>:8000)")
    ap.add_argument("--endpoint", default="/predict", help="Endpoint path (e.g., /predict or /{model}/predict)")
    ap.add_argument("--image", required=True, help="Path to image file")
    ap.add_argument("--model", default="pph", help="Model name, default: pph")
    ap.add_argument("--file-field", default="file", help="Form field name expected by the API (default: file)")
    ap.add_argument("--timeout", type=int, default=90, help="HTTP timeout seconds")
    args = ap.parse_args()

    url = build_url(args.base, args.endpoint, args.model)
    params = {} if "{model}" in args.endpoint else {"model": args.model}

    img_path = Path(args.image)
    if not img_path.exists():
        raise SystemExit(f"Image not found: {img_path}")

    mime = mimetypes.guess_type(str(img_path))[0] or "application/octet-stream"
    files = {args.file_field: (img_path.name, open(img_path, "rb"), mime)}

    print(f"POST {url}")
    if params:
        print(f"  params: {params}")
    print(f"  sending: {img_path} as '{args.file_field}'")

    try:
        resp = requests.post(url, params=params, files=files, timeout=args.timeout)
    finally:
        files[args.file_field][1].close()

    print(f"\n[{resp.status_code}]")
    ctype = resp.headers.get("content-type", "")
    if "application/json" in ctype:
        try:
            print(json.dumps(resp.json(), indent=2))
        except Exception:
            print(resp.text[:1000])
    else:
        print(resp.text[:1000])

if __name__ == "__main__":
    main()
