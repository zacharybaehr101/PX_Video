"""
upload_bunny_renders.py

Uploads finished video renders to a Bunny Storage "renders/" folder and
prints the resulting URLs to the GitHub Actions job summary, so you don't
have to dig through workflow artifacts to grab a link.
"""

import argparse
import json
import os

import requests

BUNNY_STORAGE_HOST = "storage.bunnycdn.com"


def bunny_headers():
    headers = {"AccessKey": os.environ["BUNNY_STORAGE_API_KEY"]}
    return headers


def upload_file(local_path: str, remote_path: str):
    zone = os.environ["BUNNY_STORAGE_ZONE"]
    url = f"https://{BUNNY_STORAGE_HOST}/{zone}/{remote_path.lstrip('/')}"
    with open(local_path, "rb") as f:
        resp = requests.put(url, headers={**bunny_headers(), "Content-Type": "application/octet-stream"}, data=f, timeout=300)
    resp.raise_for_status()
    return url


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--renders-dir", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    output_prefix = manifest.get("output_prefix", "video")
    uploaded = []

    for fname in sorted(os.listdir(args.renders_dir)):
        if not fname.lower().endswith(".mp4"):
            continue
        local_path = os.path.join(args.renders_dir, fname)
        remote_path = f"renders/{output_prefix}/{fname}"
        url = upload_file(local_path, remote_path)
        uploaded.append({"file": fname, "url": url})
        print(f"Uploaded: {fname} -> {url}")

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as f:
            f.write(f"## Rendered videos: {manifest.get('title', output_prefix)}\n\n")
            for item in uploaded:
                f.write(f"- **{item['file']}**: {item['url']}\n")

    print(json.dumps({"uploaded": uploaded}, indent=2))


if __name__ == "__main__":
    main()
