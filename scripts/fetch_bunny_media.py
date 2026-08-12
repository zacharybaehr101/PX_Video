"""
fetch_bunny_media.py

Pulls every file a manifest references (voice, music, speaker clips, and
the entire b-roll media_folder) from Bunny Storage down to a local folder
for the render step. Uses the Bunny Storage pull-zone REST API.

Requires env vars: BUNNY_STORAGE_API_KEY, BUNNY_STORAGE_ZONE
"""

import argparse
import json
import os

import requests

BUNNY_STORAGE_HOST = "storage.bunnycdn.com"


def bunny_headers():
    return {"AccessKey": os.environ["BUNNY_STORAGE_API_KEY"]}


def download_file(remote_path: str, local_path: str):
    zone = os.environ["BUNNY_STORAGE_ZONE"]
    url = f"https://{BUNNY_STORAGE_HOST}/{zone}/{remote_path.lstrip('/')}"
    resp = requests.get(url, headers=bunny_headers(), stream=True, timeout=60)
    resp.raise_for_status()
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with open(local_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)


def list_folder(remote_folder: str):
    zone = os.environ["BUNNY_STORAGE_ZONE"]
    url = f"https://{BUNNY_STORAGE_HOST}/{zone}/{remote_folder.rstrip('/')}/"
    resp = requests.get(url, headers=bunny_headers(), timeout=60)
    resp.raise_for_status()
    return resp.json()


def remote_path_from_url_or_key(value: str) -> str:
    """Accepts either a full CDN URL or a bare storage-relative path."""
    if value.startswith("http"):
        # Strip scheme + host, leaving the storage-relative path
        return "/".join(value.split("/")[3:])
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    os.makedirs(args.out_dir, exist_ok=True)

    # Voice + music
    if manifest.get("voice_audio_url"):
        remote = remote_path_from_url_or_key(manifest["voice_audio_url"])
        download_file(remote, os.path.join(args.out_dir, "voice.mp3"))
        print("Downloaded voice.mp3")

    music_cfg = manifest.get("music", {})
    if music_cfg.get("mode") == "specific" and music_cfg.get("track"):
        remote = remote_path_from_url_or_key(music_cfg["track"])
        local_name = os.path.basename(remote)
        download_file(remote, os.path.join(args.out_dir, local_name))
        print(f"Downloaded music track: {local_name}")
    elif music_cfg.get("mode") == "auto":
        # Pull the whole tagged library folder; picking happens at build time
        # against assets/music-library-index.json mood tags.
        try:
            entries = list_folder(music_cfg.get("library_folder", "music-library/"))
            for entry in entries:
                if entry.get("IsDirectory"):
                    continue
                remote = f"{music_cfg.get('library_folder', 'music-library/').rstrip('/')}/{entry['ObjectName']}"
                download_file(remote, os.path.join(args.out_dir, entry["ObjectName"]))
            print(f"Downloaded {len(entries)} music library file(s)")
        except requests.HTTPError as e:
            print(f"WARNING: could not list music library folder: {e}")

    # Speaker clips
    for speaker in manifest.get("speakers", []):
        clip = speaker.get("clip")
        if clip:
            remote = remote_path_from_url_or_key(clip)
            download_file(remote, os.path.join(args.out_dir, os.path.basename(clip)))
            print(f"Downloaded speaker clip: {clip}")

    # B-roll folder (everything in media_folder)
    media_folder = manifest.get("media_folder")
    if media_folder:
        entries = list_folder(media_folder)
        for entry in entries:
            if entry.get("IsDirectory"):
                continue
            remote = f"{media_folder.rstrip('/')}/{entry['ObjectName']}"
            download_file(remote, os.path.join(args.out_dir, entry["ObjectName"]))
        print(f"Downloaded {len(entries)} b-roll file(s) from {media_folder}")


if __name__ == "__main__":
    main()
