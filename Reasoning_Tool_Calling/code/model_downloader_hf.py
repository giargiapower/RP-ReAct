#!/usr/bin/env python3
"""
program.py – log in to Hugging Face and download Qwen/Qwen3-14B

Usage
-----
$ export HF_TOKEN="hf_************************************"
$ python program.py                   # downloads to ./Qwen3-14B

or

$ python program.py --token hf_******** --dest /path/to/models
"""

from huggingface_hub import login, snapshot_download
import argparse
import os
import sys

def main() -> None:
    parser = argparse.ArgumentParser(description="Download google/gemma-3-12b-it from the HF Hub")
    parser.add_argument("--token", help="Hugging Face access token (defaults to $HF_TOKEN)")
    parser.add_argument("--dest", default="gemma-3-12b-it",
                        help="Destination folder (default: google/gemma-3-12b-it)")
    parser.add_argument("--revision", default=None,
                        help="Branch, tag, or commit to download (default: latest main)")
    args = parser.parse_args()

    token = args.token or os.getenv("HF_TOKEN")
    if not token:
        sys.exit("❌  No Hugging Face token provided. Use --token or set $HF_TOKEN.")

    # 1 – authenticate (stores token in ~/.cache/huggingface/token)
    login(token)

    # 2 – download the repository as a local snapshot
    print("⏬  Downloading openai/gpt-oss-20b … this may take a while ⏬")
    snapshot_download(
        repo_id="google/gemma-3-12b-it",
        revision=args.revision,
        local_dir=args.dest,
        local_dir_use_symlinks=False,   # makes a full, independent copy
        resume_download=True            # resumes if interrupted
    )
    print(f"✅  Finished. Model files are in: {os.path.abspath(args.dest)}")

if __name__ == "__main__":
    main()