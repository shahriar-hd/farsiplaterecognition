"""
Script to create repo and upload all prepared models in .uploads/ to Hugging Face Hub.
Repo format: {username}/{repo_name}
Uses token from .env (HF_TOKEN) or environment variable.
"""

import os
import argparse
from huggingface_hub import HfApi, create_repo

def load_env_file():
    if os.path.exists(".env"):
        with open(".env") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip().strip("'\"")

def upload():
    parser = argparse.ArgumentParser(description="Upload Iranian LPR models to Hugging Face")
    parser.add_argument("--repo-id", type=str, default="iranian-car-plate-recognition-yolov8", help="Hugging Face repo name")
    parser.add_argument("--private", action="store_true", help="Set repository to private")
    args = parser.parse_args()

    load_env_file()
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN not found in environment or .env file!")

    api = HfApi(token=hf_token)
    user = api.whoami()
    username = user["name"]

    full_repo_id = f"{username}/{args.repo_id}" if "/" not in args.repo_id else args.repo_id
    print(f"Target repository: https://huggingface.co/{full_repo_id}")

    try:
        create_repo(repo_id=full_repo_id, token=hf_token, repo_type="model", private=args.private, exist_ok=True)
        print("✅ Repository verified/created.")
    except Exception as e:
        print(f"Note: {e}")

    print("\nUploading folder '.uploads/' to Hugging Face...")
    api.upload_folder(
        folder_path=".uploads",
        repo_id=full_repo_id,
        repo_type="model",
        commit_message="Add Iranian LPR YOLOv8 weights (safetensors, onnx, ncnn, pt)",
        token=hf_token
    )
    print(f"\n🎉 Successfully uploaded models to: https://huggingface.co/{full_repo_id}")

if __name__ == "__main__":
    upload()
