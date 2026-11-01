"""
Upload Gradio demo directly as a Hugging Face Space.
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

def upload_space():
    parser = argparse.ArgumentParser(description="Upload Iranian LPR Demo to Hugging Face Spaces")
    parser.add_argument("--space-id", type=str, default="iranian-car-plate-recognition-demo", help="HF Space repo name")
    parser.add_argument("--private", action="store_true", help="Set space to private")
    args = parser.parse_args()

    load_env_file()
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN not found in environment or .env file!")

    api = HfApi(token=hf_token)
    user = api.whoami()
    username = user["name"]

    full_space_id = f"{username}/{args.space_id}" if "/" not in args.space_id else args.space_id
    print(f"Target Hugging Face Space: https://huggingface.co/spaces/{full_space_id}")

    try:
        create_repo(
            repo_id=full_space_id,
            token=hf_token,
            repo_type="space",
            space_sdk="gradio",
            private=args.private,
            exist_ok=True
        )
        print("✅ Space repository verified/created.")
    except Exception as e:
        print(f"Note: {e}")

    print("\nUploading folder 'hf_space/' to Hugging Face Space...")
    api.upload_folder(
        folder_path="hf_space",
        repo_id=full_space_id,
        repo_type="space",
        commit_message="Initial deploy for Iranian LPR YOLOv8 demo",
        token=hf_token
    )
    print(f"\n🎉 Successfully deployed Hugging Face Space: https://huggingface.co/spaces/{full_space_id}")

if __name__ == "__main__":
    upload_space()
