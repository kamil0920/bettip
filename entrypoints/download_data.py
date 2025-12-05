# entrypoints/download_data.py
import os
from huggingface_hub import snapshot_download


def download_data():
    repo_id = "czlowiekZplanety/bettip-data"
    token = os.getenv("HF_TOKEN")

    if not token:
        print("⚠️ WARNING: HF_TOKEN not found. Download might fail for private repos.")

    print("📥 Downloading data from Hugging Face...")
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=".",
            allow_patterns=["data/**"],
            ignore_patterns=[".gitattributes"],
            token=token
        )
        print("✅ Data synced locally!")
    except Exception as e:
        print(f"❌ Download failed: {e}")
        exit(1)


if __name__ == "__main__":
    download_data()
