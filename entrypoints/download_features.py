# entrypoints/download_features.py
import os
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

load_dotenv()

def download_features():
    repo_id = "czlowiekZplanety/bettip-data"
    token = os.getenv("HF_TOKEN")

    if not token:
        print("⚠️ WARNING: HF_TOKEN not found. Download might fail for private repos.")

    print("📥 Downloading ONLY features & odds from Hugging Face...")
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=".",
            allow_patterns=["data/03-features/**", "data/odds-cache/**"],
            token=token
        )
        print("✅ Features synced locally (Ready for Training)")
    except Exception as e:
        print(f"❌ Download failed: {e}")
        exit(1)

if __name__ == "__main__":
    download_features()
