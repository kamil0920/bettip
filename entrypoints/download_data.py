# entrypoints/download_data.py
import os
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

load_dotenv()


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
            # Only download raw and preprocessed data
            # Features (03-features) should be regenerated fresh each run
            allow_patterns=["data/01-raw/**", "data/02-preprocessed/**", "data/odds-cache/**"],
            ignore_patterns=[".gitattributes", "data/03-features/**"],
            token=token
        )
        print("✅ Data synced locally (raw + preprocessed only, features will be regenerated)")
    except Exception as e:
        print(f"❌ Download failed: {e}")
        exit(1)


if __name__ == "__main__":
    download_data()
