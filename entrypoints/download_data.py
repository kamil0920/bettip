from huggingface_hub import snapshot_download
import os

def download_data():
    print("📥 Downloading data from Hugging Face...")
    snapshot_download(
        repo_id="czlowiekZplanety/bettip-data",
        repo_type="dataset",
        local_dir=".",
        allow_patterns=["data/**"],
        ignore_patterns=[".gitattributes"]
    )
    print("✅ Data synced locally!")

if __name__ == "__main__":
    download_data()
