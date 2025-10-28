from huggingface_hub import HfApi

if __name__ == "__main__":
    api = HfApi()
    api.upload_folder(
        folder_path="/fs/scratch/PAS2836/yusenpeng_checkpoint/BioCLIP",  # local directory with your model files
        repo_id="YusenPeng/DRIP_checkpoints",
        repo_type="model",
        commit_message="Upload all DRIP model checkpoints"
    )