import os
from huggingface_hub import snapshot_download

from const import ROOT_PATH_MODELS

MODELS = {
    "qwen_llm": "Qwen/Qwen3-4B-Instruct-2507-FP8",
    "qwen_image_edit": "ovedrive/qwen-image-edit-4bit"
}

def download_model(model_name: str, target_dir: str):
    print(f"Скачиваем модель {model_name} в {target_dir}")
    snapshot_download(
        repo_id=model_name,
        local_dir=target_dir,
        local_dir_use_symlinks=False
    )
    print(f"✅ Модель '{model_name}' скачана в: {target_dir}")

def main():
    os.makedirs(ROOT_PATH_MODELS, exist_ok=True)
    print(f"📁 Папка '{ROOT_PATH_MODELS}' готова.")

    for folder, repo_id in MODELS.items():
        target_path = os.path.join(ROOT_PATH_MODELS, folder)
        os.makedirs(target_path, exist_ok=True)  
        download_model(repo_id, target_path)

if __name__ == "__main__":
    main()
