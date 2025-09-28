import os
from huggingface_hub import snapshot_download
from const import BASE_DIR, MODELS


def download_model(model_name: str, target_dir: str):
    print(f"Скачиваем модель {model_name} в {target_dir}")
    snapshot_download(
        repo_id=model_name,
        local_dir=target_dir,
        local_dir_use_symlinks=False
    )
    print(f" Модель '{model_name}' скачана в: {target_dir}")


def main():
    os.makedirs(BASE_DIR, exist_ok=True)
    print(f"Папка '{BASE_DIR}' готова.")

    for folder, repo_id in MODELS.items():
        target_path = os.path.join(BASE_DIR, folder)
        download_model(repo_id, target_path)


if __name__ == "__main__":
    main()