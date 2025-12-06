import os

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

load_dotenv()

MODELS = {
    "qwen_llm": "Qwen/Qwen3-4B-Instruct-2507-FP8",           
    "qwen_image_edit": "ovedrive/Qwen-Image-Edit-2509-4bit",      
    "qwen_detect_product": "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
    "ultra": "Qwen/Qwen-Image-Edit"
}


def download_model(repo_id: str, target_dir: str):
    print(f"🚀 Скачиваем модель {repo_id} в {target_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=target_dir,
        local_dir_use_symlinks=False,
        ignore_patterns=["*.msgpack"],
        token=os.getenv("HUGGINGFACE_HUB_TOKEN")
    )
    print(f"✅ Модель '{repo_id}' готова в: {target_dir}")


def main():
    os.makedirs("models", exist_ok=True)
    print(f"📁 Папка для моделей: models")

    for folder, repo_id in MODELS.items():
        target_path = os.path.join("models", folder)

        if os.path.exists(target_path) and os.listdir(target_path):
            print(f"ℹ️  Модель '{repo_id}' уже скачана в {target_path}. Пропускаем.")
            continue

        os.makedirs(target_path, exist_ok=True)
        download_model(repo_id, target_path)


if __name__ == "__main__":
    main()
