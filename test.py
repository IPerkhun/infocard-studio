#%%
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

MODEL_ID = "models/qwen_image_edit"

# -----------------------------
# Функция: паддинг под нужный размер
# -----------------------------
def pad_to_size(img: Image.Image, target_w=900, target_h=1200):
    img = img.copy()
    w, h = img.size

    # Вписываем изображение без искажения (аналог contain-fit)
    ratio = min(target_w / w, target_h / h)
    new_w = int(w * ratio)
    new_h = int(h * ratio)

    img = img.resize((new_w, new_h), Image.LANCZOS)

    # создаём целевой холст 900×1200 (можно менять цвет)
    background = Image.new("RGB", (target_w, target_h), (255, 255, 255))

    # центрируем изображение
    offset = ((target_w - new_w) // 2, (target_h - new_h) // 2)
    background.paste(img, offset)

    return background


pipe = QwenImageEditPlusPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

#%%
import os

OUTPUT_DIR = 'edits_images'
INPUT_DIR = 'edits_image'

os.makedirs(OUTPUT_DIR, exist_ok=True)

for filename in os.listdir(INPUT_DIR):
    if not filename.lower().endswith(('png', 'jpg', 'jpeg')):
        continue

    full_path = os.path.join(INPUT_DIR, filename)
    print(f'Обработка {filename}')

    image = Image.open(full_path).convert('RGB')
    image = pad_to_size(image, 900, 1200)


# image = Image.open("images/2943977489_4.png").convert("RGB")

# # ВАЖНО: правильная подготовка → остаётся ровное 900×1200 без искажения
# image = pad_to_size(image, 900, 1200)

    prompt = (
    """Replace the background with a modern kitchen with a Christmas theme. Keep the product unchanged, preserving its original
shape and details. Remove all existing background elements."""
    )

    negative_prompt = (
        """No people, no faces, no hands, no text, no logos, no labels, no distortions, no reflections, 
        no shadows on the product, no color changes to the product, no extra objects, no artifacts, no blur, no low quality, 
        no watermarks, no decorations covering the product."""
    )


    inputs = {
        "image": [image],                    
        "prompt": [prompt],                  
        "negative_prompt": [negative_prompt],
        "num_inference_steps": 35,
        "guidance_scale": 1.0,
        "true_cfg_scale": 4.0,
    }

    with torch.inference_mode():
        result = pipe(**inputs)

    out_image = result.images[0]


    base, _ = os.path.splitext(filename)
    out_name = f"{base}_edit.png"
    out_path = os.path.join(OUTPUT_DIR, out_name)

    out_image.save(out_path)

    # out_image.save("output_qwen_2509_4bit_test8.png")
    # print("Сохранено: output_qwen_2509_4bit_test.png")

print(f"Сохранено в папку {OUTPUT_DIR}")

#%%
'''Для 1 изображения'''
import os

OUTPUT_DIR = 'edits_image'
INPUT_DIR = 'edits_image'

os.makedirs(OUTPUT_DIR, exist_ok=True)

for filename in os.listdir(INPUT_DIR):
    if not filename.lower().endswith(('png', 'jpg', 'jpeg')):
        continue

    full_path = os.path.join(INPUT_DIR, filename)
    print(f'Обработка {filename}')

    image = Image.open(full_path).convert('RGB')
    image = pad_to_size(image, 900, 1200)


# image = Image.open("images/2943977489_4.png").convert("RGB")

# # ВАЖНО: правильная подготовка → остаётся ровное 900×1200 без искажения
# image = pad_to_size(image, 900, 1200)

    prompt = (
    """Replace the background with a festive cream-gold living room that still feels connected to a warm Christmas kitchen style: decorated tree, garlands, soft bokeh, and subtle kitchen elements in the background. Keep pot unchanged. No text. Bright lighting"""
    )

    negative_prompt = (
        """No people, no faces, no hands, no text, no logos, no labels, no distortions, no reflections, 
        no shadows on the product, no color changes to the product, no extra objects, no artifacts, no blur, no low quality, 
        no watermarks, no decorations covering the product."""
    )


    inputs = {
        "image": [image],                    
        "prompt": [prompt],                  
        "negative_prompt": [negative_prompt],
        "num_inference_steps": 35,
        "guidance_scale": 1.0,
        "true_cfg_scale": 4.0,
    }

    with torch.inference_mode():
        result = pipe(**inputs)

    out_image = result.images[0]


    base, _ = os.path.splitext(filename)
    out_name = f"{base}_edit.png"
    out_path = os.path.join(OUTPUT_DIR, out_name)

    out_image.save(out_path)

    # out_image.save("output_qwen_2509_4bit_test8.png")
    # print("Сохранено: output_qwen_2509_4bit_test.png")

print(f"Сохранено в папку {OUTPUT_DIR}")
#%%

