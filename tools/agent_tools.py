from langchain_core.tools import tool
from modules.generate_background import BackgroundGeneration
from modules.generate_product_characteristics import CharacteristicsGenerate
from schemas.schema import ImageGenInput, ImageGenOutput, OutputLLM, OutputHeaders

bg_generator = BackgroundGeneration()
char_generator = CharacteristicsGenerate()


@tool("background_generation", args_schema=ImageGenInput)
def generate_images_tool(input_data: ImageGenInput) -> ImageGenOutput:
    """
    Генерирует фон для изображений товара
    """
    images = bg_generator.get_images(input_data.dict())
    return ImageGenOutput(job_id=input_data.job_id, generated_images=images)


@tool("generate_characteristics", args_schema=ImageGenInput)
def generate_characteristics_tool(input_data: ImageGenInput) -> OutputLLM:
    """
    Генерирует title, subtitle и utp на основе характеристик товара
    """
    return char_generator.get_characteristics(input_data.dict())


@tool("generate_headers")
def generate_headers_tool(title: str) -> OutputHeaders:
    """
    Генерирует 1–4 варианта заголовков на основе основного title
    """
    return char_generator.get_headers(title)

