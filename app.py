import uvicorn
from fastapi import FastAPI, HTTPException

from modules.generate_background import BackgroundGeneration
from modules.generate_product_characteristics import CharacteristicsGenerate
from schemas.schema import OutputLLM, ProductData, ResponseFormat

app = FastAPI()
bg_generator = BackgroundGeneration()
char_gen = CharacteristicsGenerate()


@app.post("/generate", response_model=ResponseFormat)
def generate_images(data: ProductData) -> ResponseFormat:
    try:
        encoded_images = bg_generator.get_images(data.model_dump())
        return ResponseFormat(job_id=data.job_id, generated_images=encoded_images)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/generate-characteristics", response_model=OutputLLM)
def generate_characteristics(data: ProductData) -> OutputLLM:
    try:
        result = char_gen.get_characteristics(data.model_dump())
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=2031)