import time
import uuid
from threading import Lock
from typing import Any, Dict, List, Optional, Union

import httpx
import uvicorn
from fastapi import BackgroundTasks, Body, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, HttpUrl, ValidationError

from pipeline import ProductCardGeneration
from schemas.schema import ProductCardResponse, ProductData

app = FastAPI(title="Product Card Service", version="1.0.0")


class StatusStore:
    def __init__(self):
        self._data: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()

    def set(self, job_id: str, **kwargs):
        with self._lock:
            self._data.setdefault(job_id, {})
            self._data[job_id].update(kwargs)

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._data.get(job_id)

    def all(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return dict(self._data)


store = StatusStore()


class ProductDataExtended(ProductData):
    callback_url: Optional[HttpUrl] = Field(default=None, description="URL для обратного вызова (опционально)")


class EnqueueItem(BaseModel):
    job_id: str
    status: str = "queued"


class EnqueueResponse(BaseModel):
    jobs: List[EnqueueItem]


def post_callback(url: Optional[str], payload: Dict[str, Any]) -> None:
    if not url:
        return
    try:
        with httpx.Client(timeout=20) as client:
            client.post(str(url), json=payload)
    except Exception:
        # без падений, это вспомогательный канал
        pass


# ---------------------- КРИТЕРИИ ПОЛНОТЫ ----------------------
def _is_empty_text(v: Any, min_len: int = 5) -> bool:
    if not isinstance(v, str):
        return True
    s = v.strip()
    return len(s) < min_len

def _is_empty_image(img: Dict[str, Any]) -> bool:
    if not isinstance(img, dict):
        return True
    b64 = img.get("image_base64") or ""
    return not isinstance(b64, str) or len(b64.strip()) == 0

def needs_regeneration(payload: Dict[str, Any]) -> (bool, List[str]):
    """Проверяем, стоит ли отправить на перегенерацию и почему."""
    reasons: List[str] = []

    # текстовый блок
    text_block = payload.get("text", {})
    if not isinstance(text_block, dict):
        reasons.append("text block is not a dict")
    else:
        empty_keys = [k for k, v in text_block.items() if _is_empty_text(v)]
        if empty_keys:
            reasons.append(f"empty text fields: {', '.join(sorted(empty_keys))}")

    # изображения
    imgs = payload.get("generated_images", [])
    if not isinstance(imgs, list) or len(imgs) < 4:
        reasons.append("generated_images list missing or < 4 items")
    else:
        empty_img_pos = [str(img.get("image_position", "?")) for img in imgs if _is_empty_image(img)]
        if empty_img_pos:
            reasons.append(f"empty images at positions: {', '.join(empty_img_pos)}")

    return (len(reasons) > 0), reasons
# --------------------------------------------------------------


def run_job(job_id: str, input_json: Dict[str, Any], max_attempts: int = 3, retry_delay: float = 0.75):
    """
    Запускает пайплайн с авто-повторами.
    1) генерируем payload,
    2) валидируем схему,
    3) проверяем на пустые поля -> если есть, уходим в повтор.
    """
    callback_url = input_json.get("callback_url")
    errors: List[Dict[str, Any]] = []

    # Не передаём callback_url внутрь пайплайна
    pipeline_input = {k: v for k, v in input_json.items() if k != "callback_url"}

    for attempt in range(1, max_attempts + 1):
        store.set(job_id, status="running", progress=0, attempt=attempt, errors=errors)
        post_callback(callback_url, {"job_id": job_id, "status": "running", "progress": 0, "attempt": attempt})

        payload = None
        try:
            generator = ProductCardGeneration()  # независимый экземпляр на каждую попытку
            payload = generator.run(input_data=pipeline_input)

            # Сначала валидируем pydantic-схемой
            try:
                result = ProductCardResponse(**payload).model_dump(mode="json")
            except ValidationError as ve:
                err = {
                    "code": "RESPONSE_VALIDATION_ERROR",
                    "message": str(ve),
                    "errors": ve.errors(),
                    "payload": payload,
                    "attempt": attempt,
                }
                errors.append(err)
                if attempt < max_attempts:
                    store.set(job_id, status="retrying", progress=0, attempt=attempt, errors=errors, last_error=err)
                    post_callback(callback_url, {
                        "job_id": job_id, "status": "retrying", "progress": 0, "attempt": attempt, "error": err
                    })
                    time.sleep(retry_delay * attempt)
                    continue
                # на последней попытке — фиксируем fail
                store.set(job_id, status="failed", progress=0, errors=errors, error=err)
                post_callback(callback_url, {"job_id": job_id, "status": "failed", "error": err, "errors": errors})
                return

            # Дополнительная проверка на пустые поля
            must_regen, regen_reasons = needs_regeneration(result)
            if must_regen:
                err = {
                    "code": "INCOMPLETE_FIELDS",
                    "message": "payload contains empty fields",
                    "reasons": regen_reasons,
                    "attempt": attempt,
                    "payload_preview": {
                        "text_keys": list(result.get("text", {}).keys()),
                        "images": len(result.get("generated_images", [])),
                    },
                }
                errors.append(err)
                if attempt < max_attempts:
                    store.set(job_id, status="retrying", progress=0, attempt=attempt, errors=errors, last_error=err)
                    post_callback(callback_url, {
                        "job_id": job_id,
                        "status": "retrying",
                        "progress": 0,
                        "attempt": attempt,
                        "error": err,
                    })
                    time.sleep(retry_delay * attempt)
                    continue
                # если попытки исчерпаны — считаем неуспехом
                store.set(job_id, status="failed", progress=0, errors=errors, error=err)
                post_callback(callback_url, {"job_id": job_id, "status": "failed", "error": err, "errors": errors})
                return

            # успех
            store.set(job_id, status="succeeded", progress=100, result=result, attempt=attempt, errors=errors)
            post_callback(callback_url, {"job_id": job_id, "status": "succeeded", "attempt": attempt, **result})
            return

        except Exception as e:
            err = {
                "code": "INFERENCE_ERROR",
                "message": str(e),
                "input": pipeline_input,
                "partial_payload": payload,
                "attempt": attempt,
            }
            errors.append(err)

            if attempt < max_attempts:
                store.set(job_id, status="retrying", progress=0, attempt=attempt, errors=errors, last_error=err)
                post_callback(callback_url, {
                    "job_id": job_id, "status": "retrying", "progress": 0, "attempt": attempt, "error": err
                })
                time.sleep(retry_delay * attempt)
                continue

            store.set(job_id, status="failed", progress=0, errors=errors, error=err)
            post_callback(callback_url, {"job_id": job_id, "status": "failed", "error": err, "errors": errors})
            return


@app.post("/get_product_card", status_code=202, response_model=EnqueueResponse)
def get_product_card(
    background_tasks: BackgroundTasks,
    data: Union[ProductDataExtended, List[ProductDataExtended]] = Body(...),
):
    items: List[ProductDataExtended] = data if isinstance(data, list) else [data]
    jobs: List[EnqueueItem] = []
    for item in items:
        job_id = item.job_id or f"job_{uuid.uuid4().hex[:8]}"
        input_json = item.model_dump(mode="json", exclude_none=True)
        store.set(job_id, status="queued", progress=0, input=input_json)
        background_tasks.add_task(run_job, job_id, input_json)
        jobs.append(EnqueueItem(job_id=job_id))
    return JSONResponse(status_code=202, content=EnqueueResponse(jobs=jobs).model_dump())


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    st = store.get(job_id)
    if not st:
        raise HTTPException(status_code=404, detail="job not found")
    return st


@app.get("/jobs")
def get_all_jobs():
    data = store.all()
    return [{"job_id": jid, "status": info.get("status")} for jid, info in data.items()]


@app.get("/results")
def get_all_results():
    data = store.all()
    out: List[Dict[str, Any]] = []
    for jid, info in data.items():
        res = info.get("result")
        if not res:
            continue
        out.append({
            "job_id": jid,
            "text": res.get("text", {}),
            "generated_images": res.get("generated_images", []),
        })
    return out


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=2031)
