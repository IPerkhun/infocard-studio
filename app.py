import time
import uvicorn
import uuid
from threading import Lock
from typing import Any, Dict, List, Optional, Union

import httpx
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
    callback_url: Optional[HttpUrl] = Field(
        default=None, description="URL для обратного вызова (опционально)"
    )


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
        pass


def run_job(job_id: str, input_json: Dict[str, Any]) -> None:
    callback_url = input_json.get("callback_url")
    pipeline_input = {k: v for k, v in input_json.items() if k != "callback_url"}

    store.set(
        job_id,
        status="running",
        progress=0,
        started_at=time.time(),
        input=pipeline_input,
    )
    post_callback(callback_url, {"job_id": job_id, "status": "running"})

    try:
        generator = ProductCardGeneration()
        payload = generator.run(input_data=pipeline_input)
        result = ProductCardResponse(**payload).model_dump(mode="json")

        store.set(
            job_id,
            status="succeeded",
            progress=100,
            finished_at=time.time(),
            result=result,
        )
        post_callback(callback_url, {"job_id": job_id, "status": "succeeded", **result})
        return

    except ValidationError as ve:
        err = {
            "code": "RESPONSE_VALIDATION_ERROR",
            "message": str(ve),
            "errors": ve.errors(),
        }
        store.set(
            job_id, status="failed", progress=0, finished_at=time.time(), error=err
        )
        post_callback(
            callback_url, {"job_id": job_id, "status": "failed", "error": err}
        )
        return

    except Exception as e:
        err = {
            "code": "INFERENCE_ERROR",
            "message": str(e),
        }
        store.set(
            job_id, status="failed", progress=0, finished_at=time.time(), error=err
        )
        post_callback(
            callback_url, {"job_id": job_id, "status": "failed", "error": err}
        )
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
        store.set(
            job_id,
            status="queued",
            progress=0,
            created_at=time.time(),
            input=input_json,
        )
        background_tasks.add_task(run_job, job_id, input_json)
        jobs.append(EnqueueItem(job_id=job_id))
    return JSONResponse(
        status_code=202, content=EnqueueResponse(jobs=jobs).model_dump(mode="json")
    )


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
        out.append(
            {
                "job_id": jid,
                "label": res.get("label"), 
                "text": res.get("text", {}),
                "generated_images": res.get("generated_images", []),
            }
        )
    return out



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=2031)
