import logging

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

from typing import Dict, Type

from langgraph.graph import END, StateGraph
from pydantic import BaseModel


from models_init.llm_manager import LLMManager
from prompts.prompt import (
    PROMPT_DETECT_LMS,
    PROMPT_DETECT_OBJECT,
    PROMPT_GENERATE_ADDITIONAL_ANGLES,
    PROMPT_GENERATE_CHARACTERISTICS,
    PROMPT_GENERATE_DESCRIPTION,
    PROMPT_GENERATE_HEADERS,
    PROMPT_GENERATE_IMAGE_L,
    PROMPT_GENERATE_IMAGE_M,
    PROMPT_GENERATE_IMAGE_S,
    PROMPT_GENERATE_SPECS,
    PROMPT_VALIDATE_IMAGE_QUALITY,
    PROMPT_VALIDATE_IMAGE_VL,
)
from schemas.schema import (
    AgentState,
    DescriptionOutput,
    DetectProductOutput,
    HeadersOutput,
    ImageGenOutput,
    OutputLLM,
    SpecsOutput,
    ImageQualityOutput,
)
from tools.agent_tools import (
    detect_product_tool,
    generate_images_tool,
    inspect_generated_image_tool,
)

_LABEL_PROMPTS = {
    "L": PROMPT_GENERATE_IMAGE_L,
    "M": PROMPT_GENERATE_IMAGE_M,
    "S": PROMPT_GENERATE_IMAGE_S,
}


class ProductCardGeneration:
    def __init__(self) -> None:
        self.llm = LLMManager().llm
        self.graph = self._build_graph()

    def _so(self, schema: Type[BaseModel]):
        return self.llm.with_structured_output(schema=schema)

    def _normalize_photos(self, state: AgentState) -> AgentState:
        try:
            photos = state.input_data.get("product_photos") or []
            logger.info("normalize_photos: received %d photos", len(photos))

            norm: list[dict] = []
            seen = set()

            for idx, it in enumerate(photos):
                if isinstance(it, str):
                    url = it
                    pos = idx + 1
                else:
                    url = it.get("image_url") or it.get("url")
                    pos = it.get("image_position") or (idx + 1)

                if not url or url in seen:
                    continue

                seen.add(url)
                norm.append(
                    {
                        "image_url": url,
                        "image_position": int(pos),
                    }
                )

            norm.sort(key=lambda x: x["image_position"])
            norm = norm[:4]

            for i, n in enumerate(norm, start=1):
                n["image_position"] = i

            state.input_data["product_photos"] = norm
            state.result["need_variations"] = max(0, 4 - len(norm))
            return state

        except Exception:
            state.result["need_variations"] = 0
            state.input_data["product_photos"] = []
            return state

    def _augment_photos(self, state: AgentState) -> AgentState:
        need = state.result.get("need_variations", 0)
        photos = state.input_data.get("product_photos") or []

        logger.info(
            "augment_photos: need_variations=%d, current_photos=%d",
            need,
            len(photos),
        )

        if need <= 0 or not photos:
            state.result["augment_prompt"] = ""
            return state

        product_name = state.input_data.get("product_name", "товар")
        state.result["augment_prompt"] = PROMPT_GENERATE_ADDITIONAL_ANGLES.format(
            product_name=product_name
        )

        last_pos = photos[-1]["image_position"]
        augmented = []
        for i in range(need):
            clone = dict(photos[-1])
            clone["image_position"] = last_pos + i + 1
            augmented.append(clone)

        logger.info(
            "augment_photos: augmented %d photos, total=%d, added_positions=%s",
            len(augmented),
            len(state.input_data["product_photos"]),
            [p["image_position"] for p in augmented],
        )

        state.input_data["product_photos"] = photos + augmented
        return state

    def _detect_label(self, state: AgentState) -> AgentState:
        photos = state.input_data.get("product_photos") or []

        first = photos[0]
        image_url = first.get("image_url") if isinstance(first, dict) else first

        raw_vl_text = detect_product_tool.invoke(
            {"image_url": image_url, "question": PROMPT_DETECT_OBJECT}
        )

        prompt = PROMPT_DETECT_LMS.format(
            product_name=state.input_data.get("product_name", ""),
            product_properties=state.input_data.get("product_properties", ""),
            vision_raw=raw_vl_text,
        )

        runnable = self._so(DetectProductOutput)
        det: DetectProductOutput = runnable.invoke(prompt)

        state.result["label"] = det.label

        logger.info("detect_label: detected label=%s", det.label)

        return state

    def _gen_background(self, state: AgentState) -> AgentState:
        label = state.result.get("label") or "M"
        if label == "NONE":
            state.result["generated_images"] = []
            return state

        base_prompt = _LABEL_PROMPTS.get(label, PROMPT_GENERATE_IMAGE_M)
        product_name = state.input_data.get("product_name", "item")
        extra = state.result.get("augment_prompt") or ""
        prompt_text = f"{base_prompt.format(product_name=product_name)} {extra}".strip()

        photos = state.input_data.get("product_photos") or []
        photos_with_url = [
            p for p in photos if isinstance(p, dict) and p.get("image_url")
        ]
        urls = [p["image_url"] for p in photos_with_url]

        if not urls:
            state.result["generated_images"] = []
            return state

        img_out: ImageGenOutput = generate_images_tool.invoke(
            {"product_photos": urls, "prompt": prompt_text}
        )

        generated = [
            {
                "image_position": meta.get("image_position"),
                "image_base64": b64 or "",
            }
            for meta, b64 in zip(photos_with_url, img_out.generated_images)
        ]

        state.result["generated_images"] = generated

        logger.info(
            "gen_background: generated_images_count=%d, positions=%s",
            len(generated),
            [g["image_position"] for g in generated],
        )

        return state

    def _validate_images(self, state: AgentState) -> AgentState:
        generated = state.result.get("generated_images") or []
        if not generated:
            return state

        product_name = state.input_data.get("product_name", "товар")

        label = state.result.get("label") or "M"
        base_prompt = _LABEL_PROMPTS.get(label, PROMPT_GENERATE_IMAGE_M)
        extra = state.result.get("augment_prompt") or ""
        prompt_text = f"{base_prompt.format(product_name=product_name)} {extra}".strip()

        photos = state.input_data.get("product_photos") or []
        photos_with_url = [
            p for p in photos if isinstance(p, dict) and p.get("image_url")
        ]

        if not photos_with_url:
            return state

        n = min(len(generated), len(photos_with_url))
        generated = generated[:n]
        photos_with_url = photos_with_url[:n]

        new_generated = list(generated)

        attempts = state.result.setdefault("regen_attempts", {})

        logger.info(
            "validate_images: start validation for %d images, label=%s, prompt=%r",
            n,
            label,
            prompt_text,
        )

        for idx in range(n):
            img_meta = generated[idx]
            img_b64 = img_meta.get("image_base64") or ""
            pos = img_meta.get("image_position")

            if not img_b64:
                continue

            if attempts.get(pos, 0) >= 3:
                continue

            raw_vl_text = inspect_generated_image_tool.invoke(
                {"image_base64": img_b64, "question": PROMPT_VALIDATE_IMAGE_VL.strip()}
            )

            logger.debug(
                "validate_images: position=%s raw_vl_text=%s",
                pos,
                raw_vl_text,
            )

            prompt = PROMPT_VALIDATE_IMAGE_QUALITY.format(
                product_name=product_name,
                vision_raw=raw_vl_text,
            )

            runnable = self._so(ImageQualityOutput)
            quality: ImageQualityOutput = runnable.invoke(prompt)

            logger.info(
                "validate_images: position=%s status=%s reason=%s attempts=%d",
                pos,
                quality.status,
                quality.reason,
                attempts.get(pos, 0),
            )

            if quality.status == "OK":
                continue

            src_url = photos_with_url[idx]["image_url"]

            try:
                regen_out: ImageGenOutput = generate_images_tool.invoke(
                    {"product_photos": [src_url], "prompt": prompt_text}
                )

                attempts[pos] = attempts.get(pos, 0) + 1

                if regen_out.generated_images:
                    new_generated[idx] = {
                        "image_position": pos,
                        "image_base64": regen_out.generated_images[0] or "",
                    }

            except Exception:
                attempts[pos] = attempts.get(pos, 0) + 1
                continue

        state.result["generated_images"] = new_generated

        logger.info(
            "validate_images: finished. regen_attempts=%s",
            attempts,
        )

        return state

    def _gen_characteristics_giga(self, state: AgentState) -> AgentState:
        prompt = PROMPT_GENERATE_CHARACTERISTICS.format(
            product_name=state.input_data.get("product_name", ""),
            product_properties=state.input_data.get("product_properties") or "",
        )

        runnable = self._so(OutputLLM)
        out: OutputLLM = runnable.invoke(prompt)

        state.result["characteristics"] = out.model_dump()
        return state

    def _gen_headers(self, state: AgentState) -> AgentState:
        title = (state.result.get("characteristics") or {}).get("title", "")
        prompt = PROMPT_GENERATE_HEADERS.format(title=title)

        runnable = self._so(HeadersOutput)
        out: HeadersOutput = runnable.invoke(prompt)

        headers = (out.headers or [])[:4]
        headers += [""] * (4 - len(headers))
        state.result["headers"] = headers

        logger.info("gen_headers: headers=%s", headers)

        return state

    def _gen_specs(self, state: AgentState) -> AgentState:
        prompt = PROMPT_GENERATE_SPECS.format(
            product_name=state.input_data.get("product_name", ""),
            product_properties=state.input_data.get("product_properties") or "",
        )

        runnable = self._so(SpecsOutput)
        out: SpecsOutput = runnable.invoke(prompt)

        state.result["specs_text"] = out.text or ""

        logger.info(
            "gen_specs: specs_text_preview=%r",
            (
                (state.result["specs_text"][:120] + "...")
                if state.result["specs_text"]
                else ""
            ),
        )

        return state

    def _gen_description(self, state: AgentState) -> AgentState:
        ch = state.result.get("characteristics") or {}
        utp_lines = "\n".join(
            f"УТП {u['number']}: {u['text']}"
            for u in ch.get("utp", [])
            if isinstance(u, dict) and "number" in u and "text" in u
        )

        print(f"Залупа {utp_lines}")

        prompt = PROMPT_GENERATE_DESCRIPTION.format(
            title=ch.get("title", ""),
            subtitle=ch.get("subtitle", ""),
            utp_lines=utp_lines,
            specs_text=state.result.get("specs_text", ""),
        )

        runnable = self._so(DescriptionOutput)
        out: DescriptionOutput = runnable.invoke(prompt)
        desc_text = (out.text or "").strip()

        state.result["description_text"] = desc_text

        logger.info(
            "gen_description: description_preview=%r",
            (desc_text[:120] + "...") if desc_text else "",
        )

        return state

    def _build_graph(self):
        g = StateGraph(AgentState)

        g.add_node("normalize_photos", self._normalize_photos)
        g.add_node("augment_photos", self._augment_photos)
        g.add_node("detect_label", self._detect_label)
        g.add_node("gen_background", self._gen_background)
        g.add_node("validate_images", self._validate_images)
        g.add_node("gen_characteristics_giga", self._gen_characteristics_giga)
        g.add_node("gen_headers", self._gen_headers)
        g.add_node("gen_specs", self._gen_specs)
        g.add_node("gen_description", self._gen_description)

        g.set_entry_point("normalize_photos")

        g.add_edge("normalize_photos", "augment_photos")
        g.add_edge("augment_photos", "detect_label")
        g.add_edge("detect_label", "gen_background")
        g.add_edge("gen_background", "validate_images")
        g.add_edge("validate_images", "gen_characteristics_giga")
        g.add_edge("gen_characteristics_giga", "gen_headers")
        g.add_edge("gen_headers", "gen_specs")
        g.add_edge("gen_specs", "gen_description")
        g.add_edge("gen_description", END)

        return g.compile()

    def _build_payload(self, state: AgentState) -> dict:
        job_id = state.input_data.get("job_id")

        raw_label = state.result.get("label")
        label = "M" if raw_label == "NONE" else raw_label
        prefix = label.lower()

        ch = state.result.get("characteristics") or {}
        utp_list = ch.get("utp", []) or []

        def utp(i: int) -> str:
            idx = i - 1
            return utp_list[idx]["text"] if 0 <= idx < len(utp_list) else ""

        text_block = {
            f"{prefix}-1-slide-title": ch.get("title", "") or "",
            f"{prefix}-1-slide-subtitle": ch.get("subtitle", "") or "",
            f"{prefix}-1-slide-utp-1": utp(1),
            f"{prefix}-1-slide-utp-2": utp(2),
            f"{prefix}-2-slide-utp-3": utp(3),
            f"{prefix}-2-slide-utp-3-continue": ch.get("utp_3_continue", "") or "",
            f"{prefix}-3-slide-utp-4": utp(4),
            f"{prefix}-3-slide-utp-4-continue": ch.get("utp_4_continue", "") or "",
            f"{prefix}-3-slide-utp-5": utp(5),
            f"{prefix}-3-slide-utp-5-continue": ch.get("utp_5_continue", "") or "",
            f"{prefix}-4-slide-utp-6": utp(6),
            f"{prefix}-4-slide-utp-7": utp(7),
            f"{prefix}-4-slide-utp-8": utp(8),
            f"{prefix}-specs": state.result.get("specs_text", "") or "",
            f"{prefix}-description": state.result.get("description_text", "") or "",
        }

        images = state.result.get("generated_images") or []

        imgs = []
        for meta in images[:4]:
            imgs.append(
                {
                    "image_position": meta.get("image_position"),
                    "image_base64": meta.get("image_base64", "") or "",
                }
            )

        while len(imgs) < 4:
            imgs.append(
                {
                    "image_position": len(imgs) + 1,
                    "image_base64": "",
                }
            )

        payload = {
            "job_id": job_id,
            "text": text_block,
            "generated_images": imgs,
        }

        return payload

    def run_state(self, input_data: Dict) -> AgentState:
        init = AgentState(input_data=input_data)
        out = self.graph.invoke(init)
        return out if isinstance(out, AgentState) else AgentState(**out)

    def run(self, input_data: Dict) -> dict:
        state = self.run_state(input_data)
        return self._build_payload(state=state)
