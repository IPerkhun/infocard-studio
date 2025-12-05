import logging
import time
from typing import Optional

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
    PROMPT_GENERATE_IMAGE_L_1,
    PROMPT_GENERATE_IMAGE_L_2,
    PROMPT_GENERATE_IMAGE_L_3,
    PROMPT_GENERATE_IMAGE_L_4,
    PROMPT_GENERATE_IMAGE_M_1,
    PROMPT_GENERATE_IMAGE_M_2,
    PROMPT_GENERATE_IMAGE_M_3,
    PROMPT_GENERATE_IMAGE_M_4,
    PROMPT_GENERATE_IMAGE_S_1,
    PROMPT_GENERATE_IMAGE_S_2,
    PROMPT_GENERATE_IMAGE_S_3,
    PROMPT_GENERATE_IMAGE_S_4,
    PROMPT_GENERATE_SPECS,
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
)
from tools.agent_tools import (
    detect_product_tool,
    generate_images_tool,
    inspect_generated_image_tool,
)

_LABEL_PROMPTS = {
    "L": {
        1: PROMPT_GENERATE_IMAGE_L_1,
        2: PROMPT_GENERATE_IMAGE_L_2,
        3: PROMPT_GENERATE_IMAGE_L_3,
        4: PROMPT_GENERATE_IMAGE_L_4,
    },
    "M": {
        1: PROMPT_GENERATE_IMAGE_M_1,
        2: PROMPT_GENERATE_IMAGE_M_2,
        3: PROMPT_GENERATE_IMAGE_M_3,
        4: PROMPT_GENERATE_IMAGE_M_4,
    },
    "S": {
        1: PROMPT_GENERATE_IMAGE_S_1,
        2: PROMPT_GENERATE_IMAGE_S_2,
        3: PROMPT_GENERATE_IMAGE_S_3,
        4: PROMPT_GENERATE_IMAGE_S_4,
    },
}


class ProductCardGeneration:
    def __init__(self) -> None:
        self.llm = LLMManager().llm
        self.graph = self._build_graph()

    def _so(self, schema: Type[BaseModel]):
        return self.llm.with_structured_output(schema=schema)

    def _invoke_llm_with_retry(
        self,
        prompt: str,
        schema: Type[BaseModel],
        max_attempts: int = 3,
        delay: float = 0.5,
        log_prefix: str = "",
    ) -> Optional[BaseModel]:
        """
        Вызывает LLM с structured_output, делает max_attempts попыток.
        Если всё упало — возвращает None, НО не бросает исключение.
        """
        runnable = self._so(schema)
        last_exc: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                logger.info(
                    "%sLLM call attempt %d/%d",
                    f"{log_prefix}: " if log_prefix else "",
                    attempt,
                    max_attempts,
                )
                return runnable.invoke(prompt)
            except Exception as e:
                last_exc = e
                logger.exception(
                    "%sLLM call failed on attempt %d/%d: %s",
                    log_prefix,
                    attempt,
                    max_attempts,
                    e,
                )
                if attempt < max_attempts:
                    time.sleep(delay)

        logger.error(
            "%sLLM call failed after %d attempts: %s",
            log_prefix,
            max_attempts,
            last_exc,
        )
        return None

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
        photos = state.input_data["product_photos"]
        first = photos[0]
        image_url = first["image_url"] if isinstance(first, dict) else first

        vision_raw = detect_product_tool.invoke(
            {"image_url": image_url, "question": PROMPT_DETECT_OBJECT}
        )

        logger.info("_detect_label_vision_raw=%s", vision_raw)

        prompt = PROMPT_DETECT_LMS.format(vision_raw=vision_raw)

        runnable = self._so(DetectProductOutput)
        det: DetectProductOutput = runnable.invoke(prompt)

        logger.info("label=%s", det)

        state.result["label"] = det.label

        return state

    def _gen_background(self, state: AgentState) -> AgentState:
        label = state.result.get("label") or "M"
        if label == "NONE":
            state.result["generated_images"] = []
            return state

        product_name = state.input_data.get("product_name", "item")
        extra = state.result.get("augment_prompt") or ""

        photos = state.input_data.get("product_photos") or []
        photos_with_url = [
            p for p in photos if isinstance(p, dict) and p.get("image_url")
        ]

        if not photos_with_url:
            state.result["generated_images"] = []
            return state

        label_prompts = _LABEL_PROMPTS.get(label) or _LABEL_PROMPTS["M"]

        generated = []

        logger.info(
            "gen_background: start per-image generation, label=%s, photos=%d",
            label,
            len(photos_with_url),
        )

        for meta in photos_with_url:
            pos = int(meta.get("image_position", 1))
            src_url = meta["image_url"]

            base_prompt = label_prompts.get(pos, label_prompts[1])
            prompt_text = (
                f"{base_prompt.format(product_name=product_name)} {extra}".strip()
            )

            try:
                img_out: ImageGenOutput = generate_images_tool.invoke(
                    {"product_photos": [src_url], "prompt": prompt_text}
                )
                img_b64 = (img_out.generated_images or [""])[0] or ""

                generated.append(
                    {
                        "image_position": pos,
                        "image_base64": img_b64,
                    }
                )

                logger.info(
                    "gen_background: generated image for position=%d (len_b64=%d)",
                    pos,
                    len(img_b64),
                )

            except Exception as e:
                logger.exception(
                    "gen_background: failed to generate image for position=%d: %s",
                    pos,
                    e,
                )
                generated.append(
                    {
                        "image_position": pos,
                        "image_base64": "",
                    }
                )

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

        label_prompts = _LABEL_PROMPTS.get(label) or _LABEL_PROMPTS["M"]

        for idx in range(n):
            img_meta = generated[idx]
            img_b64 = img_meta.get("image_base64") or ""
            pos = int(img_meta.get("image_position", idx + 1))

            if not img_b64:
                continue

            if attempts.get(pos, 0) >= 1:
                continue

            raw_vl_text = inspect_generated_image_tool.invoke(
                {"image_base64": img_b64, "question": PROMPT_VALIDATE_IMAGE_VL.strip()}
            )

            logger.info(
                "описание картинки=%s", raw_vl_text,
            )

            status_str = str(raw_vl_text).strip().upper()

            if status_str == "OK":
                logger.info(
                    "validate_images: position=%s vision_status=OK -> keep image",
                    pos,
                )
                continue

            logger.info(
                "validate_images: position=%s vision_status=%s -> regenerate",
                pos,
                status_str,
            )

            src_url = photos_with_url[idx]["image_url"]

            base_prompt = label_prompts.get(pos, label_prompts[1])
            extra = state.result.get("augment_prompt") or ""
            prompt_text = (
                f"{base_prompt.format(product_name=product_name)} {extra}".strip()
            )

            logger.info(
                "validate_images: position=%s REGEN PROMPT:\n%s",
                pos,
                prompt_text,
            )

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

                    logger.info(
                        "validate_images: position=%s regenerated successfully (attempt=%d)",
                        pos,
                        attempts[pos],
                    )
                else:
                    logger.info(
                        "validate_images: position=%s regen returned empty image (attempt=%d)",
                        pos,
                        attempts[pos],
                    )

            except Exception as e:
                attempts[pos] = attempts.get(pos, 0) + 1
                logger.exception(
                    "validate_images: position=%s regen failed on attempt=%d: %s",
                    pos,
                    attempts[pos],
                    e,
                )
                continue

        state.result["generated_images"] = new_generated

        logger.info(
            "validate_images: finished. regen_attempts=%s",
            attempts,
        )

        return state

    def _gen_characteristics_giga(self, state: AgentState) -> AgentState:
        product_name = state.input_data.get("product_name", "")
        product_properties = state.input_data.get("product_properties") or ""

        prompt = PROMPT_GENERATE_CHARACTERISTICS.format(
            product_name=product_name,
            product_properties=product_properties,
        )

        out: OutputLLM | None = self._invoke_llm_with_retry(
            prompt,
            OutputLLM,
            max_attempts=3,
            delay=0.5,
            log_prefix="gen_characteristics_giga",
        )

        if out is None:
            logger.error(
                "gen_characteristics_giga: failed, leaving characteristics=None"
            )
            state.result["characteristics"] = None
            return state

        logger.info("gen_characteristics_giga: УТП: %s", out)
        state.result["characteristics"] = out
        return state

    def _gen_headers(self, state: AgentState) -> AgentState:
        ch = state.result.get("characteristics")
        title = ch.title if ch else ""

        prompt = PROMPT_GENERATE_HEADERS.format(title=title)

        out: HeadersOutput | None = self._invoke_llm_with_retry(
            prompt,
            HeadersOutput,
            max_attempts=3,
            delay=0.5,
            log_prefix="gen_headers",
        )

        if out is None:
            headers: list[str] = [""]
            logger.error("gen_headers: failed, using empty header")
        else:
            headers = (out.headers or [])[:1]
            headers += [""] * (1 - len(headers))

        state.result["headers"] = headers
        logger.info("ХЕДЕРЫ: %s", headers)
        return state

    def _gen_specs(self, state: AgentState) -> AgentState:
        prompt = PROMPT_GENERATE_SPECS.format(
            product_name=state.input_data.get("product_name", ""),
            product_properties=state.input_data.get("product_properties") or "",
        )

        out: SpecsOutput | None = self._invoke_llm_with_retry(
            prompt,
            SpecsOutput,
            max_attempts=3,
            delay=0.5,
            log_prefix="gen_specs",
        )

        if out is None:
            specs_text = ""
            logger.error("gen_specs: failed, specs_text will be empty")
        else:
            specs_text = out.text or ""

        state.result["specs_text"] = specs_text

        logger.info(
            "СПЕКИ=%r",
            (specs_text[:120] + "...") if specs_text else "",
        )

        return state

    def _gen_description(self, state: AgentState) -> AgentState:
        ch = state.result.get("characteristics")
        if ch is None:
            state.result["description_text"] = ""
            logger.warning("gen_description: no characteristics, description empty")
            return state

        utp_lines = "\n".join(
            f"УТП {i}: {text}" for i, text in enumerate(ch.utp, start=1)
        )

        prompt = PROMPT_GENERATE_DESCRIPTION.format(
            title=ch.title,
            subtitle=ch.subtitle,
            utp_lines=utp_lines,
            specs_text=state.result.get("specs_text", ""),
        )

        out: DescriptionOutput | None = self._invoke_llm_with_retry(
            prompt,
            DescriptionOutput,
            max_attempts=3,
            delay=0.5,
            log_prefix="gen_description",
        )

        if out is None:
            desc_text = ""
            logger.error("gen_description: failed, description_text will be empty")
        else:
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

        ch = state.result.get("characteristics")

        title = ""
        subtitle = ""
        utp_list: list[str] = []
        utp_3_continue = ""
        utp_4_continue = ""
        utp_5_continue = ""

        if ch is not None:
            title = ch.title
            subtitle = ch.subtitle
            utp_list = ch.utp or []
            utp_3_continue = ch.utp_3_continue
            utp_4_continue = ch.utp_4_continue
            utp_5_continue = ch.utp_5_continue

        def utp(i: int) -> str:
            idx = i - 1
            return utp_list[idx] if 0 <= idx < len(utp_list) else ""

        text_block = {
            f"{prefix}-1-slide-title": title,
            f"{prefix}-1-slide-subtitle": subtitle,
            f"{prefix}-1-slide-utp-1": utp(1),
            f"{prefix}-1-slide-utp-2": utp(2),
            f"{prefix}-2-slide-utp-3": utp(3),
            f"{prefix}-2-slide-utp-3-continue": utp_3_continue,
            f"{prefix}-3-slide-utp-4": utp(4),
            f"{prefix}-3-slide-utp-4-continue": utp_4_continue,
            f"{prefix}-3-slide-utp-5": utp(5),
            f"{prefix}-3-slide-utp-5-continue": utp_5_continue,
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

        return {
            "job_id": job_id,
            "text": text_block,
            "generated_images": imgs,
        }

    def run_state(self, input_data: Dict) -> AgentState:
        init = AgentState(input_data=input_data)
        out = self.graph.invoke(init)
        return out if isinstance(out, AgentState) else AgentState(**out)

    def run(self, input_data: Dict) -> dict:
        state = self.run_state(input_data)
        return self._build_payload(state=state)
