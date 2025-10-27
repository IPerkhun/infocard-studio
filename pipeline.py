from __future__ import annotations

from typing import Dict, List

from langgraph.graph import END, StateGraph

from models_init.llm_manager import LLMManager
from prompts.prompt import (PROMPT_DETECT_LMS,
                            PROMPT_GENERATE_ADDITIONAL_ANGLES,
                            PROMPT_GENERATE_CHARACTERISTICS,
                            PROMPT_GENERATE_DESCRIPTION,
                            PROMPT_GENERATE_HEADERS, PROMPT_GENERATE_IMAGE_L,
                            PROMPT_GENERATE_IMAGE_M, PROMPT_GENERATE_IMAGE_S,
                            PROMPT_GENERATE_SPECS)
from schemas.schema import (AgentState, DescriptionOutput, DetectProductOutput,
                            HeadersOutput, ImageGenOutput, OutputLLM,
                            SpecsOutput)
from tools.agent_tools import detect_product_tool, generate_images_tool

_LABEL_PROMPTS = {
    "L": PROMPT_GENERATE_IMAGE_L,
    "M": PROMPT_GENERATE_IMAGE_M,
    "S": PROMPT_GENERATE_IMAGE_S,
}


class ProductCardGeneration:
    def __init__(self) -> None:
        self.llm = LLMManager().llm
        self.graph = self._build_graph()

    def _so(self, model, run_name: str):
        return self.llm.with_structured_output(model, strict=True).with_config(
            {"run_name": run_name}
        )

    @staticmethod
    def _as_urls(photos: List) -> List[str]:
        urls: List[str] = []
        for p in photos or []:
            if isinstance(p, str):
                urls.append(p)
            elif isinstance(p, dict):
                u = p.get("image_url")
                if u:
                    urls.append(u)
        return urls

    def _normalize_photos(self, state: AgentState) -> AgentState:
        photos = state.input_data.get("product_photos", []) or []

        norm: List[Dict] = []
        seen = set()
        for idx, it in enumerate(photos):
            if isinstance(it, str):
                url = it
                pos = idx + 1
                img_id = None
            else:
                url = it.get("image_url")
                pos = it.get("image_position") or (idx + 1)
                img_id = it.get("image_id")

            if not url or url in seen:
                continue
            seen.add(url)
            norm.append(
                {
                    "image_url": url,
                    "image_position": int(pos),
                    "image_id": img_id,
                }
            )

        norm.sort(key=lambda x: x["image_position"])
        norm = norm[:4]

        for i, n in enumerate(norm, start=1):
            n["image_position"] = i

        state.input_data["product_photos"] = norm
        state.result["need_variations"] = max(0, 4 - len(norm))
        return state

    def _augment_photos(self, state: AgentState) -> AgentState:
        need = state.result.get("need_variations", 0)
        if need <= 0:
            state.result["augment_prompt"] = ""
            return state

        product_name = state.input_data.get("product_name", "товар")
        state.result["augment_prompt"] = PROMPT_GENERATE_ADDITIONAL_ANGLES.format(
            product_name=product_name
        )

        photos = state.input_data["product_photos"]
        last_pos = photos[-1]["image_position"] if photos else 0

        augmented = []
        for i in range(need):
            clone = photos[-1].copy()
            clone["image_position"] = last_pos + i + 1
            augmented.append(clone)

        state.input_data["product_photos"] = photos + augmented
        return state

    def _detect_label(self, state: AgentState) -> AgentState:
        tpl = (state.input_data.get("template") or "").strip().upper()
        if tpl in {"L", "M", "S", "NONE"}:
            state.result["label"] = tpl
            state.result["detected_by"] = "template"

            print(f"_detect_label{state.result}")

            return state

        urls = self._as_urls(state.input_data.get("product_photos"))
        if not urls:
            state.result["label"] = "NONE"
            state.result["detected_by"] = "empty"
            return state

        raw = detect_product_tool.invoke(
            {"image_url": urls[0], "question": PROMPT_DETECT_LMS}
        )
        det: DetectProductOutput = self._so(DetectProductOutput, "detect_lms").invoke(
            raw
        )
        state.result["label"] = det.label
        state.result["detected_by"] = "detector"

        return state

    def _gen_background(self, state: AgentState) -> AgentState:
        label = state.result.get("label", "M")
        if label == "NONE":
            state.result.update(
                {
                    "generated_images": [],
                    "used_label": "NONE",
                    "used_prompt": None,
                }
            )
            return state

        base_prompt = _LABEL_PROMPTS.get(label, PROMPT_GENERATE_IMAGE_M)
        extra = state.result.get("augment_prompt") or ""
        prompt_text = f"{base_prompt.format(product_name=state.input_data.get('product_name', 'item'))} {extra}".strip()

        photos = state.input_data.get("product_photos") or []
        urls = [
            p.get("image_url")
            for p in photos
            if isinstance(p, dict) and p.get("image_url")
        ]
        if not urls:
            state.result.update(
                {
                    "generated_images": [],
                    "used_label": label,
                    "used_prompt": prompt_text,
                }
            )
            return state

        img_out: ImageGenOutput = generate_images_tool.invoke(
            {
                "product_photos": urls,
                "prompt": prompt_text,
            }
        )

        enriched = []
        for meta, b64 in zip(photos, img_out.generated_images):
            enriched.append(
                {
                    "image_id": meta.get("image_id"),
                    "image_position": meta.get("image_position"),
                    "image_base64": b64,
                }
            )

        state.result.update(
            {
                "generated_images": enriched[:4],
                "used_label": label,
                "used_prompt": prompt_text,
            }
        )
        return state

    def _gen_characteristics(self, state: AgentState) -> AgentState:
        prompt = PROMPT_GENERATE_CHARACTERISTICS.format(
            product_name=state.input_data["product_name"],
            product_properties=state.input_data.get("product_properties") or "",
        )

        out = self._so(OutputLLM, "generate_characteristics").invoke(prompt)

        if not isinstance(out, OutputLLM):
            from pydantic import ValidationError

            try:
                out = OutputLLM.model_validate(out)
            except ValidationError:
                import json

                raw = out if isinstance(out, str) else str(out)
                s = raw.find("{")
                e = raw.rfind("}")
                data = json.loads(raw[s : e + 1])
                out = OutputLLM.model_validate(data)

        state.result["characteristics"] = out.model_dump()

        print(f"characteristics{state.result["characteristics"]}")
        return state

    def _gen_headers(self, state: AgentState) -> AgentState:
        title = state.result["characteristics"]["title"]
        out: HeadersOutput = self._so(HeadersOutput, "generate_headers").invoke(
            PROMPT_GENERATE_HEADERS.format(title=title)
        )
        state.result["headers"] = out.headers
        return state

    def _gen_specs(self, state: AgentState) -> AgentState:
        out: SpecsOutput = self._so(SpecsOutput, "generate_specs").invoke(
            PROMPT_GENERATE_SPECS.format(
                product_name=state.input_data["product_name"],
                product_properties=state.input_data.get("product_properties") or "",
            )
        )
        state.result["specs_text"] = out.text
        return state

    def _gen_description(self, state: AgentState) -> AgentState:
        ch = state.result["characteristics"]
        utp_lines = "\n".join(
            f"УТП {u['number']}: {u['text']}" for u in ch.get("utp", [])
        )
        out: DescriptionOutput = self._so(
            DescriptionOutput, "generate_description"
        ).invoke(
            PROMPT_GENERATE_DESCRIPTION.format(
                title=ch.get("title", ""),
                subtitle=ch.get("subtitle", ""),
                utp_lines=utp_lines,
                specs_text=state.result.get("specs_text", ""),
            )
        )
        state.result["description_text"] = out.text
        print(f"description_text{state.result["description_text"]}")
        return state

    def _build_graph(self):
        g = StateGraph(AgentState)
        g.add_node("normalize_photos", self._normalize_photos)
        g.add_node("augment_photos", self._augment_photos)
        g.add_node("detect_label", self._detect_label)
        g.add_node("gen_background", self._gen_background)
        g.add_node("gen_characteristics", self._gen_characteristics)
        g.add_node("gen_headers", self._gen_headers)
        g.add_node("gen_specs", self._gen_specs)
        g.add_node("gen_description", self._gen_description)

        g.set_entry_point("normalize_photos")
        g.add_edge("normalize_photos", "augment_photos")
        g.add_edge("augment_photos", "detect_label")
        g.add_edge("detect_label", "gen_background")
        g.add_edge("gen_background", "gen_characteristics")
        g.add_edge("gen_characteristics", "gen_headers")
        g.add_edge("gen_headers", "gen_specs")
        g.add_edge("gen_specs", "gen_description")
        g.add_edge("gen_description", END)

        return g.compile()

    @staticmethod
    def _build_payload(state: AgentState) -> dict:
        job_id = state.input_data.get("job_id")
        images = state.result.get("generated_images") or []
        label = state.result.get("used_label") or state.result.get("label", "M")
        prefix = (label or "M").lower()

        ch = state.result.get("characteristics", {}) or {}
        title = ch.get("title", "") or ""
        subtitle = ch.get("subtitle", "") or ""
        utp_list = ch.get("utp", []) or []

        def utp(i: int) -> str:
            idx = i - 1
            return utp_list[idx]["text"] if 0 <= idx < len(utp_list) else ""

        utp_3_cont = ch.get("utp_3_continue", "") or ""
        utp_4_cont = ch.get("utp_4_continue", "") or ""
        utp_5_cont = ch.get("utp_5_continue", "") or ""

        specs_text = state.result.get("specs_text", "") or ""
        description_text = state.result.get("description_text", "") or ""

        text_block = {
            f"{prefix}-1-slide-title": title,
            f"{prefix}-1-slide-subtitle": subtitle,
            f"{prefix}-1-slide-utp-1": utp(1),
            f"{prefix}-1-slide-utp-2": utp(2),

            f"{prefix}-2-slide-utp-3": utp(3),
            f"{prefix}-2-slide-utp-3-continue": utp_3_cont,

            f"{prefix}-3-slide-utp-4": utp(4),
            f"{prefix}-3-slide-utp-4-continue": utp_4_cont,
            f"{prefix}-3-slide-utp-5": utp(5),
            f"{prefix}-3-slide-utp-5-continue": utp_5_cont,

            f"{prefix}-4-slide-utp-6": utp(6),
            f"{prefix}-4-slide-utp-7": utp(7),
            f"{prefix}-4-slide-utp-8": utp(8),

            f"{prefix}-specs": specs_text,
            f"{prefix}-description": description_text,
        }

        gen_images = []
        for meta in images:
            gen_images.append(
                {
                    "image_position": meta.get("image_position"),
                    "image_base64": meta.get("image_base64", ""),
                }
            )

        payload = {
            "job_id": job_id,
            "text": text_block,
            "generated_images": gen_images,
        }
        return payload


    def run_state(self, input_data: Dict) -> AgentState:
        init = AgentState(input_data=input_data)
        out = self.graph.invoke(init)
        return out if isinstance(out, AgentState) else AgentState(**out)

    def run(self, input_data: Dict) -> dict:
        state = self.run_state(input_data)
        return self._build_payload(state)
