from langgraph.graph import END, StateGraph

from models_init.llm_manager import LLMManager
from prompts.prompt import (
    PROMPT_DETECT_LMS,
    PROMPT_GENERATE_ADDITIONAL_ANGLES,
    PROMPT_GENERATE_CHARACTERISTICS,
    PROMPT_GENERATE_DESCRIPTION,
    PROMPT_GENERATE_HEADERS,
    PROMPT_GENERATE_IMAGE_L,
    PROMPT_GENERATE_IMAGE_M,
    PROMPT_GENERATE_IMAGE_S,
    PROMPT_GENERATE_SPECS,
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
from tools.agent_tools import detect_product_tool, generate_images_tool


class ProductCardGeneration:
    def __init__(self) -> None:
        self.llm = LLMManager().llm
        self.graph = self._build_graph()

    def _normalize_photos(self, state: AgentState) -> AgentState:
        photos = state.input_data.get("product_photos", [])
        if not photos:
            state.result["skip_generation"] = True
            state.result["generated_images"] = []
            state.result["used_label"] = "NONE"
            state.result["used_prompt"] = None
            state.result["need_variations"] = 0

            return state
        
        if len(photos) > 4:
            photos = photos[:4]
        state.input_data["product_photos"] = photos
        state.result["need_variations"] = max(0, 4 - len(photos))

        return state

    def _augment_photos(self, state: AgentState) -> AgentState:
        need = state.result.get("need_variations", 0)
        if state.result.get("skip_generation") or need == 0:
            state.result["augment_prompt"] = ""
            return state
        product_name = state.input_data.get("product_name", "товар")
        state.result["augment_prompt"] = PROMPT_GENERATE_ADDITIONAL_ANGLES.format(
            product_name=product_name
        )
        photos = state.input_data["product_photos"]
        state.input_data["product_photos"] = photos + [photos[-1]] * need

        return state

    def _detector_product(self, state: AgentState) -> AgentState:
        image_url = state.input_data.get("product_photos", [None])[0]
        raw_text = detect_product_tool.invoke(
            {"image_url": image_url, "question": PROMPT_DETECT_LMS}
        )
        so_chain = self.llm.with_structured_output(
            DetectProductOutput, strict=True
        ).with_config({"run_name": "detect_lms"})
        det: DetectProductOutput = so_chain.invoke(raw_text)
        label = getattr(det.label, "value", det.label)
        state.result["label"] = label

        return state

    def _background_generation(self, state: AgentState) -> AgentState:
        photos = state.input_data.get("product_photos", [])
        product_name = state.input_data.get("product_name", "kitchenware item")
        label = state.result.get("label", "M")

        if state.result.get("skip_generation"):
            return state
        
        if label == "NONE":
            state.result["skip_generation"] = True
            state.result["generated_images"] = []
            state.result["used_label"] = "NONE"
            state.result["used_prompt"] = None

            return state
        
        base_prompt = {
            "L": PROMPT_GENERATE_IMAGE_L,
            "M": PROMPT_GENERATE_IMAGE_M,
            "S": PROMPT_GENERATE_IMAGE_S,
        }.get(label, PROMPT_GENERATE_IMAGE_M)
        extra = state.result.get("augment_prompt") or ""
        prompt_text = f"{base_prompt.format(product_name=product_name)} {extra}".strip()
        img_out: ImageGenOutput = generate_images_tool.invoke(
            {"product_photos": photos, "prompt": prompt_text}
        )

        state.result["generated_images"] = img_out.generated_images[:4]
        state.result["used_label"] = label
        state.result["used_prompt"] = prompt_text
        state.result["skip_generation"] = False

        return state

    def _generate_characteristics(self, state: AgentState) -> AgentState:
        product_name = state.input_data["product_name"]
        product_props = state.input_data["product_properties"]
        prompt = PROMPT_GENERATE_CHARACTERISTICS.format(
            product_name=product_name,
            product_properties=product_props,
        )
        so_chain = self.llm.with_structured_output(OutputLLM, strict=True).with_config(
            {"run_name": "generate_characteristics"}
        )
        out: OutputLLM = so_chain.invoke(prompt)
        state.result["characteristics"] = out.model_dump()

        return state

    def _generate_headers(self, state: AgentState) -> AgentState:
        title = state.result["characteristics"]["title"]
        prompt = PROMPT_GENERATE_HEADERS.format(title=title)
        so_chain = self.llm.with_structured_output(
            HeadersOutput, strict=True
        ).with_config({"run_name": "generate_headers_so"})

        out: HeadersOutput = so_chain.invoke(prompt)
        state.result["headers"] = out.headers

        return state

    def _generate_specs(self, state: AgentState) -> AgentState:
        product_name = state.input_data["product_name"]
        product_properties = state.input_data["product_properties"]
        prompt = PROMPT_GENERATE_SPECS.format(
            product_name=product_name,
            product_properties=product_properties,
        )
        so_chain = self.llm.with_structured_output(
            SpecsOutput, strict=True
        ).with_config({"run_name": "generate_specs_so"})
        out: SpecsOutput = so_chain.invoke(prompt)
        state.result["specs_text"] = out.text

        return state

    def _generate_description(self, state: AgentState) -> AgentState:
        ch = state.result["characteristics"]
        title = ch.get("title", "")
        subtitle = ch.get("subtitle", "")
        utp = ch.get("utp", [])
        utp_lines = "\n".join(
            f"УТП {item.get('number')}: {item.get('text')}" for item in utp
        )
        specs_text = state.result.get("specs_text", "")
        prompt = PROMPT_GENERATE_DESCRIPTION.format(
            title=title,
            subtitle=subtitle,
            utp_lines=utp_lines,
            specs_text=specs_text,
        )
        so_chain = self.llm.with_structured_output(
            DescriptionOutput, strict=True
        ).with_config({"run_name": "generate_description_so"})
        out: DescriptionOutput = so_chain.invoke(prompt)
        state.result["description_text"] = out.text

        return state

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(AgentState)
        graph.add_node("normalize_photos", self._normalize_photos)
        graph.add_node("augment_photos", self._augment_photos)
        graph.add_node("detector_product", self._detector_product)
        graph.add_node("background_generation", self._background_generation)
        graph.add_node("generate_characteristics", self._generate_characteristics)
        graph.add_node("generate_headers", self._generate_headers)
        graph.add_node("generate_specs", self._generate_specs)
        graph.add_node("generate_description", self._generate_description)
        graph.set_entry_point("normalize_photos")
        graph.add_edge("normalize_photos", "augment_photos")
        graph.add_edge("augment_photos", "detector_product")
        graph.add_edge("detector_product", "background_generation")
        graph.add_edge("background_generation", "generate_characteristics")
        graph.add_edge("generate_characteristics", "generate_headers")
        graph.add_edge("generate_headers", "generate_specs")
        graph.add_edge("generate_specs", "generate_description")
        graph.add_edge("generate_description", END)

        return graph.compile()

    def _build_payload(self, state: AgentState) -> dict:
        job_id = state.input_data.get("job_id")
        images = state.result.get("generated_images", [])
        label = state.result.get("used_label") or state.result.get("label", "M")
        prefix = (label or "M").lower()
        ch = state.result["characteristics"]
        title = ch["title"]
        subtitle = ch["subtitle"]
        utp = ch["utp"]
        specs_text = state.result.get("specs_text", "")
        description_text = state.result.get("description_text", "")
        text = {
            f"{prefix}-1-slide-title": title,
            f"{prefix}-1-slide-subtitle": subtitle,
            f"{prefix}-specs": specs_text,
            f"{prefix}-description": description_text,
        }
        text.update({f"{prefix}-{i}-utp": utp[i - 1]["text"] for i in range(1, 9)})
        return {"job_id": job_id, "generated_images": images, "text": text}

    def run_state(self, input_data) -> AgentState:
        init_state = AgentState(input_data=input_data)
        out = self.graph.invoke(init_state)
        
        return out if isinstance(out, AgentState) else AgentState(**out)

    def run(self, input_data) -> dict:
        return self._build_payload(self.run_state(input_data))
