from langgraph.graph import StateGraph, END

from schemas.schema import (
    AgentState,
    OutputLLM,
    ImageGenOutput,
    SpecsOutput,
    HeadersOutput,
    DetectProductOutput,
    DescriptionOutput,
)
from models_init.llm_manager import LLMManager
from tools.agent_tools import generate_images_tool, detect_product_tool
from prompts.prompt import (
    PROMPT_DETECT_LMS,
    PROMPT_GENERATE_SPECS,
    PROMPT_GENERATE_HEADERS,
    PROMPT_GENERATE_CHARACTERISTICS,
    PROMPT_GENERATE_DESCRIPTION,
)


class ProductCardGeneration:
    def __init__(self):
        self.llm = LLMManager().llm
        self.graph = self._build_graph()

    def _detector_product(self, state: AgentState) -> AgentState:
        image_url = state.input_data.get("product_photos", [None])[0]
        raw_text = detect_product_tool.invoke(
            {"image_url": image_url, "question": PROMPT_DETECT_LMS}
        )

        so_chain = self.llm.with_structured_output(
            DetectProductOutput, strict=True
        ).with_config({"run_name": "detect_lms"})

        det: DetectProductOutput = so_chain.invoke(raw_text)
        state.result["detected_product"] = det.model_dump()

        return state

    def _background_generation(self, state: AgentState) -> AgentState:
        photos = state.input_data.get("product_photos", [])
        img_out: ImageGenOutput = generate_images_tool.invoke(
            {"product_photos": photos}
        )
        state.result["generated_images"] = img_out.generated_images

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
            [f"УТП {item.get('number')}: {item.get('text')}" for item in utp]
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
        graph.add_node("detector_product", self._detector_product)
        graph.add_node("background_generation", self._background_generation)
        graph.add_node("generate_characteristics", self._generate_characteristics)
        graph.add_node("generate_headers", self._generate_headers)
        graph.add_node("generate_specs", self._generate_specs)
        graph.add_node("generate_description", self._generate_description) 

        graph.set_entry_point("detector_product")
        
        graph.add_edge("detector_product", "background_generation")
        graph.add_edge("background_generation", "generate_characteristics")
        graph.add_edge("generate_characteristics", "generate_headers")
        graph.add_edge("generate_headers", "generate_specs")
        graph.add_edge("generate_specs", "generate_description")           
        graph.add_edge("generate_description", END)                       

        return graph.compile()

    def run(self, input_data) -> AgentState:
        state = AgentState(input_data=input_data)
        return self.graph.invoke(state)