#%%
from langgraph.graph import StateGraph, END
from schemas.schema import AgentState, OutputLLM, OutputHeaders
from models_init.llm_manager import LLMManager
from tools.agent_tools import (
    generate_images_tool,
    generate_characteristics_tool,
    generate_headers_tool,
)
from pydantic import ValidationError


class ProductCardGeneration:
    def __init__(self):
        self.llm = LLMManager(local_llm=True).client
        self.graph = self._build_graph()

    def _detector_product(self, state: AgentState) -> AgentState:
        # пока пустой детектор (проходим дальше)
        return state

    def _generate_images_node(self, state: AgentState) -> AgentState:
        img_result = generate_images_tool.invoke(dict(state.input_data))
        state.result.update({
            "job_id": img_result.job_id,
            "generated_images": img_result.generated_images,
        })
        return state

    def _generate_characteristics_node(self, state: AgentState) -> AgentState:
        raw_result = generate_characteristics_tool.invoke(dict(state.input_data))

        if isinstance(raw_result, OutputLLM):
            parsed = raw_result
        else:
            try:
                parsed = OutputLLM.model_validate_json(raw_result)
            except ValidationError as e:
                raise ValueError(f"Не удалось распарсить ответ модели: {e}")

        state.result["generated_characteristics"] = parsed.model_dump()
        return state

    def _generate_headers_node(self, state: AgentState) -> AgentState:
        title = state.result["generated_characteristics"]["title"]
        headers_result = generate_headers_tool.invoke({"title": title})

        if isinstance(headers_result, OutputHeaders):
            headers = headers_result.model_dump()
        else:
            headers = headers_result

        state.result["generated_headers"] = headers
        return state

    # 🔹 новая пустая нода для описания товара
    def _generate_description_node(self, state: AgentState) -> AgentState:
        # тут позже подключим тул/LLM; пока — плейсхолдер
        state.result["generated_description"] = ""
        return state

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(AgentState)

        graph.add_node("detector_product", self._detector_product)
        graph.add_node("background_generation", self._generate_images_node)
        graph.add_node("characteristics_generation", self._generate_characteristics_node)
        graph.add_node("headers_generation", self._generate_headers_node)
        graph.add_node("description_generation", self._generate_description_node)  # ← добавили

        graph.add_edge("detector_product", "background_generation")
        graph.add_edge("background_generation", "characteristics_generation")
        graph.add_edge("characteristics_generation", "headers_generation")
        graph.add_edge("headers_generation", "description_generation")              # ← вставили сюда
        graph.add_edge("description_generation", END)

        graph.set_entry_point("detector_product")
        return graph.compile()

    def run(self, input_data: dict) -> AgentState:
        init_state = AgentState(input_data=input_data)
        return self.graph.invoke(init_state)

# %%
temp = ProductCardGeneration()
# %%
temp.run(
    input_data={
        "job_id": "1500cffa-5c2b-47b3-b879-3b6689fde248",
        "product_sku": "293243771",
        "product_name": "Сковорода 24 см Current с антипригарным покрытием",
        "product_properties": "Цвет: красный. Количество сковород в наборе: 1 шт. Для индукционных плит: нет. Количество предметов в упаковке: 1 шт. Материал ручки: бакелит. Материал посуды: алюминий. Внутреннее покрытие: антипригарное. Тип сковороды: классическая. Диаметр крышки: 26 см. Диаметр дна сковороды: 20.8 см. Высота борта сковороды: 4.9 см. Особенности: индикация нагрева; фиксированная ручка; антипригарное покрытие Titanium. Тип крышки: без крышки. Форма изделия: круглая. Страна производства: Россия. Комплектация: сковорода 26 см – 1 шт. Глубина предмета: 44.7 см. Диаметр предмета: 26 см. Ширина предмета: 26.4 см. Вес без упаковки: 0.72 кг. Вес с упаковкой: 0.75 кг. Длина упаковки: 47 см. Высота упаковки: 9 см. Ширина упаковки: 31 см.",
        "product_photos": [
            "https://goods-photos.static1-sima-land.com/items/20392/0/1600.jpg",
        ],
    }
)
# %%
