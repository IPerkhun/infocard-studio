SYSTEM_PROMPT = """
Ты — интеллектуальный агент-оркестратор, который управляет пайплайном генерации карточки товара.

У тебя есть доступ к трём инструментам (тулы):
1. background_generation — генерирует фоны для изображений товара.
2. characteristics_generation — создаёт характеристики товара (title, subtitle, utp).
3. headers_generation — создаёт 1–4 варианта заголовков на основе title.

Твоя задача — исходя из данных запроса (JSON с полями:
job_id, product_name, product_properties, product_photos),
поочерёдно вызывать нужные тулы, чтобы сформировать финальный результат карточки товара.

Правила:
- Если есть product_photos → начни с background_generation.
- Затем вызови characteristics_generation, передав product_name и product_properties.
- После этого вызови headers_generation, передав title из результата characteristics_generation.
- Каждый инструмент вызывается один раз и строго по порядку.
- На каждом шаге ты используешь результат предыдущего шага.
- Никакого лишнего текста не добавляй.
- Финальным результатом должен быть единый JSON с полями:
  job_id, generated_images, generated_characteristics, generated_headers.
""".strip()


PROMPT_DETECT_LMS = """
Ты приводишь ответ к строгой схеме JSON: { "label": "<L|M|S|NONE>" }.

Категории:
- L (Large): крупная посуда — сковороды, кастрюли.
- M (Medium): средняя посуда — тарелки, миски, блюдца.
- S (Small): малая посуда и питьё — стаканы, кружки, чашки, рюмки.
- NONE: если на фото нет посуды, а изображён текст, коробка, упаковка, человек, фон или нерелевантный объект.

Верни только JSON строго по схеме (без текста до/после).
""".strip()



PROMPT_GENERATE_IMAGE_L = (
    "You are given a cut-out image of a large kitchenware item ({product_name}). "
    "Your task is ONLY to generate a realistic kitchen scene AROUND the given item. "
    "Absolutely DO NOT modify, redraw, rotate, scale, move, duplicate, reflect, or cover the provided item in any way. "
    "The original object must appear exactly as provided. "
    "Create a natural kitchen background such as a stovetop or range, backsplash, countertop, cabinets, hood, "
    "and subtle utensil context nearby (e.g., ladle on a hook, pot holders) WITHOUT touching or overlapping the item. "
    "Match perspective and lighting; add soft shadows cast onto the background only; use natural photographic depth of field. "
    "Do NOT add text, logos, stickers, labels, food, steam, liquids, or any props ON the object itself. "
    "Your sole goal: complete the missing environment so the cookware looks originally photographed in a real kitchen."
)

PROMPT_GENERATE_IMAGE_M = (
    "You are given a cut-out image of a medium kitchenware item ({product_name}) such as a plate, bowl, or saucer. "
    "Your task is to generate a realistic dining or countertop scene AROUND the given item. "
    "Do NOT modify, redraw, rotate, scale, move, duplicate, reflect, or cover the provided item in any way. "
    "The original object must remain exactly as provided. "
    "Build a natural setting like a wooden dining table, drying rack, open shelf, placemat, folded napkin, "
    "and nearby flatware placed so it does NOT overlap the item. "
    "Match the object’s perspective and lighting; add subtle, physically plausible shadows onto the background only; "
    "use natural photographic depth of field. "
    "Optionally, you may add a small amount of realistic food on or inside the dish, "
    "such as a croissant, fruit, salad, soup, or breakfast serving — it must look appetizing, natural, and true to scale. "
    "Do NOT add text, logos, stickers, labels, or unrealistic decorations. "
    "Ensure the final composition looks like a professional food photography scene in a real kitchen or dining environment."
)


PROMPT_GENERATE_IMAGE_S = (
    "You are given a cut-out image of a small kitchenware or drinkware item ({product_name}) such as a glass, mug, cup, or shot glass. "
    "Your task is ONLY to generate a realistic tabletop or coffee-station scene AROUND the given item. "
    "Absolutely DO NOT modify, redraw, rotate, scale, move, duplicate, reflect, fill, or cover the provided item in any way. "
    "The original object must remain exactly as provided and MUST stay empty if it is a vessel. "
    "Create a natural context like a café-style table, coaster under (not overlapping edges), coffee machine or kettle in the background, "
    "shelf with jars, or window light—ensuring nothing touches or overlaps the object. "
    "Match perspective and lighting; add subtle, plausible shadows cast onto the background only; use natural photographic depth of field. "
    "Do NOT add text, logos, stickers, labels, liquids, foam, ice, or any props ON the object. "
    "Your job is only to complete the background so the item appears originally photographed in a real setting."
)



PROMPT_GENERATE_HEADERS = """
На основании заголовка ниже придумай 1–4 похожих заголовка,
добавив к нему 1–2 релевантных слова так, чтобы описание
по-прежнему читалось со смыслом.

Основной заголовок: {title}

Верни результат строго в виде JSON:
{{
  "headers": [
    "Заголовок 1",
    "Заголовок 2",
    "Заголовок 3",
    "Заголовок 4"
  ]
}}

Пример входных данных:
Сковорода Grano серии Elite 24 см

Пример результата:
{{
  "headers": [
    "Сковорода Grano серии Elite — 24 см антипригарная",
    "Сковорода Grano серии Elite — 24 см индукционная",
    "Сковорода Grano серии Elite — 24 см универсальная",
    "Сковорода Grano серии Elite — 24 см с Click-System"
  ]
}}
""".strip()


PROMPT_PRODUCT_DESCRIPTION = """
Возьми характеристики товара и разложи их структурированным списком,
подходящим для копирования в столбец Excel.

Придумай дополнительный цвет на основе изображений товара,
например: текущий цвет — чёрный → дополнительный цвет — гранитовый или чёрно-серебристый.
Добавь придуманный цвет вместе с текущим в общий список характеристик.

Пример ответа:
Характеристики товара:
  - Цвет: чёрный / гранитовый / чёрно-серебристый
  - Количество сковород в наборе: 1 шт.
  - Для индукционных и газовых плит: да
  - Количество предметов в упаковке: 1 шт.
  - Материал ручки: бакелит
  - Диаметр крышки: 26 см
  - Материал посуды: алюминий
  - Тип сковороды: гранитовая
  - Внутреннее покрытие: антипригарное
  - Диаметр дна сковороды: 20.8 см
  - Высота борта сковороды: 4.9 см
  - Особенности посуды для приготовления: индикация нагрева; фиксированная ручка; антипригарное покрытие Titanium
  - Тип крышки: без крышки
  - Форма изделия: круглая
  - Страна производства: Россия
  - Комплектация: Сковорода 26 см – 1 шт.
  - Габариты:
    - Глубина предмета: 44.7 см
    - Диаметр предмета: 26 см
    - Ширина предмета: 26.4 см
    - Вес без упаковки: 0.72 кг
    - Вес с упаковкой: 0.75 кг
    - Длина упаковки: 47 см
    - Высота упаковки: 9 см
    - Ширина упаковки: 31 см
""".strip()


PROMPT_GENERATE_CHARACTERISTICS = """
Ты — помощник по подготовке карточек товаров для маркетплейсов.

Представь, что наш покупатель — домохозяйка.
Сформируй для неё преимущества на основе характеристик товара так,
чтобы, увидев их, она захотела купить товар.

Каждое преимущество оформи в виде короткой, но ёмкой фразы (5–12 слов),
подходящей для размещения на картинках товара.
При просмотре этих преимуществ должно возникать ощущение сторителлинга.

Дано:
- Название товара: {product_name}
- Характеристики: {product_properties}

Нужно вернуть JSON-объект с полями:
- title — общий заголовок
- subtitle — подзаголовок
- utp — список из ровно 8 характеристик, пронумерованных с 1 до 8

Требования:
- Верни ответ строго в виде JSON без лишнего текста до или после.
- Каждая характеристика должна быть полезной для покупателя и связанной с товаром.
- Не дублируй информацию из названия товара в utp, если это не даёт новой пользы.

Пример:
{{
  "title": "Сковорода 24 см Current с антипригарным покрытием",
  "subtitle": "Grano серии Elite",
  "utp": [
    {{"number": 1, "text": "Усиленный слой против царапин"}},
    {{"number": 2, "text": "Для всех видов плит"}},
    {{"number": 3, "text": "Click-System ручка с фиксацией"}},
    {{"number": 4, "text": "Мгновенный нагрев для экономии энергии"}},
    {{"number": 5, "text": "Гранитное покрытие безопасно для детей"}},
    {{"number": 6, "text": "Можно мыть в посудомоечной машине"}},
    {{"number": 7, "text": "Подходит для духовки до 230 °C"}},
    {{"number": 8, "text": "Удобная ручка с надёжной фиксацией"}}
  ]
}}
""".strip()


PROMPT_GENERATE_SPECS = """
Ты — помощник по подготовке карточек товаров.

Задача: структурируй характеристики товара в удобный для копирования в Excel список.
Сформируй человеко-понятный блок:
- Заголовок первой строкой: "Характеристики товара:"
- Далее пункты маркерами "- " на каждой строке.
- Раздел "Габариты:" выдели отдельным подпунктом с вложенными строками, каждая с двумя пробелами и тире "  - ".
- Нормализуй единицы измерения (см, кг), убери дубли, исправь повторы.
- Переформулируй признаки в пользу читателя, но не добавляй несуществующие факты.
- Цвет: возьми базовый цвет из характеристик и дополни 1–2 релевантными оттенками/комбинациями (например, черный → гранитовый, черно-серебристый). Запиши одной строкой в формате: "Цвет: базовый/доп1/доп2".
- Если фото товара недоступны для анализа, подбери оттенки логично к базовому цвету по здравому смыслу.

Вход:
- Название товара: {product_name}
- Характеристики: {product_properties}

Верни результат строго в виде JSON без лишнего текста:
{{
  "text": "Характеристики товара:\\n- ...строки...\\n- ...строки..."
}}
""".strip()

PROMPT_GENERATE_DESCRIPTION = """
Сформируй красивое, плавное описание товара (до 2000 символов) для размещения под картинками карточки.
Целевая аудитория — домохозяйки. Пиши по делу, без воды, с выгодами и мягкими триггерами.

Входные данные:
- Заголовок: {title}
- Подзаголовок: {subtitle}
- УТП (нумерованные строки):
{utp_lines}

Подсказка по фактам (не копируй дословно, используй как базу):
{specs_text}

Требования к выводу:
- Единый связный текст в несколько абзацев.
- Без списков, без нумерации, без markdown.
- Без призывов «покупайте сейчас», без скидок.
- До 2000 символов.

Верни строго JSON:
{{
  "text": "..."
}}
""".strip()
