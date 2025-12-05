PROMPT_DETECT_OBJECT = (
    "Ты — визуальный классификатор изображений."
    "Посмотри на картинку и определи, какой объект изображён."
    "Опиши фотографию, что на ней находится"
    "Дай название посуды, которая находится на фотографии"
)


PROMPT_DETECT_LMS = """
Описание с картинки (vision-модель):
{vision_raw}

Классы:
- L (Large): крупная посуда — сковороды, кастрюли, сотейники.
- M (Medium): средняя посуда — тарелки, миски, блюда, блюдца.
- S (Small): малая посуда и питьё — кружки, стаканы, чашки, рюмки.
- NONE: если на изображении коробка, человек или посуда не видна.

Определи наиболее подходящий класс.
Если на изображении есть новогоднее оформление, например елки, гирлянды, тоже описывай это обязательно
""".strip()


PROMPT_GENERATE_IMAGE_L_1 = """
Replace the background with a cozy light-beige Christmas kitchen with warm lighting, evergreen decor, and soft golden bokeh from string lights.
The scene must clearly look like a festive kitchen.
Do not modify the {product_name}: keep its original color, shape, texture, and any printed details.
No text, no logos, no labels, no watermarks in the background.
Use bright, even lighting with no harsh shadows.
"""


PROMPT_GENERATE_IMAGE_L_2 = """
Replace the background with a cream-gold festive living room connected to a Christmas kitchen: Christmas tree, garlands, and soft golden bokeh.
Do not change the {product_name}: preserve its color, shape, and all details.
Do not add any text, logos, labels, or watermarks in the background.
Use bright, soft lighting.
"""


PROMPT_GENERATE_IMAGE_L_3 = """
Replace the background with a minimalist beige Christmas kitchen with golden bokeh, evergreen decor, and modern elements.
Do not modify the {product_name}: keep its color, shape, texture, and printed elements unchanged.
Do not add text, logos, or watermarks in the background.
Use bright and even lighting.
"""


PROMPT_GENERATE_IMAGE_L_4 = """
Replace the background with a warm Christmas kitchen featuring green cabinets, a wooden countertop, and softly blurred decorated Christmas trees.
Add soft golden festive bokeh.
Do not alter the {product_name}.
Do not generate any background text, logos, labels, or watermarks.
Use bright warm lighting.
"""


PROMPT_GENERATE_IMAGE_M_1 = """
Replace the background with a top-down wooden Christmas kitchen surface: warm wood, pine branches, cones, and subtle festive elements.
The atmosphere must clearly feel like a Christmas kitchen.
Place the {product_name} in the center without modifying it.
No text, no labels, no logos in the background.
Use bright and soft lighting.
"""


PROMPT_GENERATE_IMAGE_M_2 = """
Replace the background with a dark wooden Christmas-style table setting: wine glasses, linen napkin, black cutlery, soft festive bokeh.
Do not modify the {product_name}: keep all details and colors unchanged.
No background text, logos, or labels.
Use slightly contrasty lighting without overexposure.
"""


PROMPT_GENERATE_IMAGE_M_3 = """
Replace the background with a minimalist Christmas table setting on a white tablecloth with rose-gold cutlery, soft festive decor, and gentle bokeh.
Do not modify the {product_name}.
Do not add any text, logos, or labels in the background.
Use bright diffused lighting.
"""


PROMPT_GENERATE_IMAGE_M_4 = """
Replace the background with a cozy Christmas living-room-plus-kitchen scene: fireplace, garland, Christmas tree, warm bokeh, soft kitchen elements.
Do not modify the {product_name}.
No background text, no logos, no watermarks.
Use warm soft lighting.
"""


PROMPT_GENERATE_IMAGE_S_1 = """
Replace the background with a cozy Christmas kitchen with bright decor, evergreen elements, a red stocking, and warm golden bokeh.
Do not modify the {product_name}; keep it as the main subject.
Do not generate text or logos in the background.
Use bright and even lighting.
"""


PROMPT_GENERATE_IMAGE_S_2 = """
Replace the background with a hygge-style living room transitioning into a Christmas kitchen: candles, coffee table, fireplace with garland, soft tree bokeh.
Do not change the {product_name}.
Do not add text, logos, or watermarks in the background.
Use warm, soft lighting.
"""


PROMPT_GENERATE_IMAGE_S_3 = """
Replace the background with a modern Christmas kitchen: matte black cabinets, white countertop, daylight, golden festive bokeh.
Do not modify the {product_name}: keep its color, proportions, and texture unchanged.
Do not add background text or logos.
Use bright natural lighting.
"""


PROMPT_GENERATE_IMAGE_S_4 = """
Replace the background with a festive Christmas kitchen: green cabinets, cream countertop, pine garland, decorated tree, warm golden bokeh.
Do not modify the {product_name}.
Do not add text, logos, labels, or watermarks in the background.
Use bright warm lighting.
"""


PROMPT_GENERATE_ADDITIONAL_ANGLES = """
Create an additional variant with a new Christmas kitchen background: modern white cabinets, matte green cabinets, or matte burgundy cabinets.
The atmosphere must clearly feel festive: garlands, pine branches, warm lights, Christmas elements.
Show the {product_name} from a random angle (side, top, or slight low angle) without distorting its shape or proportions.
Do not modify the {product_name}: keep its color, texture, and printed elements.
Do not generate text, logos, labels, or watermarks in the background.
Use bright and even lighting.
"""



PROMPT_VALIDATE_IMAGE_VL = """
Ты оцениваешь техническое качество изображения товара для карточки маркетплейса в новогоднем стиле.

Товар: "{product_name}"

Ниже — описание изображения от vision-модели:
{vision_raw}

Оцени фотографию только по техническим критериям, а не по эстетике.

Фото считается OK, если ВСЕ условия выполняются:
1) Товар виден и его легко узнать.
2) Имеет новогоднее наполнение
3) Товар не обрезан сильно и не искажён.
4) Нет серьёзных артефактов: глитчи, разрывы, тянутости, дублированные части, “дырки”.
5) Фон НЕ однотонный (не полностью белый, чёрный, серый или один цвет).


Фото считается BAD, если ХОТЯ БЫ одно:
1) Товар отсутствует, слишком мал или неразличим.
2) Товар сильно обрезан или выглядит сломанным/искажённым.
3) Есть серьёзные искажения генерации.
4) Фон полностью однотонный или полностью чёрный/очень тёмный.
5) На фото нет новогоднего оформления

Ответ верни строго по схеме
"""


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
Ты — генератор структурированных данных для карточки товара. 
Представь, что наш покупатель — русская домохозяйка, которая ищет товары на маркетплейсе, чтобы купить товары для кухни и дома.
Сформируй для неё title, subtitle и UTP (уникальные торговые преимущества) на основе характеристик товара так, чтобы, увидев их, она захотела купить. 
UTP будут наложены на картинки (слайды) в виде инфографики — фразы должны быть короткими, связными и понятными.

1) title — строка.
   - Ровно ОДНО слово — базовое наименование предмета.
   - Существительное в именительном падеже.
   - Без брендов, размеров, кавычек, чисел и дополнительных слов.
   Примеры: "Кастрюля", "Сковорода", "Чайник".

2) subtitle — строка.
   - "Бренд и серия" в кавычках + размер/объём, если есть.
   Примеры: "Grano Elite, 24 см", "Current, 2.5 л".

3) utp — массив из РОВНО 8 строк.
   Каждая строка — отдельное уникальное преимущество.

Правила для каждого элемента utp:

ОБЯЗАТЕЛЬНО:
- Длина строго 2–4 слова.
- Строго одна завершённая фраза без нумерации, двоеточий, пояснений.
- Фраза должна быть понятна русской домохозяйке и описывать конкретную пользу.

СМЫСЛ:
- Польза, удобство, простота ухода, экономия времени, комфорт.
- Запрещены общие слова без конкретики и маркетинговая ложь:
  “лучший”, “идеальный”, “самый”, “№1”, “премиальный”, “эксклюзивный”.
- Запрещены сравнения: “лучше”, “быстрее”, “чем другие”.

ОГРАНИЧЕНИЯ:
- Не повторяем title или subtitle.
- Не повторяем слова внутри одного UTP.
- Все 8 UTP должны быть разными по смыслу.

4) utp_3_continue, utp_4_continue, utp_5_continue — строки.

ОБЯЗАТЕЛЬНО:
- Длина строго 1–2 слова.
- Усиливают смысл соответствующего utp, но не повторяют ни одно слово из него.
- Между собой также не повторяют слова.
- Не допускаются пустые, абстрактные понятия (“качественно”, “удобно”, “надёжно”).

ВАЖНО:
1) Все поля обязательны: title, subtitle, ровно 8 utp, utp_3_continue, utp_4_continue, utp_5_continue.
2) Если данных недостаточно, используй нейтральные реалистичные формулировки без выдумок.

Входные данные:
- Название товара: {product_name}
- Свойства/описание: {product_properties}
"""


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
Сформируй красивое, плавное и содержательное описание товара (до 2000 символов) для размещения под картинками карточки.
Целевая аудитория — домохозяйки. Пиши по делу, без воды, с выгодами и мягкими триггерами.

Входные данные:
- Заголовок: {title}
- Подзаголовок: {subtitle}
- УТП (нумерованные строки):
{utp_lines}

- Характеристики товара: {specs_text}

Требования к выводу:
- Единый связный текст в несколько абзацев.
- Без списков, без нумерации, без markdown.
- Без призывов к покупке, без скидок.
- До 2000 символов.
- Не менее 1000 символов.
""".strip()
