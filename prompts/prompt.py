PROMPT_DETECT_OBJECT = (
    "Ты — визуальный классификатор изображений."
    "Посмотри на картинку и определи, какой объект изображён."
    "Опиши фотографию, что на ней находится"
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
Верни только нужное значение.
""".strip()


PROMPT_GENERATE_IMAGE_L_1 = """
Replace the background with a cozy holiday kitchen in light beige tones.
Include soft warm bokeh from garland lights, blurred white cabinets, and subtle festive décor.
The background MUST be strongly Christmas-themed: ornaments, warm lights, evergreen branches.
Do not alter the shape of {product_name}. Remove all text.
"""

PROMPT_GENERATE_IMAGE_L_2 = """
Replace the background with a warm festive living room in a cream-gold palette.
Use blurred Christmas trees, golden garland bokeh, beige walls, and soft evening light.
The background MUST be strongly Christmas-themed and decorative.
Do not change the original form of {product_name}. Remove all text.
"""

PROMPT_GENERATE_IMAGE_L_3 = """
Replace the background with a bright minimalist kitchen with soft daylight and clean beige-cream tones.
Add subtle golden garland bokeh on evergreen branches and a modern Christmas aesthetic.
The background MUST be strongly Christmas-themed.
Do not modify the shape of {product_name}. Remove all text.
"""

PROMPT_GENERATE_IMAGE_L_4 = """
Replace the background with a warm sunlit kitchen: green matte cabinets, wooden countertops, and rustic elements.
Add two blurred Christmas trees and golden bokeh from holiday garlands.
The background MUST be strongly Christmas-themed and cozy.
Do not alter the original shape of {product_name}. Remove all text.
"""


PROMPT_GENERATE_IMAGE_M_1 = """
Replace the background with a top-down cozy holiday scene.
Use warm soft light on a wooden surface with: a dark green linen napkin, pine branch, cones, a golden jingle bell, and a wooden star.
The background MUST be strongly Christmas-themed.
Do not modify {product_name}. You may add one small food item or utensil, but do not cover the product. Remove all text.
"""

PROMPT_GENERATE_IMAGE_M_2 = """
Replace the background with a modern fine-dining setting.
Use a dark wooden table, soft side light, shallow depth of field; two red wine glasses, a gray linen napkin, matte black utensils, and a concrete stand with porous black stones.
Ensure a subtle Christmas atmosphere through warm festive accents.
Do not modify {product_name}. You may add one small food item or utensil, but do not cover the product. Remove all text.
"""

PROMPT_GENERATE_IMAGE_M_3 = """
Replace the background with a minimalist fine-dining flat lay.
Use a white linen tablecloth, rose-gold utensils, a gray napkin, a small matte graphite plate with an olive branch, and smoky-pink glasses.
Soft diffused daylight and a muted neutral palette are required; add gentle Christmas elements.
Do not modify {product_name}. You may add one small food item or utensil, but do not cover the product. Remove all text.
"""

PROMPT_GENERATE_IMAGE_M_4 = """
Replace the background with a cozy festive living room.
Include a lit fireplace with a pine garland, a blurred Christmas tree with golden bokeh lights, and warm evening illumination on cream-beige walls.
The background MUST be strongly Christmas-themed.
Do not modify {product_name}. You may add one small food item or utensil, but do not cover the product. Remove all text.
"""


PROMPT_GENERATE_IMAGE_S_1 = """
Replace the background with a cozy festive kitchen in warm evening tones and shallow depth of field.
Include golden garland bokeh on evergreen branches, softly blurred white cabinets and stove, and a red Christmas stocking on the wall.
Use a cream-beige palette with soft diffused warm light. Strong Christmas atmosphere required.
Do not change or extend {product_name}. Remove all text.
"""

PROMPT_GENERATE_IMAGE_S_2 = """
Replace the background with a hygge-inspired living room with fine bokeh.
Use a light-wood round coffee table with lit candles in glass holders; in the back, a blurred stone-faced fireplace with a pine garland and a blurred Christmas tree with golden bokeh.
Combine warm firelight with soft daylight from the right. Calm minimalist Christmas mood.
Do not change or extend {product_name}. Remove all text.
"""

PROMPT_GENERATE_IMAGE_S_3 = """
Replace the background with a soft-bokeh modern kitchen: matte black shaker cabinets, white countertop, white subway tiles, a black curved faucet, and a built-in oven with metal accents.
Add gentle daylight from the left and subtle golden garland bokeh on evergreen branches.
Ensure a clean, modern, Christmas-themed atmosphere.
Do not change or extend {product_name}. Remove all text.
"""

PROMPT_GENERATE_IMAGE_S_4 = """
Replace the background with a festive kitchen in soft bokeh: green cabinets, a cream countertop, a pine garland, and a blurred Christmas tree with golden bokeh and red ornaments.
Use warm evening light and a calm cream-green palette with shallow depth of field.
The background MUST be distinctly Christmas-themed.
Do not change or extend {product_name}. Remove all text.
"""


PROMPT_GENERATE_ADDITIONAL_ANGLES = """
Replace the background with one of the following kitchen styles and show {product_name} from a random angle (side, top, or low angle):
- modern white cabinets
- natural materials with matte green cabinets
- modern kitchen with matte burgundy cabinets

The background MUST be strongly Christmas-themed: rich holiday decorations, garlands, warm lights, Christmas tree elements, or festive ornaments. This requirement is mandatory.

Do not change or reconstruct the product.
"""


PROMPT_VALIDATE_IMAGE_VL = """
Describe the image focusing on technical details.

Required product: {product_name}

Describe:
1) Is the product visible and recognizable?
2) Does the product look complete or is it cut off or heavily distorted?
3) Are there any AI artifacts: glitches, holes, warped geometry, duplicated parts, missing texture?
4) What does the background look like: kitchen interior, table, room, plain solid color, very dark, etc.?

Do not judge aesthetics or composition. Just describe what you see.
"""


PROMPT_VALIDATE_IMAGE_QUALITY = """
Ты оцениваешь техническую пригодность фото товара для карточки маркетплейса.

Товар: "{product_name}"

Описание изображения от vision-модели:
{vision_raw}

Реши, подходит ли это фото для карточки товара по ТЕХНИЧЕСКИМ критериям.

Фото считается OK, если одновременно выполняются условия:
1) Товар виден отчётливо и различим (можно понять, что это за предмет).
2) Товар не сильно обрезан, не заменён другим объектом и не искажён.
3) Нет явных артефактов генерации: глитчи, “дыры”, странные искажённые части, разорванная геометрия, явные ошибки фона.
4) Фон может быть кухней, столом, интерьером — это НОРМАЛЬНО и ДОПУСТИМО.

Фото считается BAD, только если:
1) Товар отсутствует или почти не виден,
2) Товар сильно обрезан, распался, заменён или невозможно понять, что это он,
3) Есть явные серьёзные артефакты генерации (поломанные части, сильно испорченный фон, “дырки”, ломанный объект),
4) Фон полностью пустой, однотонный, полностью чёрный/очень тёмный или выглядит как явная ошибка генерации.

НЕ считай обычный кухонный интерьер, стол, посуду или декор ошибкой или отвлекающим фоном — это допустимо для карточки товара.

Ответ верни строго в виде:
status = "OK" или "BAD"
reason = краткое объяснение одной строкой.
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
Сформируй для неё title, subtitle и UTP (уникальные торговые преимущества) на основе характеристик товара так, чтобы, увидев их, она захотела купить. Помни эти UTP потом будут наложены на картинки (слайды) с товаром в виде инфографики, поэтому опиши их, чтобы смысл преимущества был понятен носителю русского языка — фразы должны быть связными и со смыслом.

Входные данные:
- Название товара: {product_name}
- Свойства/описание: {product_properties}

1) title — строка.
   - Ровно ОДНО слово — базовое наименование предмета.
   - Существительное в именительном падеже.
   - Без брендов, размеров, кавычек, чисел и дополнительных слов.
   Примеры: "Кастрюля", "Сковорода", "Чайник".

2) subtitle — строка.
   - "Бренд и серия" в кавычках + размер/объём, если есть.
   Примеры: "\"Grano Elite\", 24 см", "\"Current\", 2.5 л".

3) utp — массив из РОВНО 8 объектов: {{"number": <1..8>, "text": "<строка>"}}.

Правила для поля text:

ОБЯЗАТЕЛЬНО:
- Длина строго 2-4 слова.
- Никаких пояснений, нумерации, двоеточий, подзаголовков. 
- Строго одна связная фраза.

СМЫСЛ:
- Преимущество должно быть понятным русской домохозяйке («что облегчает?», «что экономит?», «что упрощает уход?»).
- Запрещены: штампы, общие слова без конкретики, абстракции.
- Запрещена маркетинговая ложь: “лучший”, “идеальный”, “самый”, “№1”, “премиальный”, “эксклюзивный”.
- Запрещены сравнения: “лучше”, “быстрее”, “выше”, “чем другие”.

ОГРАНИЧЕНИЯ:
- НЕ повторяем title или subtitle.
- НЕ повторяем одни и те же слова в пределах одного UTP.
- Все 8 UTP ДОЛЖНЫ быть разными по смыслу.

4) utp_3_continue, utp_4_continue, utp_5_continue — строки.

ОБЯЗАТЕЛЬНО:
- Длина строго 1–2 слова. Если фраза содержит >2 слов — она считается НЕДОПУСТИМОЙ и должна быть пересоздана.
- Продолжение — это короткое уточнение, которое усиливает смысл UTP, но НЕ дублирует его содержательно.
- Продолжение НЕ может повторять НИ ОДНО слово из соответствующего UTP (даже в другой форме).
- НЕЛЬЗЯ использовать пустые абстракции (“качественно”, “удобно”, “надёжно”, “красиво”, “хорошо”).
- Все три продолжения должны различаться по смыслу между собой.

ВАЖНО:
1) Все поля (title, subtitle, все 8 utp, utp_3_continue, utp_4_continue, utp_5_continue) обязательны.
2) Никаких брендов в title. Бренд — только в subtitle.
3) Никаких повторов UTP между собой и внутри одного UTP.
4) Продолжения (utp_3_continue, utp_4_continue, utp_5_continue) НЕ повторяют слова UTP и НЕ дублируют друг друга.
5) Если данных мало, используй нейтральные, правдоподобные формулировки без выдуманных фактов.

Пример целевого формата (ОБРАЗЕЦ СТРУКТУРЫ, а не подсказка по содержанию, НЕЛЬЗЯ копировать текст содержательно):

{{
  "title": "Только базовое наименование (одно существительное, например \"Кастрюля\")",
  "subtitle": "Бренд и серия в кавычках, с размером или объёмом, если есть",
  "utp": [
    {{"number": 1, "text": "UTP-1"}},
    {{"number": 2, "text": "UTP-2"}},
    {{"number": 3, "text": "UTP-3"}},
    {{"number": 4, "text": "UTP-4"}},
    {{"number": 5, "text": "UTP-5"}},
    {{"number": 6, "text": "UTP-6"}},
    {{"number": 7, "text": "UTP-7"}},
    {{"number": 8, "text": "UTP-8"}}
  ],
  "utp_3_continue": "Пример продолжения для UTP-3",
  "utp_4_continue": "Пример продолжения для UTP-4",
  "utp_5_continue": "Пример продолжения для UTP-5"
}}
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
