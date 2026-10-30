"""Frozen 20 tool prompts and three continuations from the historical evaluation.

Paths and service names below are inert model input. No tool is ever executed.
"""

def fn(name, desc, props, req):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": req}}}

TOOLS = [
    fn("get_weather", "Текущая погода в городе", {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, ["city"]),
    fn("convert_currency", "Перевести сумму из одной валюты в другую", {"amount": {"type": "number"}, "from": {"type": "string", "description": "код ISO, например USD"}, "to": {"type": "string", "description": "код ISO"}}, ["amount", "from", "to"]),
    fn("create_event", "Создать событие в календаре", {"title": {"type": "string"}, "date": {"type": "string", "description": "YYYY-MM-DD"}, "time": {"type": "string", "description": "HH:MM"}, "duration_minutes": {"type": "integer"}}, ["title", "date", "time"]),
    fn("run_shell", "Выполнить команду в bash на сервере", {"command": {"type": "string"}}, ["command"]),
    fn("search_files", "Найти файлы по шаблону имени в каталоге", {"pattern": {"type": "string"}, "path": {"type": "string"}}, ["pattern", "path"]),
    fn("send_message", "Отправить сообщение в Telegram", {"to": {"type": "string"}, "text": {"type": "string"}}, ["to", "text"]),
]

def has(v, *subs):
    v = str(v).lower()
    return all(s.lower() in v for s in subs)

CASES = [
    ("Какая сейчас погода в Казани?", "get_weather", lambda a: has(a.get("city"), "казан") or has(a.get("city"), "kazan")),
    ("What's the weather in Berlin in fahrenheit?", "get_weather", lambda a: has(a.get("city"), "berlin") and a.get("unit") == "fahrenheit"),
    ("Сколько будет 250 долларов в евро?", "convert_currency", lambda a: float(a.get("amount", 0)) == 250 and has(a.get("from"), "usd") and has(a.get("to"), "eur")),
    ("Переведи 1 500 000 рублей в юани", "convert_currency", lambda a: float(a.get("amount", 0)) == 1500000 and has(a.get("from"), "rub") and has(a.get("to"), "cny")),
    ("Поставь встречу «Созвон с Пашей» на 14 октября 2026 в 15:30 на 45 минут", "create_event", lambda a: a.get("date") == "2026-10-14" and a.get("time") == "15:30" and int(a.get("duration_minutes", 0)) == 45 and has(a.get("title"), "паш")),
    ("Создай событие 'Замена БП' 2026-11-02 в 09:00", "create_event", lambda a: a.get("date") == "2026-11-02" and a.get("time") == "09:00" and has(a.get("title"), "бп")),
    ("Покажи, сколько места свободно на /mnt/ai", "run_shell", lambda a: has(a.get("command"), "df") and has(a.get("command"), "/mnt/ai")),
    ("Посмотри последние 50 строк журнала юнита pearl-miner", "run_shell", lambda a: has(a.get("command"), "journalctl", "pearl-miner") and "50" in str(a.get("command"))),
    ("Проверь, запущен ли сервис keryx-miner", "run_shell", lambda a: has(a.get("command"), "keryx-miner") and (has(a.get("command"), "systemctl") or has(a.get("command"), "pgrep") or has(a.get("command"), "ps"))),
    ("Найди все файлы *.gguf в /mnt/ai/models", "search_files", lambda a: has(a.get("pattern"), ".gguf") and has(a.get("path"), "/mnt/ai/models")),
    ("Find all python files in /home/nekto/scripts", "search_files", lambda a: has(a.get("pattern"), ".py") and has(a.get("path"), "/home/nekto/scripts")),
    ("Напиши Андрею в телеграм: майнер перезапущен, всё в порядке", "send_message", lambda a: has(a.get("to"), "андре") and has(a.get("text"), "майнер")),
    ("Send a telegram message to @devops saying 'backup finished'", "send_message", lambda a: has(a.get("to"), "devops") and has(a.get("text"), "backup")),
    ("Сколько будет 17 умножить на 23?", None, None),
    ("Объясни коротко, что такое MoE в нейросетях", None, None),
    ("Привет! Как дела?", None, None),
    ("Какая погода в Токио и сколько это в фаренгейтах?", "get_weather", lambda a: has(a.get("city"), "токио") or has(a.get("city"), "tokyo")),
    ("Курс: 99.5 евро в фунты стерлингов", "convert_currency", lambda a: float(a.get("amount", 0)) == 99.5 and has(a.get("from"), "eur") and has(a.get("to"), "gbp")),
    ("Перезапусти юнит llm-duty", "run_shell", lambda a: has(a.get("command"), "systemctl", "restart", "llm-duty")),
    ("Найди логи (*.log) в /mnt/ai/logs", "search_files", lambda a: has(a.get("pattern"), ".log") and has(a.get("path"), "/mnt/ai/logs")),
]

FOLLOW = [
    ("Какая погода в Омске?", "get_weather", {"city": "Омск"}, {"city": "Омск", "temp_c": -7, "condition": "снег"}, ["-7", "снег"]),
    ("Сколько будет 120 USD в EUR?", "convert_currency", {"amount": 120, "from": "USD", "to": "EUR"}, {"result": 103.38, "rate": 0.8615}, ["103.38"]),
    ("Сколько свободно на /mnt/ai?", "run_shell", {"command": "df -h /mnt/ai"}, {"stdout": "Filesystem Size Used Avail Use% Mounted on\n/dev/nvme0n1p1 3,6T 2,1T 1,5T 59% /mnt/ai", "exit_code": 0}, ["1,5", "т"]),
]
