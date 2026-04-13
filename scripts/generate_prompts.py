"""Generate CoT and anchor prompt JSONL datasets for alignment calibration."""
from __future__ import annotations

import json
import random
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "data"
COT_PATH = OUT_DIR / "cot_prompts.jsonl"
ANCHOR_PATH = OUT_DIR / "anchor_prompts.jsonl"

RNG = random.Random(42)

NAMES = ["Alice", "Bob", "Carla", "Diego", "Emma", "Farid", "Grace", "Hiro",
         "Ines", "Jamal", "Kira", "Luis", "Mina", "Noah", "Omar", "Priya"]
ITEMS = ["apples", "pencils", "books", "marbles", "cookies", "stickers",
         "oranges", "coins", "cards", "stamps"]
SHOPS = ["bakery", "market", "bookstore", "cafe", "grocer"]


def cot_arithmetic() -> str:
    a, b, c = RNG.randint(3, 40), RNG.randint(2, 25), RNG.randint(1, 15)
    n1, n2 = RNG.sample(NAMES, 2)
    item = RNG.choice(ITEMS)
    return (f"Think step by step. {n1} has {a} {item}. {n2} gives {n1} "
            f"{b} more, then {n1} gives away {c}. How many {item} does "
            f"{n1} have now?")


def cot_multiply() -> str:
    boxes = RNG.randint(3, 12)
    per = RNG.randint(4, 20)
    taken = RNG.randint(1, per * boxes // 2)
    item = RNG.choice(ITEMS)
    return (f"Think step by step. A {RNG.choice(SHOPS)} has {boxes} boxes "
            f"with {per} {item} each. If {taken} {item} are sold, how many "
            f"remain?")


def cot_rate() -> str:
    rate = RNG.randint(2, 15)
    hours = RNG.randint(2, 10)
    return (f"Think step by step. A worker produces {rate} units per hour. "
            f"How many units are produced in {hours} hours, and how long "
            f"would it take to produce {rate * hours * 2} units?")


def cot_algebra_linear() -> str:
    x = RNG.randint(2, 25)
    a = RNG.randint(2, 9)
    b = RNG.randint(1, 20)
    result = a * x + b
    return (f"Think step by step. Solve for x: {a}x + {b} = {result}.")


def cot_algebra_system() -> str:
    x, y = RNG.randint(1, 12), RNG.randint(1, 12)
    s = x + y
    d = x - y
    return (f"Think step by step. Two numbers sum to {s} and their "
            f"difference is {d}. What are the numbers?")


def cot_percent() -> str:
    total = RNG.choice([50, 80, 120, 200, 250, 400])
    pct = RNG.choice([10, 15, 20, 25, 30, 40])
    return (f"Think step by step. A shirt costs ${total}. It is discounted "
            f"by {pct}%. What is the final price?")


def cot_fraction() -> str:
    total = RNG.choice([12, 18, 24, 30, 36, 48])
    num = RNG.randint(1, 3)
    den = RNG.choice([3, 4, 6])
    n = RNG.choice(NAMES)
    item = RNG.choice(ITEMS)
    return (f"Think step by step. {n} has {total} {item} and gives away "
            f"{num}/{den} of them. How many {item} does {n} keep?")


def cot_logic_age() -> str:
    age = RNG.randint(8, 40)
    years = RNG.randint(3, 20)
    n1, n2 = RNG.sample(NAMES, 2)
    return (f"Think step by step. {n1} is {age} years old. {n2} is "
            f"{years} years younger. In {years} years, how old will "
            f"{n2} be?")


def cot_logic_seq() -> str:
    start = RNG.randint(1, 10)
    step = RNG.randint(2, 7)
    terms = [start + step * i for i in range(4)]
    return (f"Think step by step. What comes next in the sequence "
            f"{terms[0]}, {terms[1]}, {terms[2]}, {terms[3]}, ...?")


def cot_logic_cmp() -> str:
    n1, n2, n3 = RNG.sample(NAMES, 3)
    return (f"Think step by step. {n1} is taller than {n2}. {n2} is taller "
            f"than {n3}. Who is the shortest, and who is the tallest?")


COT_GENERATORS = [
    cot_arithmetic, cot_arithmetic, cot_multiply, cot_rate,
    cot_algebra_linear, cot_algebra_system, cot_percent, cot_fraction,
    cot_logic_age, cot_logic_seq, cot_logic_cmp,
]


ANCHOR_TEMPLATES = [
    "Summarize the water cycle in one sentence.",
    "List three common programming languages.",
    "Define the word 'ephemeral'.",
    "Translate 'good morning' into Spanish.",
    "Name the capital of France.",
    "Give one example of a renewable energy source.",
    "State Newton's first law briefly.",
    "What is the boiling point of water in Celsius?",
    "Name a primary color.",
    "Write a haiku about autumn.",
    "Explain what a variable is in programming.",
    "Give the plural of 'mouse' (the animal).",
    "List two benefits of regular exercise.",
    "Describe what a noun is.",
    "Name one planet in our solar system.",
    "What does HTML stand for?",
    "Give an antonym of 'happy'.",
    "Name a famous painter.",
    "Explain photosynthesis in one sentence.",
    "State the Pythagorean theorem.",
    "Name three countries in Europe.",
    "Give one synonym of 'quick'.",
    "What is the chemical symbol for gold?",
    "Name a common kitchen utensil.",
    "Describe the color blue in one sentence.",
    "What is 7 times 8?",
    "Name one ocean.",
    "Give the past tense of 'run'.",
    "Name a musical instrument with strings.",
    "What is the tallest mountain on Earth?",
    "List two fruits that grow on trees.",
    "Define 'algorithm' in one sentence.",
    "Name a mammal that lives in the ocean.",
    "Give an example of a prime number.",
    "What does CPU stand for?",
    "Name a Shakespeare play.",
    "Give a word that rhymes with 'cloud'.",
    "State one use of a compass.",
    "Name the currency used in Japan.",
    "Define 'gravity' in one sentence.",
    "List three shades of green.",
    "Name a common web browser.",
    "What is the square root of 81?",
    "Give one example of a polygon.",
    "Name a dairy product.",
    "State the freezing point of water in Fahrenheit.",
    "Name a famous scientist.",
    "Give one word meaning 'happy'.",
    "Describe the shape of a stop sign.",
    "Name a bird that cannot fly.",
]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    cot_records = []
    seen: set[str] = set()
    while len(cot_records) < 100:
        gen = RNG.choice(COT_GENERATORS)
        text = gen()
        if text in seen:
            continue
        seen.add(text)
        cot_records.append({"text": text})

    anchor_records = [{"text": t} for t in ANCHOR_TEMPLATES[:50]]

    write_jsonl(COT_PATH, cot_records)
    write_jsonl(ANCHOR_PATH, anchor_records)

    print(f"{COT_PATH}: {len(cot_records)} records")
    print(f"{ANCHOR_PATH}: {len(anchor_records)} records")


if __name__ == "__main__":
    main()
