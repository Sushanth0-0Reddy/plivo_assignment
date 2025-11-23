import argparse
import json
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, List, Tuple

from faker import Faker


DIGIT_WORDS = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}


def random_digit_word(ch: str) -> str:
    if random.random() < 0.5:
        return ch
    if random.random() < 0.1:
        return "oh" if ch == "0" else DIGIT_WORDS[ch]
    return DIGIT_WORDS[ch]


def speak_number_string(number: str) -> str:
    tokens = []
    for ch in number:
        if ch == " ":
            continue
        tokens.append(random_digit_word(ch))
    return " ".join(tokens)


def stt_name(fake: Faker) -> str:
    name = fake.name()
    name = name.replace(".", "").replace("-", " ")
    return " ".join(name.lower().split())


def stt_email(name: str, fake: Faker) -> str:
    if random.random() < 0.6:
        first = name.split()[0]
        last = name.split()[-1]
        domain = random.choice(["gmail.com", "outlook.com", "yahoo.com", "mail.com"])
        email = f"{first}.{last}@{domain}"
    else:
        email = fake.email()
    email = email.lower()
    return email.replace("@", " at ").replace(".", " dot ")


def group_digits(digits: str, chunk: int = 4) -> str:
    digits = "".join(ch for ch in digits if ch.isdigit())
    return " ".join(digits[i : i + chunk] for i in range(0, len(digits), chunk))


def stt_credit_card(fake: Faker) -> str:
    raw = fake.credit_card_number()
    grouped = group_digits(raw)
    if random.random() < 0.5:
        return speak_number_string(grouped)
    return grouped


def stt_phone(fake: Faker) -> str:
    raw = fake.msisdn()[:10]
    if random.random() < 0.5:
        return speak_number_string(raw)
    segmented = " ".join(raw)
    return segmented


def stt_city(fake: Faker) -> str:
    return fake.city().lower()


def stt_location(fake: Faker) -> str:
    place = random.choice(
        [
            "central mall",
            "main street",
            "river view park",
            "seventh avenue",
            "old town square",
            "lotus tech park",
            "grand plaza",
            "airport road",
        ]
    )
    if random.random() < 0.4:
        city = stt_city(fake)
        return f"{place} {city}"
    return place


MONTH_WORDS = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
]


def stt_date(fake: Faker) -> str:
    date = datetime.today() - timedelta(days=random.randint(0, 365))
    if random.random() < 0.5:
        return date.strftime("%d %m %Y")
    day = str(date.day)
    year = str(date.year)
    month = MONTH_WORDS[date.month - 1]
    if random.random() < 0.5:
        day_spoken = speak_number_string(day.zfill(2))
        year_spoken = speak_number_string(year)
        return f"{day_spoken} {month} {year_spoken}"
    return f"{day} {month} {year}"


FILLERS = [
    "uh",
    "hmm",
    "like",
    "you know",
    "basically",
    "so",
]


class UtteranceBuilder:
    def __init__(self):
        self.parts: List[str] = []
        self.cursor = 0
        self.entities: List[dict] = []

    def add(self, segment: str, label: str = None):
        if not segment:
            return
        if self.parts:
            self.parts.append(" ")
            self.cursor += 1
        start = self.cursor
        self.parts.append(segment)
        self.cursor += len(segment)
        if label:
            self.entities.append({"start": start, "end": self.cursor, "label": label})

    def text(self) -> str:
        return "".join(self.parts)


def maybe_add_filler(builder: UtteranceBuilder):
    if random.random() < 0.3:
        builder.add(random.choice(FILLERS))


TemplateFn = Callable[[Faker], Tuple[str, List[dict]]]


def template_card_email_name(fake: Faker):
    builder = UtteranceBuilder()
    name = stt_name(fake)
    email = stt_email(name, fake)
    card = stt_credit_card(fake)
    builder.add("my credit card number is")
    builder.add(card, "CREDIT_CARD")
    maybe_add_filler(builder)
    builder.add("and email is")
    builder.add(email, "EMAIL")
    builder.add("name on the card is")
    builder.add(name, "PERSON_NAME")
    return builder.text(), builder.entities


def template_phone_city_date(fake: Faker):
    builder = UtteranceBuilder()
    phone = stt_phone(fake)
    city = stt_city(fake)
    date = stt_date(fake)
    builder.add("call me on")
    builder.add(phone, "PHONE")
    maybe_add_filler(builder)
    builder.add("i am calling from")
    builder.add(city, "CITY")
    builder.add("and i will travel on")
    builder.add(date, "DATE")
    return builder.text(), builder.entities


def template_email_only(fake: Faker):
    builder = UtteranceBuilder()
    name = stt_name(fake)
    email = stt_email(name, fake)
    builder.add("email id is")
    builder.add(email, "EMAIL")
    builder.add("person name")
    builder.add(name, "PERSON_NAME")
    return builder.text(), builder.entities


def template_location_trip(fake: Faker):
    builder = UtteranceBuilder()
    location = stt_location(fake)
    date = stt_date(fake)
    builder.add("meeting location will be")
    builder.add(location, "LOCATION")
    maybe_add_filler(builder)
    builder.add("on")
    builder.add(date, "DATE")
    return builder.text(), builder.entities


def template_card_phone(fake: Faker):
    builder = UtteranceBuilder()
    card = stt_credit_card(fake)
    phone = stt_phone(fake)
    builder.add("card digits are")
    builder.add(card, "CREDIT_CARD")
    builder.add("reach me at number")
    builder.add(phone, "PHONE")
    return builder.text(), builder.entities


def template_name_email_phone(fake: Faker):
    builder = UtteranceBuilder()
    name = stt_name(fake)
    email = stt_email(name, fake)
    phone = stt_phone(fake)
    builder.add("this is")
    builder.add(name, "PERSON_NAME")
    builder.add("send note to")
    builder.add(email, "EMAIL")
    builder.add("or ping")
    builder.add(phone, "PHONE")
    return builder.text(), builder.entities


def template_city_location(fake: Faker):
    builder = UtteranceBuilder()
    city = stt_city(fake)
    location = stt_location(fake)
    builder.add("currently staying in")
    builder.add(city, "CITY")
    builder.add("near")
    builder.add(location, "LOCATION")
    return builder.text(), builder.entities


TEMPLATES: List[TemplateFn] = [
    template_card_email_name,
    template_phone_city_date,
    template_email_only,
    template_location_trip,
    template_card_phone,
    template_name_email_phone,
    template_city_location,
]


@dataclass
class DatasetConfig:
    train_size: int
    dev_size: int
    test_size: int
    output_dir: Path
    train_seed: int = 13
    dev_seed: int = 42
    test_seed: int = 84


def generate_samples(fake: Faker, size: int) -> List[Tuple[str, List[dict]]]:
    samples = []
    for _ in range(size):
        template = random.choice(TEMPLATES)
        samples.append(template(fake))
    return samples


def generate_split(size: int, seed: int, prefix: str, include_entities: bool = True):
    random.seed(seed)
    fake = Faker()
    fake.seed_instance(seed)

    records = []
    for idx, (text, entities) in enumerate(generate_samples(fake, size)):
        record = {"id": f"{prefix}_{idx:04d}", "text": text}
        if include_entities:
            record["entities"] = entities
        records.append(record)
    return records


def write_jsonl(path: Path, rows: List[dict]):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")


def build_dataset(cfg: DatasetConfig):
    train_records = generate_split(cfg.train_size, cfg.train_seed, "utt_train", include_entities=True)
    dev_records = generate_split(cfg.dev_size, cfg.dev_seed, "utt_dev", include_entities=True)
    test_records = generate_split(cfg.test_size, cfg.test_seed, "utt_test", include_entities=False)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(cfg.output_dir / "train.jsonl", train_records)
    write_jsonl(cfg.output_dir / "dev.jsonl", dev_records)
    write_jsonl(cfg.output_dir / "test.jsonl", test_records)

    print(
        f"Wrote {len(train_records)} train, {len(dev_records)} dev and {len(test_records)} test samples to {cfg.output_dir}"
    )


def parse_args():
    ap = argparse.ArgumentParser(description="Generate synthetic STT-style PII datasets.")
    ap.add_argument("--train_size", type=int, default=900)
    ap.add_argument("--dev_size", type=int, default=150)
    ap.add_argument("--test_size", type=int, default=150)
    ap.add_argument("--out_dir", type=Path, default=Path("data"))
    ap.add_argument("--train_seed", type=int, default=13)
    ap.add_argument("--dev_seed", type=int, default=42)
    ap.add_argument("--test_seed", type=int, default=84)
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = DatasetConfig(
        train_size=args.train_size,
        dev_size=args.dev_size,
        test_size=args.test_size,
        output_dir=args.out_dir,
        train_seed=args.train_seed,
        dev_seed=args.dev_seed,
        test_seed=args.test_seed,
    )
    build_dataset(cfg)


if __name__ == "__main__":
    main()

