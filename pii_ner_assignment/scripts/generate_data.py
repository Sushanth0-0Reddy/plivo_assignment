import argparse
import json
import random
import yaml
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, List, Tuple, Dict, Any

from faker import Faker


@dataclass
class STTNoiseConfig:
    """Configuration for STT noise simulation parameters."""
    digit_to_word_ratio: float = 0.5      # Probability of converting digits to words
    zero_to_oh_ratio: float = 0.1         # Probability of saying "oh" instead of "zero"
    filler_ratio: float = 0.3             # Probability of adding filler words ("uh", "hmm")
    lowercase_ratio: float = 1.0          # Probability of lowercasing names/cities
    email_semantic_link: float = 0.6      # Probability name appears in email address
    spoken_card_ratio: float = 0.5        # Probability credit card digits are spoken
    spoken_phone_ratio: float = 0.5       # Probability phone digits are spoken
    spoken_date_ratio: float = 0.5        # Probability dates are spoken vs numeric
    location_with_city_ratio: float = 0.4 # Probability location includes city name
    
    # STT spacing errors
    extra_space_ratio: float = 0.15       # Probability of inserting extra spaces ("john  smith")
    missing_space_ratio: float = 0.1      # Probability of missing spaces ("johnsmith")
    at_spacing_variation: float = 0.3     # Variable spacing around "at" in emails
    
    def to_dict(self):
        """Export config for logging."""
        return {
            "digit_to_word_ratio": self.digit_to_word_ratio,
            "zero_to_oh_ratio": self.zero_to_oh_ratio,
            "filler_ratio": self.filler_ratio,
            "lowercase_ratio": self.lowercase_ratio,
            "email_semantic_link": self.email_semantic_link,
            "spoken_card_ratio": self.spoken_card_ratio,
            "spoken_phone_ratio": self.spoken_phone_ratio,
            "spoken_date_ratio": self.spoken_date_ratio,
            "location_with_city_ratio": self.location_with_city_ratio,
            "extra_space_ratio": self.extra_space_ratio,
            "missing_space_ratio": self.missing_space_ratio,
            "at_spacing_variation": self.at_spacing_variation,
        }


# Global noise config (will be set during dataset generation)
NOISE_CONFIG = STTNoiseConfig()


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
    """Convert digit to word based on noise config."""
    if random.random() < NOISE_CONFIG.digit_to_word_ratio:
        return ch  # Keep as digit
    if random.random() < NOISE_CONFIG.zero_to_oh_ratio:
        return "oh" if ch == "0" else DIGIT_WORDS[ch]
    return DIGIT_WORDS[ch]


def speak_number_string(number: str) -> str:
    """Convert number string to spoken format with noise."""
    tokens = []
    for ch in number:
        if ch == " ":
            continue
        tokens.append(random_digit_word(ch))
    return " ".join(tokens)


def apply_spacing_noise(text: str) -> str:
    """Apply STT spacing errors to text."""
    words = text.split()
    
    # Extra space insertion (double spaces between words)
    if random.random() < NOISE_CONFIG.extra_space_ratio and len(words) > 1:
        # Insert extra space between random words
        idx = random.randint(0, len(words) - 2)
        return " ".join(words[:idx+1]) + "  " + " ".join(words[idx+1:])
    
    # Missing space (concatenate two words)
    if random.random() < NOISE_CONFIG.missing_space_ratio and len(words) > 1:
        idx = random.randint(0, len(words) - 2)
        words[idx] = words[idx] + words[idx + 1]
        del words[idx + 1]
    
    return " ".join(words)


def stt_name(fake: Faker) -> str:
    """Generate STT-style name (lowercase, no punctuation)."""
    name = fake.name()
    name = name.replace(".", "").replace("-", " ")
    # Apply lowercase based on config
    if random.random() < NOISE_CONFIG.lowercase_ratio:
        name = name.lower()
    name = " ".join(name.split())
    # Apply spacing noise
    name = apply_spacing_noise(name)
    return name


def stt_email(name: str, fake: Faker) -> str:
    """Generate STT-style email with semantic link to name."""
    # Semantic consistency: email matches name
    if random.random() < NOISE_CONFIG.email_semantic_link:
        first = name.split()[0]
        last = name.split()[-1]
        domain = random.choice(["gmail.com", "outlook.com", "yahoo.com", "mail.com"])
        email = f"{first}.{last}@{domain}"
    else:
        email = fake.email()
    
    # Apply lowercase based on config
    if random.random() < NOISE_CONFIG.lowercase_ratio:
        email = email.lower()
    
    # STT converts @ and . to words with variable spacing
    if random.random() < NOISE_CONFIG.at_spacing_variation:
        # Variable spacing: "at" could have extra spaces or be inconsistent
        at_replacement = random.choice([" at ", "  at  ", " at", "at "])
        dot_replacement = random.choice([" dot ", "  dot  ", " dot", "dot "])
    else:
        # Standard spacing
        at_replacement = " at "
        dot_replacement = " dot "
    
    email = email.replace("@", at_replacement).replace(".", dot_replacement)
    
    # Normalize multiple spaces (STT sometimes does this)
    if random.random() < 0.1:
        email = " ".join(email.split())  # Normalize all spacing
    
    return email


def group_digits(digits: str, chunk: int = 4) -> str:
    digits = "".join(ch for ch in digits if ch.isdigit())
    return " ".join(digits[i : i + chunk] for i in range(0, len(digits), chunk))


def stt_credit_card(fake: Faker) -> str:
    """Generate STT-style credit card (spoken or numeric)."""
    raw = fake.credit_card_number()
    grouped = group_digits(raw)
    if random.random() < NOISE_CONFIG.spoken_card_ratio:
        return speak_number_string(grouped)
    return grouped


def stt_phone(fake: Faker) -> str:
    """Generate STT-style phone number (spoken or spaced digits)."""
    raw = fake.msisdn()[:10]
    if random.random() < NOISE_CONFIG.spoken_phone_ratio:
        return speak_number_string(raw)
    segmented = " ".join(raw)
    return segmented


def stt_city(fake: Faker) -> str:
    """Generate STT-style city name."""
    city = fake.city()
    # Apply lowercase based on config
    if random.random() < NOISE_CONFIG.lowercase_ratio:
        city = city.lower()
    return city


def stt_location(fake: Faker) -> str:
    """Generate STT-style location (with optional city)."""
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
    if random.random() < NOISE_CONFIG.location_with_city_ratio:
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


# ========== PRESIDIO-INSPIRED TEMPLATES (Adapted for STT) ==========

def presidio_01(fake: Faker):
    """my credit card {{credit_card}} has been lost can you block it"""
    builder = UtteranceBuilder()
    card = stt_credit_card(fake)
    builder.add("my credit card")
    builder.add(card, "CREDIT_CARD")
    builder.add("has been lost can you block it")
    return builder.text(), builder.entities


def presidio_02(fake: Faker):
    """need to change billing date of my card {{credit_card}}"""
    builder = UtteranceBuilder()
    card = stt_credit_card(fake)
    builder.add("need to change billing date of my card")
    builder.add(card, "CREDIT_CARD")
    return builder.text(), builder.entities


def presidio_03(fake: Faker):
    """i have lost my card {{credit_card}} my name is {{person}}"""
    builder = UtteranceBuilder()
    card = stt_credit_card(fake)
    name = stt_name(fake)
    builder.add("i have lost my card")
    builder.add(card, "CREDIT_CARD")
    builder.add("my name is")
    builder.add(name, "PERSON_NAME")
    return builder.text(), builder.entities


def presidio_04(fake: Faker):
    """didnt get message on my registered {{phone}}"""
    builder = UtteranceBuilder()
    phone = stt_phone(fake)
    builder.add("didnt get message on my registered")
    builder.add(phone, "PHONE")
    return builder.text(), builder.entities


def presidio_05(fake: Faker):
    """send last billed amount for card {{credit_card}} to {{email}}"""
    builder = UtteranceBuilder()
    card = stt_credit_card(fake)
    name = stt_name(fake)
    email = stt_email(name, fake)
    builder.add("send last billed amount for card")
    builder.add(card, "CREDIT_CARD")
    builder.add("to")
    builder.add(email, "EMAIL")
    return builder.text(), builder.entities


def presidio_06(fake: Faker):
    """card {{credit_card}} is lost send new one i am in {{city}} for business"""
    builder = UtteranceBuilder()
    card = stt_credit_card(fake)
    city = stt_city(fake)
    builder.add("card")
    builder.add(card, "CREDIT_CARD")
    builder.add("is lost send new one i am in")
    builder.add(city, "CITY")
    builder.add("for business")
    return builder.text(), builder.entities


def presidio_07(fake: Faker):
    """please have manager call me at {{phone}}"""
    builder = UtteranceBuilder()
    phone = stt_phone(fake)
    builder.add("please have manager call me at")
    builder.add(phone, "PHONE")
    return builder.text(), builder.entities


def presidio_08(fake: Faker):
    """my name is {{person}}"""
    builder = UtteranceBuilder()
    name = stt_name(fake)
    builder.add("my name is")
    builder.add(name, "PERSON_NAME")
    return builder.text(), builder.entities


def presidio_09(fake: Faker):
    """my name is {{person}} but everyone calls me {{first_name}}"""
    builder = UtteranceBuilder()
    name = stt_name(fake)
    first = name.split()[0]
    builder.add("my name is")
    builder.add(name, "PERSON_NAME")
    builder.add("but everyone calls me")
    builder.add(first, "PERSON_NAME")
    return builder.text(), builder.entities


def presidio_10(fake: Faker):
    """whats your email {{email}}"""
    builder = UtteranceBuilder()
    name = stt_name(fake)
    email = stt_email(name, fake)
    builder.add("whats your email")
    builder.add(email, "EMAIL")
    return builder.text(), builder.entities


def presidio_11(fake: Faker):
    """whats your name {{person}}"""
    builder = UtteranceBuilder()
    name = stt_name(fake)
    builder.add("whats your name")
    builder.add(name, "PERSON_NAME")
    return builder.text(), builder.entities


def presidio_12(fake: Faker):
    """how can we reach you call {{phone}}"""
    builder = UtteranceBuilder()
    phone = stt_phone(fake)
    builder.add("how can we reach you call")
    builder.add(phone, "PHONE")
    return builder.text(), builder.entities


def presidio_13(fake: Faker):
    """my friend lives in {{city}}"""
    builder = UtteranceBuilder()
    city = stt_city(fake)
    builder.add("my friend lives in")
    builder.add(city, "CITY")
    return builder.text(), builder.entities


def presidio_14(fake: Faker):
    """we moved here from {{city}}"""
    builder = UtteranceBuilder()
    city = stt_city(fake)
    builder.add("we moved here from")
    builder.add(city, "CITY")
    return builder.text(), builder.entities


def presidio_15(fake: Faker):
    """please send portfolio to {{email}}"""
    builder = UtteranceBuilder()
    name = stt_name(fake)
    email = stt_email(name, fake)
    builder.add("please send portfolio to")
    builder.add(email, "EMAIL")
    return builder.text(), builder.entities


def presidio_16(fake: Faker):
    """name {{person}} phone {{phone}}"""
    builder = UtteranceBuilder()
    name = stt_name(fake)
    phone = stt_phone(fake)
    builder.add("name")
    builder.add(name, "PERSON_NAME")
    builder.add("phone")
    builder.add(phone, "PHONE")
    return builder.text(), builder.entities


def presidio_17(fake: Faker):
    """can someone call me on {{phone}}"""
    builder = UtteranceBuilder()
    phone = stt_phone(fake)
    builder.add("can someone call me on")
    builder.add(phone, "PHONE")
    return builder.text(), builder.entities


def presidio_18(fake: Faker):
    """i would like to stop receiving messages to {{phone}}"""
    builder = UtteranceBuilder()
    phone = stt_phone(fake)
    builder.add("i would like to stop receiving messages to")
    builder.add(phone, "PHONE")
    return builder.text(), builder.entities


def presidio_19(fake: Faker):
    """please charge my credit card number is {{credit_card}}"""
    builder = UtteranceBuilder()
    card = stt_credit_card(fake)
    builder.add("please charge my credit card number is")
    builder.add(card, "CREDIT_CARD")
    return builder.text(), builder.entities


def presidio_20(fake: Faker):
    """i want to cancel my card {{credit_card}} because i lost it"""
    builder = UtteranceBuilder()
    card = stt_credit_card(fake)
    builder.add("i want to cancel my card")
    builder.add(card, "CREDIT_CARD")
    builder.add("because i lost it")
    return builder.text(), builder.entities


def presidio_21(fake: Faker):
    """im in {{city}} at the conference"""
    builder = UtteranceBuilder()
    city = stt_city(fake)
    builder.add("im in")
    builder.add(city, "CITY")
    builder.add("at the conference")
    return builder.text(), builder.entities


def presidio_22(fake: Faker):
    """hi my card {{credit_card}} was declined call {{phone}}"""
    builder = UtteranceBuilder()
    card = stt_credit_card(fake)
    phone = stt_phone(fake)
    builder.add("hi my card")
    builder.add(card, "CREDIT_CARD")
    builder.add("was declined call")
    builder.add(phone, "PHONE")
    return builder.text(), builder.entities


def presidio_23(fake: Faker):
    """change my email from {{email}} to {{email}}"""
    builder = UtteranceBuilder()
    name1 = stt_name(fake)
    name2 = stt_name(fake)
    email1 = stt_email(name1, fake)
    email2 = stt_email(name2, fake)
    builder.add("change my email from")
    builder.add(email1, "EMAIL")
    builder.add("to")
    builder.add(email2, "EMAIL")
    return builder.text(), builder.entities


def presidio_24(fake: Faker):
    """call {{phone}} or {{phone}}"""
    builder = UtteranceBuilder()
    phone1 = stt_phone(fake)
    phone2 = stt_phone(fake)
    builder.add("call")
    builder.add(phone1, "PHONE")
    builder.add("or")
    builder.add(phone2, "PHONE")
    return builder.text(), builder.entities


def presidio_25(fake: Faker):
    """this is {{person}} send note to {{email}}"""
    builder = UtteranceBuilder()
    name1 = stt_name(fake)
    name2 = stt_name(fake)
    email = stt_email(name2, fake)
    builder.add("this is")
    builder.add(name1, "PERSON_NAME")
    builder.add("send note to")
    builder.add(email, "EMAIL")
    return builder.text(), builder.entities


def presidio_26(fake: Faker):
    """visiting {{city}} next {{date}}"""
    builder = UtteranceBuilder()
    city = stt_city(fake)
    date = stt_date(fake)
    builder.add("visiting")
    builder.add(city, "CITY")
    builder.add("next")
    builder.add(date, "DATE")
    return builder.text(), builder.entities


def presidio_27(fake: Faker):
    """meet me at {{location}}"""
    builder = UtteranceBuilder()
    location = stt_location(fake)
    builder.add("meet me at")
    builder.add(location, "LOCATION")
    return builder.text(), builder.entities


def presidio_28(fake: Faker):
    """restaurant is at {{location}} in {{city}}"""
    builder = UtteranceBuilder()
    location = stt_location(fake)
    city = stt_city(fake)
    builder.add("restaurant is at")
    builder.add(location, "LOCATION")
    builder.add("in")
    builder.add(city, "CITY")
    return builder.text(), builder.entities


def presidio_29(fake: Faker):
    """email {{person}} at {{email}} card {{credit_card}}"""
    builder = UtteranceBuilder()
    name = stt_name(fake)
    email = stt_email(name, fake)
    card = stt_credit_card(fake)
    builder.add("email")
    builder.add(name, "PERSON_NAME")
    builder.add("at")
    builder.add(email, "EMAIL")
    builder.add("card")
    builder.add(card, "CREDIT_CARD")
    return builder.text(), builder.entities


def presidio_30(fake: Faker):
    """please block card no {{credit_card}}"""
    builder = UtteranceBuilder()
    card = stt_credit_card(fake)
    builder.add("please block card no")
    builder.add(card, "CREDIT_CARD")
    return builder.text(), builder.entities


# ========== ORIGINAL TEMPLATES ==========

def template_phone_only(fake: Faker):
    builder = UtteranceBuilder()
    phone = stt_phone(fake)
    builder.add("contact number is")
    builder.add(phone, "PHONE")
    return builder.text(), builder.entities


def template_date_location(fake: Faker):
    builder = UtteranceBuilder()
    date = stt_date(fake)
    location = stt_location(fake)
    builder.add("scheduled for")
    builder.add(date, "DATE")
    builder.add("at")
    builder.add(location, "LOCATION")
    return builder.text(), builder.entities


def template_card_only(fake: Faker):
    builder = UtteranceBuilder()
    card = stt_credit_card(fake)
    builder.add("card digits are")
    builder.add(card, "CREDIT_CARD")
    return builder.text(), builder.entities


def template_email_phone(fake: Faker):
    builder = UtteranceBuilder()
    name = stt_name(fake)
    email = stt_email(name, fake)
    phone = stt_phone(fake)
    builder.add("contact")
    builder.add(email, "EMAIL")
    builder.add("phone")
    builder.add(phone, "PHONE")
    return builder.text(), builder.entities


def template_name_only(fake: Faker):
    builder = UtteranceBuilder()
    name = stt_name(fake)
    builder.add("customer name")
    builder.add(name, "PERSON_NAME")
    return builder.text(), builder.entities


def template_city_date(fake: Faker):
    builder = UtteranceBuilder()
    city = stt_city(fake)
    date = stt_date(fake)
    builder.add("arriving in")
    builder.add(city, "CITY")
    maybe_add_filler(builder)
    builder.add("on")
    builder.add(date, "DATE")
    return builder.text(), builder.entities


def template_location_only(fake: Faker):
    builder = UtteranceBuilder()
    location = stt_location(fake)
    builder.add("venue is")
    builder.add(location, "LOCATION")
    return builder.text(), builder.entities


# ========== ALL TEMPLATES ==========

# TRAIN TEMPLATES (30 templates - 68%)
TRAIN_TEMPLATES: List[TemplateFn] = [
    # Original templates
    template_card_email_name,
    template_phone_city_date,
    template_email_only,
    template_location_trip,
    template_card_phone,
    template_name_email_phone,
    template_city_location,
    # Presidio-inspired (first 23 for train)
    presidio_01,
    presidio_02,
    presidio_03,
    presidio_04,
    presidio_05,
    presidio_06,
    presidio_07,
    presidio_08,
    presidio_09,
    presidio_10,
    presidio_11,
    presidio_12,
    presidio_13,
    presidio_14,
    presidio_15,
    presidio_16,
    presidio_17,
]

# DEV/TEST TEMPLATES (14 templates - 32%)
DEV_TEST_TEMPLATES: List[TemplateFn] = [
    # Original new templates
    template_phone_only,
    template_date_location,
    template_card_only,
    template_email_phone,
    template_name_only,
    template_city_date,
    template_location_only,
    # Presidio-inspired (remaining 7 for dev/test)
    presidio_18,
    presidio_19,
    presidio_20,
    presidio_21,
    presidio_22,
    presidio_23,
    presidio_24,
    presidio_25,
    presidio_26,
    presidio_27,
    presidio_28,
    presidio_29,
    presidio_30,
]

# All templates combined (for backwards compatibility)
TEMPLATES: List[TemplateFn] = TRAIN_TEMPLATES + DEV_TEST_TEMPLATES


@dataclass
class DatasetConfig:
    train_size: int
    dev_size: int
    test_size: int
    output_dir: Path
    train_seed: int = 13
    dev_seed: int = 42
    test_seed: int = 84


def generate_samples(fake: Faker, size: int, templates: List[TemplateFn] = None) -> List[Tuple[str, List[dict]]]:
    """Generate samples using specified templates."""
    if templates is None:
        templates = TEMPLATES
    samples = []
    for _ in range(size):
        template = random.choice(templates)
        samples.append(template(fake))
    return samples


def generate_split(size: int, seed: int, prefix: str, templates: List[TemplateFn] = None, include_entities: bool = True):
    """Generate a split using specific templates to prevent memorization."""
    random.seed(seed)
    fake = Faker()
    fake.seed_instance(seed)

    records = []
    for idx, (text, entities) in enumerate(generate_samples(fake, size, templates)):
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


def build_dataset(cfg: DatasetConfig, template_strategy: str = "no_overlap"):
    """Generate datasets with configurable template strategies.
    
    Args:
        cfg: Dataset configuration
        template_strategy: One of:
            - "no_overlap": Train and dev/test use different templates (tests generalization)
            - "allow_overlap": All splits use all templates (tests recognition)
    """
    
    if template_strategy == "no_overlap":
        train_templates = TRAIN_TEMPLATES
        dev_test_templates = DEV_TEST_TEMPLATES
        print(f"Template Strategy: NO OVERLAP")
        print(f"  Train uses {len(train_templates)} templates")
        print(f"  Dev/Test use {len(dev_test_templates)} DIFFERENT templates")
        print(f"  => Tests model's ability to GENERALIZE to new sentence structures")
    elif template_strategy == "allow_overlap":
        train_templates = TEMPLATES  # All templates
        dev_test_templates = TEMPLATES  # Same templates
        print(f"Template Strategy: ALLOW OVERLAP")
        print(f"  All splits use all {len(TEMPLATES)} templates")
        print(f"  => Tests model's PII RECOGNITION with familiar structures")
        print(f"  => Different seeds ensure different entity values")
    else:
        raise ValueError(f"Unknown template_strategy: {template_strategy}")
    
    print(f"  Total unique templates: {len(TEMPLATES)}")
    
    train_records = generate_split(cfg.train_size, cfg.train_seed, "utt_train", train_templates, include_entities=True)
    dev_records = generate_split(cfg.dev_size, cfg.dev_seed, "utt_dev", dev_test_templates, include_entities=True)
    test_records = generate_split(cfg.test_size, cfg.test_seed, "utt_test", dev_test_templates, include_entities=False)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(cfg.output_dir / "train.jsonl", train_records)
    write_jsonl(cfg.output_dir / "dev.jsonl", dev_records)
    write_jsonl(cfg.output_dir / "test.jsonl", test_records)

    print(f"\n[OK] Wrote {len(train_records)} train samples")
    print(f"[OK] Wrote {len(dev_records)} dev samples")
    print(f"[OK] Wrote {len(test_records)} test samples")
    print(f"[OK] All datasets saved to {cfg.output_dir}")


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if not config_path.exists():
        print(f"Warning: Config file {config_path} not found, using defaults")
        return {}
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def apply_noise_preset(config: Dict[str, Any]) -> STTNoiseConfig:
    """Apply noise preset and create STTNoiseConfig."""
    active_preset = config.get("active_preset", "realistic")
    
    # Load preset values
    presets = config.get("presets", {})
    preset_values = presets.get(active_preset, {})
    
    # If custom, use stt_noise section directly
    if active_preset == "custom":
        noise_config = config.get("stt_noise", {})
    else:
        # Use preset values
        noise_config = preset_values
        print(f"Using STT noise preset: '{active_preset}' - {preset_values.get('description', '')}")
    
    return STTNoiseConfig(
        digit_to_word_ratio=noise_config.get("digit_to_word_ratio", 0.5),
        zero_to_oh_ratio=noise_config.get("zero_to_oh_ratio", 0.1),
        filler_ratio=noise_config.get("filler_ratio", 0.3),
        lowercase_ratio=noise_config.get("lowercase_ratio", 1.0),
        email_semantic_link=noise_config.get("email_semantic_link", 0.6),
        spoken_card_ratio=noise_config.get("spoken_card_ratio", 0.5),
        spoken_phone_ratio=noise_config.get("spoken_phone_ratio", 0.5),
        spoken_date_ratio=noise_config.get("spoken_date_ratio", 0.5),
        location_with_city_ratio=noise_config.get("location_with_city_ratio", 0.4),
        extra_space_ratio=noise_config.get("extra_space_ratio", 0.15),
        missing_space_ratio=noise_config.get("missing_space_ratio", 0.1),
        at_spacing_variation=noise_config.get("at_spacing_variation", 0.3),
    )


def parse_args():
    ap = argparse.ArgumentParser(
        description="Generate synthetic STT-style PII datasets.",
        epilog="Note: CLI arguments override YAML config values."
    )
    ap.add_argument("--config", type=Path, default=Path("config/data_generation.yaml"),
                    help="Path to YAML config file")
    ap.add_argument("--train_size", type=int, default=None,
                    help="Number of training samples (overrides config)")
    ap.add_argument("--dev_size", type=int, default=None,
                    help="Number of dev samples (overrides config)")
    ap.add_argument("--test_size", type=int, default=None,
                    help="Number of test samples (overrides config)")
    ap.add_argument("--out_dir", type=Path, default=None,
                    help="Output directory (overrides config)")
    ap.add_argument("--train_seed", type=int, default=None,
                    help="Training data seed (overrides config)")
    ap.add_argument("--dev_seed", type=int, default=None,
                    help="Dev data seed (overrides config)")
    ap.add_argument("--test_seed", type=int, default=None,
                    help="Test data seed (overrides config)")
    ap.add_argument("--preset", type=str, default=None,
                    choices=["clean", "realistic", "noisy", "custom"],
                    help="STT noise preset (overrides config)")
    return ap.parse_args()


def main():
    global NOISE_CONFIG
    
    args = parse_args()
    
    # Load YAML config
    yaml_config = load_yaml_config(args.config)
    
    # Extract sections from YAML
    dataset_config = yaml_config.get("dataset", {})
    seeds_config = yaml_config.get("seeds", {})
    template_strategy = dataset_config.get("template_strategy", "no_overlap")
    
    # Apply CLI overrides (CLI args take precedence)
    train_size = args.train_size if args.train_size is not None else dataset_config.get("train_size", 900)
    dev_size = args.dev_size if args.dev_size is not None else dataset_config.get("dev_size", 150)
    test_size = args.test_size if args.test_size is not None else dataset_config.get("test_size", 150)
    out_dir = args.out_dir if args.out_dir is not None else Path(dataset_config.get("output_dir", "data"))
    train_seed = args.train_seed if args.train_seed is not None else seeds_config.get("train_seed", 13)
    dev_seed = args.dev_seed if args.dev_seed is not None else seeds_config.get("dev_seed", 42)
    test_seed = args.test_seed if args.test_seed is not None else seeds_config.get("test_seed", 77)
    
    # Apply noise preset
    if args.preset:
        yaml_config["active_preset"] = args.preset
    NOISE_CONFIG = apply_noise_preset(yaml_config)
    
    print("=" * 60)
    print("PII NER Dataset Generation")
    print("=" * 60)
    print(f"Train size: {train_size} (limit: 500-1000 per assignment)")
    print(f"Dev size: {dev_size} (limit: 100-200 per assignment)")
    print(f"Test size: {test_size}")
    print(f"Output: {out_dir}")
    print(f"Seeds: train={train_seed}, dev={dev_seed}, test={test_seed}")
    print(f"Template Strategy: {template_strategy}")
    print("\nSTT Noise Configuration:")
    for key, value in NOISE_CONFIG.to_dict().items():
        print(f"  {key}: {value}")
    print("=" * 60)
    print()
    
    # Create dataset config
    cfg = DatasetConfig(
        train_size=train_size,
        dev_size=dev_size,
        test_size=test_size,
        output_dir=out_dir,
        train_seed=train_seed,
        dev_seed=dev_seed,
        test_seed=test_seed,
    )
    
    # Generate dataset
    build_dataset(cfg, template_strategy=template_strategy)
    
    # Save noise config for reproducibility
    config_out = out_dir / "generation_config.json"
    with open(config_out, "w") as f:
        json.dump({
            "dataset": {
                "train_size": train_size,
                "dev_size": dev_size,
                "test_size": test_size,
                "template_strategy": template_strategy,
            },
            "seeds": {
                "train_seed": train_seed,
                "dev_seed": dev_seed,
                "test_seed": test_seed,
            },
            "stt_noise": NOISE_CONFIG.to_dict(),
            "templates": {
                "train_templates": len(TRAIN_TEMPLATES),
                "dev_test_templates": len(DEV_TEST_TEMPLATES),
                "total_templates": len(TEMPLATES),
                "strategy": template_strategy,
            }
        }, f, indent=2)
    print(f"\n[OK] Generation config saved to {config_out}")


if __name__ == "__main__":
    main()

