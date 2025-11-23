ASSIGNMENT_ENTITY_LABELS = [
    "CREDIT_CARD",
    "PHONE",
    "EMAIL",
    "PERSON_NAME",
    "DATE",
    "CITY",
    "LOCATION",
]

PII_LABELS = {
    "CREDIT_CARD",
    "PHONE",
    "EMAIL",
    "PERSON_NAME",
    "DATE",
}

# Canonical HF label (from SoelMgd/bert-pii-detection) to use when fine-tuning for each entity.
ASSIGNMENT_TO_PRIMARY_HF = {
    "O": "O",
    "CREDIT_CARD": "CREDITCARDNUMBER",
    "PHONE": "PHONENUMBER",
    "EMAIL": "EMAIL",
    "PERSON_NAME": "FIRSTNAME",
    "DATE": "DATE",
    "CITY": "CITY",
    "LOCATION": "STATE",
}

# Map the HF model's 50+ entity labels back to our assignment schema at inference time.
HF_TO_ASSIGNMENT = {
    "CREDITCARDNUMBER": "CREDIT_CARD",
    "CREDITCARDCVV": "CREDIT_CARD",
    "CREDITCARDISSUER": "CREDIT_CARD",
    "ACCOUNTNUMBER": "CREDIT_CARD",
    "ACCOUNTNAME": "CREDIT_CARD",
    "PHONENUMBER": "PHONE",
    "PHONEIMEI": "PHONE",
    "EMAIL": "EMAIL",
    "USERNAME": "EMAIL",
    "FIRSTNAME": "PERSON_NAME",
    "LASTNAME": "PERSON_NAME",
    "MIDDLENAME": "PERSON_NAME",
    "PREFIX": "PERSON_NAME",
    "DATE": "DATE",
    "DOB": "DATE",
    "TIME": "DATE",
    "CITY": "CITY",
    "STATE": "LOCATION",
    "COUNTY": "LOCATION",
    "STREET": "LOCATION",
    "SECONDARYADDRESS": "LOCATION",
    "ZIPCODE": "LOCATION",
    "BUILDINGNUMBER": "LOCATION",
    "NEARBYGPSCOORDINATE": "LOCATION",
}


def label_is_pii(label: str) -> bool:
    """Return True if the assignment label should be treated as PII."""
    return label in PII_LABELS


def hf_label_to_assignment(label: str):
    """Map a fine-grained HF label back to the assignment entity (or None if unsupported)."""
    return HF_TO_ASSIGNMENT.get(label)


def assignment_label_to_hf(label: str) -> str:
    """Return the canonical HF label name used when fine-tuning for a given assignment entity."""
    return ASSIGNMENT_TO_PRIMARY_HF.get(label, "O")
