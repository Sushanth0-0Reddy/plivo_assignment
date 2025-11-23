# PII NER Dataset Generation - Three Methods Comparison

## Overview

We've generated **three datasets** using different strategies to test different aspects of the model:

- **Baseline** - Basic templates, simple STT noise
- **Presidio-Inspired (No Overlap)** - Tests **Natural Language Understanding + STT Generalization**
- **Presidio-Inspired (Overlap)** - Tests **STT Generalization Only**

---

## Quick Comparison Table

| Aspect | Baseline | Presidio-Inspired (No Overlap) | Presidio-Inspired (Overlap) |
|--------|----------|-------------------------------|----------------------------|
| **Templates** | 7 simple | 44 diverse (24 train / 20 dev) | 44 diverse (all shared) |
| **Template Strategy** | Random split | **Disjoint templates** | **Same templates** |
| **STT Noise** | Basic (9 params) | Enhanced (12 params) | Enhanced (12 params) |
| **Spacing Noise** | No | Yes | Yes |
| **`at` Variations** | No | Yes | Yes |
| **What It Tests** |Basic STT | **NLU + STT generalization** | **STT generalization only** |
---

## Baseline (Basic Templates)

**Configuration:** `data_baseline/`

### Templates Used (7 Simple Templates)

- **`template_card_email_name`**
  - Pattern: `"my credit card number is {card} and email is {email} name on the card is {name}"`
  - Entities: CREDIT_CARD, EMAIL, PERSON_NAME

- **`template_phone_city_date`**
  - Pattern: `"call me on {phone} i am calling from {city} and i will travel on {date}"`
  - Entities: PHONE, CITY, DATE

- **`template_email_only`**
  - Pattern: `"email id is {email} person name {name}"`
  - Entities: EMAIL, PERSON_NAME

- **`template_location_trip`**
  - Pattern: `"meeting location will be {location} on {date}"`
  - Entities: LOCATION, DATE

- **`template_card_phone`**
  - Pattern: `"card digits are {card} reach me at number {phone}"`
  - Entities: CREDIT_CARD, PHONE

- **`template_name_email_phone`**
  - Pattern: `"this is {name} send note to {email} or ping {phone}"`
  - Entities: PERSON_NAME, EMAIL, PHONE

- **`template_city_location`**
  - Pattern: `"currently staying in {city} near {location}"`
  - Entities: CITY, LOCATION

### STT Noise (9 Parameters)

```json
{
  "digit_to_word_ratio": 0.5,       // Mix of "4" and "four"
  "zero_to_oh_ratio": 0.1,          // "0" → "oh"
  "filler_ratio": 0.3,              // Add "um", "uh"
  "lowercase_ratio": 1.0,           // All lowercase
  "email_semantic_link": 0.6,        // Email matches name
  "spoken_card_ratio": 0.5,         // Spoken card numbers
  "spoken_phone_ratio": 0.5,       // Spoken phone numbers
  "spoken_date_ratio": 0.5,         // Spoken dates
  "location_with_city_ratio": 0.4   // Location + city pairs
  // Missing: extra_space, missing_space, at_spacing
}
```

### Sample Data

```json
{"text": "need to change billing date of my card four two three two 3 two 2 1 eight three 4 zero 6 2 9 0 four 2 nine"}
{"text": "please have manager call me at 4 7 6 2 4 5 3 7 8 9"}
{"text": "my credit card number is 4820 2764 0502 5079 and email is brucecassie at example dot com"}
```

---

## Presidio-Inspired (No Template Overlap)

**Configuration:** `data_method1_no_overlap/`

### Templates Used (44 Total: 24 Train + 20 Dev)

#### Train Templates (24):
**Original (7):**
- `template_card_email_name`, `template_phone_city_date`, `template_email_only`, 
- `template_location_trip`, `template_card_phone`, `template_name_email_phone`, 
- `template_city_location`

**Presidio-Inspired (17):**
- `presidio_01`: `"my credit card {card} has been lost can you block it"`
- `presidio_02`: `"need to change billing date of my card {card}"`
- `presidio_03`: `"i have lost my card {card} my name is {name}"`
- `presidio_04`: `"didnt get message on my registered {phone}"`
- `presidio_05`: `"send last billed amount for card {card} to {email}"`
- `presidio_06`: `"please have manager call me at {phone}"`
- `presidio_07`: `"whats your email {email}"`
- `presidio_08`: `"contact {name} at {email} phone {phone}"`
- `presidio_09`: `"customer name {name} date of birth {date}"`
- `presidio_10`: `"restaurant is at {location} east {city}"`
- `presidio_11`: `"how can we reach you call {phone}"`
- `presidio_12`: `"update my email from {email} to {email}"`
- `presidio_13`: `"my name is {name} {name}"` (duplicate entities)
- `presidio_14`: `"call {phone} or {phone}"` (multiple same type)
- `presidio_15`: `"hi my card {card} was declined call {phone}"`
- `presidio_16`: `"change my email from {email} to {email}"`
- `presidio_17`: `"i moved to {city} please update {email}"`

#### Dev/Test Templates (20 - Completely Different):
**Original (7):**
- `template_phone_only`, `template_date_location`, `template_card_only`, 
- `template_email_phone`, `template_name_only`, `template_city_date`, 
- `template_location_only`

**Presidio-Inspired (13):**
- `presidio_18` through `presidio_30` (different sentence structures)

### STT Noise (12 Parameters - Enhanced)

```json
{
  // All 9 baseline parameters PLUS:
  "extra_space_ratio": 0.15,        // "at example" → "at  example"
  "missing_space_ratio": 0.1,      // "at example" → "atexample"
  "at_spacing_variation": 0.3      // "at" → " at " / "at " / " at"
}
```

### Template Strategy

```python
# Train uses first 24 templates
TRAIN_TEMPLATES = [template_card_email_name, ..., presidio_17]

# Dev/Test uses DIFFERENT 20 templates
DEV_TEST_TEMPLATES = [template_phone_only, ..., presidio_30]

# Zero overlap! Forces true generalization
```

### Sample Data

**Train:**
```json
{"text": "need to change billing date of my card four two three two 3..."}
{"text": "please have manager call me at 4 7 6 2 4 5 3 7 8 9"}
```

**Dev (Different Templates!):**
```json
{"text": "contact donaldgarcia at example dot net phone 9 6 oh 0..."}
{"text": "customer name jerry ramirez date of birth january 15 1985"}
```

### What It Tests

- **Natural Language Understanding**: Can model learn entity patterns across different sentence structures?
- **STT Generalization**: Can model handle noisy STT patterns it hasn't seen in training?

**Purpose:** Tests if model truly learned PII patterns vs memorizing sentence structures

---

## Presidio-Inspired (Template Overlap)

**Configuration:** `data_method2_overlap/`

### Templates Used (44 Total - All Shared)

**Same 44 templates as Presidio-Inspired (No Overlap)**, but:
- **Train uses ALL 44 templates**
- **Dev/Test uses SAME 44 templates**
- Only entity **values** differ, not sentence structures

### STT Noise (12 Parameters - Same as Presidio-Inspired No Overlap)

```json
{
  // Same 12 parameters as Presidio-Inspired (No Overlap)
  "digit_to_word_ratio": 0.5,
  "extra_space_ratio": 0.15,
  "missing_space_ratio": 0.1,
  "at_spacing_variation": 0.3,
  // ... etc
}
```

### Template Strategy

```python
# Both train and dev use ALL templates
TRAIN_TEMPLATES = ALL_44_TEMPLATES
DEV_TEST_TEMPLATES = ALL_44_TEMPLATES  # Same!

# Templates overlap! Model can memorize positions
```

### Sample Data

**Train:**
```json
{"text": "whats your email denisewade at example dot org"}
{"text": "restaurant is at river view park east chad in lake kenneth"}
```

**Dev (Same Templates, Different Entities):**
```json
{"text": "whats your email lindsay78 at example dot org"}  // Same structure!
{"text": "restaurant is at central mall west austin in dallas"}  // Same structure!
```

### What It Tests

- **STT Generalization Only**: Can model handle noisy STT patterns (spacing, "at" variations)?
- **NOT testing NLU**: Model memorizes template positions, not general patterns

**Purpose:** Tests STT noise handling when sentence structures are familiar

---

## Visual Comparison

### Template Distribution

```
BASELINE:
Train: [T1, T2, T3, T4, T5, T6, T7] ← Random selection
Dev:   [T1, T2, T3, T4, T5, T6, T7] ← Random selection
       Templates overlap (but only 7 simple ones)

PRESIDIO-INSPIRED (NO OVERLAP):
Train: [T1, T2, ... T24] ← First 24 only
Dev:   [T25, T26, ... T44] ← Last 20 only
       Zero overlap! Forces true generalization

PRESIDIO-INSPIRED (OVERLAP):
Train: [T1, T2, ... T44] ← All 44
Dev:   [T1, T2, ... T44] ← Same 44
       Full overlap! Model can memorize positions
```

### STT Noise Evolution

```
BASELINE:                    [████████████████░░░░]  9/12 noise types (basic)
PRESIDIO-INSPIRED (BOTH):    [████████████████████]  12/12 noise types (full)
                             Added: spacing errors + at-variations
```

---

## Which Method Tests What?

| Capability | Baseline | Presidio-Inspired (No Overlap) | Presidio-Inspired (Overlap) |
|------------|----------|-------------------------------|----------------------------|
| **Basic PII Recognition** | Yes | Yes | Yes |
| **STT Noise Handling** | Basic | Full | Full |
| **NLU Generalization** | Limited | Strong | None |
| **Template Memorization** | Possible | Prevented | Occurs |
| **Production Readiness** | Good | Best | Overfits |

---

## Key Takeaways

- **Baseline**: Simple, realistic, good for assignment submission
- **Presidio-Inspired (No Overlap)**: Tests **both NLU and STT generalization** - best for research
- **Presidio-Inspired (Overlap)**: Tests **only STT generalization** - shows upper bound but overfits
