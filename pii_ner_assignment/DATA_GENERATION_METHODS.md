# PII NER Dataset Generation: Comprehensive Documentation

## Table of Contents

1. [Overview](#overview)
2. [Dataset Size Configuration](#dataset-size-configuration)
3. [STT Noise Parameters](#stt-noise-parameters)
4. [Dataset Generation Methods](#dataset-generation-methods)
5. [Template Architecture](#template-architecture)
6. [Noise Presets](#noise-presets)
7. [Reproducibility](#reproducibility)

---

## Overview

This project implements a synthetic data generation pipeline for PII Named Entity Recognition in Speech-to-Text transcripts. The generator simulates realistic STT errors and produces training, development, and test datasets with accurate character-level entity annotations.

### Key Features

- Three configurable dataset generation strategies
- 12 STT noise parameters to simulate real transcription errors
- 44 diverse conversation templates inspired by Microsoft Presidio
- Reproducible generation with seed control
- Support for 7 entity types with PII classification

### Entity Types

The system recognizes seven entity types:

**PII Entities:**
- CREDIT_CARD
- PHONE
- EMAIL
- PERSON_NAME
- DATE

**Non-PII Entities:**
- CITY
- LOCATION

---

## Dataset Size Configuration

### Assignment Constraints

The dataset size parameters follow the assignment requirements:

```yaml
dataset:
  train_size: 900        # Range: 500-1000 samples
  dev_size: 150          # Range: 100-200 samples
  test_size: 150         # Flexible
  output_dir: "data"
```

### Parameter Details

**train_size**
- Purpose: Number of training samples to generate
- Default: 900
- Range: 500-1000 per assignment constraints
- Impact: More training samples improve model generalization but increase training time

**dev_size**
- Purpose: Number of development samples for validation
- Default: 150
- Range: 100-200 per assignment constraints
- Impact: Used for hyperparameter tuning and early stopping

**test_size**
- Purpose: Number of test samples for final evaluation
- Default: 150
- Impact: Final model performance assessment without entity annotations

**output_dir**
- Purpose: Directory where generated JSONL files will be saved
- Default: "data"
- Outputs: train.jsonl, dev.jsonl, test.jsonl, generation_config.json

### Random Seeds

Different seeds ensure diversity across splits:

```yaml
seeds:
  train_seed: 13
  dev_seed: 42
  test_seed: 77
```

Using different seeds prevents identical entity values across splits while maintaining template structure based on generation method.

---

## STT Noise Parameters

The generator includes 12 configurable parameters that simulate real Speech-to-Text transcription errors and natural speech patterns.

### Digit Handling Parameters

#### digit_to_word_ratio
- Type: Float (0.0 - 1.0)
- Default: 0.5
- Purpose: Controls probability of keeping digits numeric versus converting to words
- Behavior:
  - 0.0: All digits converted to words ("4321" becomes "four three two one")
  - 1.0: All digits remain numeric ("4321" stays "4321")
  - 0.5: 50% chance of either format
- Example:
  - Input: "4242 4242 4242 4242"
  - Output (0.5): "four two four two 4242 4242 four two four two"
- Use Case: STT systems inconsistently transcribe spoken numbers

#### zero_to_oh_ratio
- Type: Float (0.0 - 1.0)
- Default: 0.1
- Purpose: When digits are spoken, controls saying "oh" instead of "zero"
- Behavior: Only applies when digit_to_word_ratio converts digits to words
- Example:
  - Input: "4200"
  - Output (0.1): "four two oh oh" (10% chance per zero)
  - Output (0.0): "four two zero zero"
- Use Case: Natural speech often uses "oh" for zero in numbers

### Conversational Noise Parameters

#### filler_ratio
- Type: Float (0.0 - 1.0)
- Default: 0.3
- Purpose: Probability of inserting conversational filler words
- Filler Words: "uh", "hmm", "like", "you know", "basically", "so"
- Example:
  - Input: "my card is 4321"
  - Output (0.3): "my card uh is 4321" or "like my card is 4321"
- Use Case: Real speech contains disfluencies that STT captures

### Capitalization Parameters

#### lowercase_ratio
- Type: Float (0.0 - 1.0)
- Default: 1.0
- Purpose: Probability of lowercasing entity names and text
- Example:
  - Input: "John Smith"
  - Output (1.0): "john smith"
  - Output (0.5): 50% "john smith", 50% "John Smith"
- Use Case: STT systems often fail to preserve proper capitalization

### Semantic Consistency Parameters

#### email_semantic_link
- Type: Float (0.0 - 1.0)
- Default: 0.6
- Purpose: Probability that email address semantically matches person name
- Example:
  - Name: "john smith"
  - Output (0.6): "john.smith@gmail.com" (60% chance)
  - Output (0.4): "randomuser123@mail.com" (40% chance)
- Use Case: Real conversations often have semantic consistency between entities

### Entity-Specific Noise Parameters

#### spoken_card_ratio
- Type: Float (0.0 - 1.0)
- Default: 0.5
- Purpose: Probability credit card numbers are fully spoken versus space-separated
- Example:
  - Input: "4321 8765 2109 4321"
  - Output (1.0): "four three two one eight seven six five..."
  - Output (0.0): "4 3 2 1 8 7 6 5..." (space-separated digits)
- Use Case: Cards can be read naturally ("forty-three twenty-one") or digit-by-digit

#### spoken_phone_ratio
- Type: Float (0.0 - 1.0)
- Default: 0.5
- Purpose: Probability phone numbers are spoken versus numeric
- Example:
  - Input: "9876543210"
  - Output (1.0): "nine eight seven six five four three two one zero"
  - Output (0.0): "9 8 7 6 5 4 3 2 1 0"
- Use Case: Phone numbers typically spoken digit-by-digit

#### spoken_date_ratio
- Type: Float (0.0 - 1.0)
- Default: 0.5
- Purpose: Probability dates are spoken versus numeric format
- Example:
  - Input: Date(2024, 5, 15)
  - Output (1.0): "fifteen may two zero two four" or "15 may 2024"
  - Output (0.0): "15 05 2024"
- Use Case: Dates can be spoken naturally or read as numbers

#### location_with_city_ratio
- Type: Float (0.0 - 1.0)
- Default: 0.4
- Purpose: Probability location names include city context
- Example:
  - Output (0.4): "central mall atlanta" (40% chance)
  - Output (0.6): "central mall" (60% chance)
- Use Case: Locations are often mentioned with city for clarity

### STT Spacing Error Parameters

These parameters simulate transcription errors in word boundary detection.

#### extra_space_ratio
- Type: Float (0.0 - 1.0)
- Default: 0.15
- Purpose: Probability of inserting extra spaces between words
- Example:
  - Input: "john smith"
  - Output (0.15): "john  smith" (double space)
- Use Case: STT systems sometimes output inconsistent spacing

#### missing_space_ratio
- Type: Float (0.0 - 1.0)
- Default: 0.1
- Purpose: Probability of missing spaces between words
- Example:
  - Input: "john smith"
  - Output (0.1): "johnsmith" (concatenated)
- Use Case: STT fails to detect word boundaries

#### at_spacing_variation
- Type: Float (0.0 - 1.0)
- Default: 0.3
- Purpose: Variable spacing around "at" and "dot" in email addresses
- Variations:
  - " at " (standard)
  - "  at  " (extra spaces)
  - " at" (space before only)
  - "at " (space after only)
- Example:
  - Input: "john@gmail.com"
  - Output: "john  at  gmail dot com" (inconsistent spacing)
- Use Case: STT produces inconsistent spacing for email separators

---

## Dataset Generation Methods

The project implements three distinct generation strategies to test different model capabilities.

### Method 1: Baseline

**Configuration:** `data_baseline/`

**Purpose:** Simple baseline with basic STT noise

#### Characteristics

- 7 simple conversation templates
- Random split between train and development sets
- 9 basic STT noise parameters (no spacing errors)
- Possible template overlap between splits

#### Templates Used

1. template_card_email_name: Credit card + email + name
2. template_phone_city_date: Phone + city + date
3. template_email_only: Email + name
4. template_location_trip: Location + date
5. template_card_phone: Credit card + phone
6. template_name_email_phone: Name + email + phone
7. template_city_location: City + location

#### STT Noise Configuration

```json
{
  "digit_to_word_ratio": 0.5,
  "zero_to_oh_ratio": 0.1,
  "filler_ratio": 0.3,
  "lowercase_ratio": 1.0,
  "email_semantic_link": 0.6,
  "spoken_card_ratio": 0.5,
  "spoken_phone_ratio": 0.5,
  "spoken_date_ratio": 0.5,
  "location_with_city_ratio": 0.4
}
```

Note: Missing spacing parameters (extra_space, missing_space, at_spacing_variation)

#### Example Output

```
"my credit card number is four two four two 4242 4242 4242 and email is john dot smith at gmail dot com"
```

#### When to Use

- Assignment submission (meets basic requirements)
- Quick baseline for comparison
- Testing basic PII recognition without complex generalization

---

### Method : No Template Overlap

**Configuration:** `data_method1_no_overlap/`

**Purpose:** Test Natural Language Understanding and STT generalization

#### Characteristics

- 44 diverse templates total
- 24 templates exclusively for training (55%)
- 20 completely different templates for dev/test (45%)
- Zero overlap between train and evaluation templates
- Full 12-parameter STT noise simulation

#### Template Strategy

```
Train Templates (24):
  - Original 7 templates
  - Presidio-inspired templates 01-17

Dev/Test Templates (20):
  - 7 new original templates
  - Presidio-inspired templates 18-30

Overlap: NONE
```

#### STT Noise Configuration

```json
{
  "digit_to_word_ratio": 0.5,
  "zero_to_oh_ratio": 0.1,
  "filler_ratio": 0.3,
  "lowercase_ratio": 1.0,
  "email_semantic_link": 0.6,
  "spoken_card_ratio": 0.5,
  "spoken_phone_ratio": 0.5,
  "spoken_date_ratio": 0.5,
  "location_with_city_ratio": 0.4,
  "extra_space_ratio": 0.15,
  "missing_space_ratio": 0.1,
  "at_spacing_variation": 0.3
}
```

#### Example Training Sample

```
"need to change billing date of my card four two three two 3 two 2 1 eight three 4 zero 6 2 9 0 four 2 nine"
```

#### Example Dev Sample (Different Template)

```
"contact donaldgarcia  at  example dot net phone 9 6 oh 0 4 5 3 7 8 9"
```

Note: The dev sample uses a completely different sentence structure not seen during training.

#### What It Tests

1. Natural Language Understanding: Can the model recognize PII patterns across different sentence structures?
2. STT Generalization: Can the model handle diverse noise patterns it hasn't seen?
3. True Generalization: Does the model learn entity patterns versus memorizing positions?

#### When to Use

- Production deployment preparation
- Research on model generalization
- Testing robustness to novel phrasing
- Realistic performance assessment

---

### Method : Template Overlap Allowed

**Configuration:** `data_method2_overlap/`

**Purpose:** Test PII recognition accuracy without confounding generalization

#### Characteristics

- Same 44 templates used in all splits
- Train, dev, and test use identical sentence structures
- Only entity values differ across splits (controlled by seeds)
- Full 12-parameter STT noise simulation

#### Template Strategy

```
Train Templates: ALL 44 templates
Dev Templates: SAME 44 templates
Test Templates: SAME 44 templates

Overlap: COMPLETE
```

Different seeds ensure different entity values:
- Train: "my name is John Smith"
- Dev: "my name is Alice Johnson" (same structure, different name)

#### STT Noise Configuration

Identical to Method 2 (all 12 parameters)

#### Example Training Sample

```
"whats your email denisewade at example dot org"
```

#### Example Dev Sample (Same Template)

```
"whats your email lindsay78  at  example dot org"
```

Note: Same template structure, different entity value, different spacing pattern.

#### What It Tests

1. PII Recognition: Can the model identify PII entities in familiar contexts?
2. STT Noise Handling: Can the model handle spacing variations and noise?
3. Upper Bound Performance: What's the best achievable accuracy without generalization challenges?

#### When to Use

- Establishing performance upper bounds
- Debugging entity recognition issues
- Testing STT noise robustness in isolation
- Understanding overfitting potential

---

## Template Architecture

### Template Categories

The 44 templates are inspired by Microsoft Presidio and cover diverse conversation scenarios:

1. Credit card scenarios (lost cards, billing updates, verification)
2. Phone number inquiries (callbacks, contact updates, verification)
3. Email communications (updates, contact info, account changes)
4. Name verification (customer service, identification)
5. Location references (meetings, addresses, travel)
6. Multi-entity conversations (combined PII elements)
7. Duplicate entities (multiple phones, email updates)
8. Question-answer patterns (natural dialogue)

### Original Templates (14 total)

#### Used in All Methods

1. template_card_email_name
2. template_phone_city_date
3. template_email_only
4. template_location_trip
5. template_card_phone
6. template_name_email_phone
7. template_city_location

#### Additional for Method 2 & 3

8. template_phone_only
9. template_date_location
10. template_card_only
11. template_email_phone
12. template_name_only
13. template_city_date
14. template_location_only

### Presidio-Inspired Templates (30 total)

Adapted from Microsoft Presidio PII recognizer test cases:

#### Training Templates (presidio_01 to presidio_17)

Examples:
- presidio_01: "my credit card {card} has been lost can you block it"
- presidio_02: "need to change billing date of my card {card}"
- presidio_03: "i have lost my card {card} my name is {name}"
- presidio_07: "please have manager call me at {phone}"
- presidio_10: "whats your email {email}"
- presidio_12: "how can we reach you call {phone}"

#### Dev/Test Templates (presidio_18 to presidio_30)

Examples:
- presidio_18: "i would like to stop receiving messages to {phone}"
- presidio_22: "hi my card {card} was declined call {phone}"
- presidio_23: "change my email from {email} to {email}" (duplicate entities)
- presidio_24: "call {phone} or {phone}" (multiple same type)
- presidio_28: "restaurant is at {location} in {city}"

### Template Distribution Visualization

```
BASELINE:
├── Train: [T1-T7] randomly selected
└── Dev:   [T1-T7] randomly selected
    └── Overlap: Possible

METHOD 1 (NO OVERLAP):
├── Train: [T1-T7, P01-P17] = 24 templates
└── Dev:   [T8-T14, P18-P30] = 20 templates
    └── Overlap: ZERO

METHOD 2 (OVERLAP):
├── Train: [T1-T14, P01-P30] = 44 templates
└── Dev:   [T1-T14, P01-P30] = 44 templates
    └── Overlap: COMPLETE
```

---

## Noise Presets

The system provides three pre-configured noise presets for quick experimentation.

### Clean Preset

**Purpose:** Minimal noise for easier model training

```yaml
presets:
  clean:
    description: "Minimal noise - easier for model"
    digit_to_word_ratio: 0.2      # 20% spoken digits
    filler_ratio: 0.1              # 10% fillers
    lowercase_ratio: 0.5           # 50% lowercase
    spoken_card_ratio: 0.2
    spoken_phone_ratio: 0.2
    extra_space_ratio: 0.05        # 5% spacing errors
    missing_space_ratio: 0.02
    at_spacing_variation: 0.1
```

**Example Output:**
```
"my card 4232 3221 8340 6290 429"
```

**When to Use:**
- Initial model development
- Debugging entity detection
- Baseline performance testing

### Realistic Preset (Default)

**Purpose:** Moderate noise simulating real STT transcripts

```yaml
presets:
  realistic:
    description: "Moderate noise - realistic STT"
    digit_to_word_ratio: 0.5      # 50% spoken digits
    filler_ratio: 0.3              # 30% fillers
    lowercase_ratio: 1.0           # All lowercase
    spoken_card_ratio: 0.5
    spoken_phone_ratio: 0.5
    extra_space_ratio: 0.15        # 15% spacing errors
    missing_space_ratio: 0.1
    at_spacing_variation: 0.3
```

**Example Output:**
```
"my card four two three two 3 two 2 1 eight three 4 zero 6 2 9 0 four 2 nine"
```

**When to Use:**
- Production model training
- Assignment submission
- Performance benchmarking

### Noisy Preset

**Purpose:** High noise for challenging model training

```yaml
presets:
  noisy:
    description: "High noise - challenging"
    digit_to_word_ratio: 0.8      # 80% spoken digits
    filler_ratio: 0.5              # 50% fillers
    lowercase_ratio: 1.0           # All lowercase
    spoken_card_ratio: 0.8
    spoken_phone_ratio: 0.8
    extra_space_ratio: 0.25        # 25% spacing errors
    missing_space_ratio: 0.2
    at_spacing_variation: 0.5
```

**Example Output:**
```
"uh my card like four two three two three hmm two two one eight three four  zero six two nine zero four two nine"
```
---

## Reproducibility

### Generation Configuration Logging

Each dataset generation saves a complete configuration file for reproducibility:

**File:** `{output_dir}/generation_config.json`

**Contents:**

```json
{
  "dataset": {
    "train_size": 900,
    "dev_size": 150,
    "test_size": 150,
    "template_strategy": "no_overlap"
  },
  "seeds": {
    "train_seed": 13,
    "dev_seed": 42,
    "test_seed": 77
  },
  "stt_noise": {
    "digit_to_word_ratio": 0.5,
    "zero_to_oh_ratio": 0.1,
    "filler_ratio": 0.3,
    "lowercase_ratio": 1.0,
    "email_semantic_link": 0.6,
    "spoken_card_ratio": 0.5,
    "spoken_phone_ratio": 0.5,
    "spoken_date_ratio": 0.5,
    "location_with_city_ratio": 0.4,
    "extra_space_ratio": 0.15,
    "missing_space_ratio": 0.1,
    "at_spacing_variation": 0.3
  },
  "templates": {
    "train_templates": 24,
    "dev_test_templates": 20,
    "total_templates": 44,
    "strategy": "no_overlap"
  }
}
```

### Usage Commands

#### Generate with Default Configuration

```bash
python scripts/generate_data.py
```

#### Generate with Specific Preset

```bash
python scripts/generate_data.py --preset clean
python scripts/generate_data.py --preset realistic
python scripts/generate_data.py --preset noisy
```

#### Generate with Custom Parameters

```bash
python scripts/generate_data.py \
  --train_size 1000 \
  --dev_size 200 \
  --preset noisy \
  --out_dir data_custom
```

#### Use Specific Configuration File

```bash
python scripts/generate_data.py --config config/method1_no_template_overlap.yaml
python scripts/generate_data.py --config config/method2_template_overlap.yaml
```

### Seed Control

Different seeds ensure dataset diversity:

- **train_seed:** Controls training data entity values and template selection
- **dev_seed:** Controls development data entity values and template selection
- **test_seed:** Controls test data entity values and template selection

Using different seeds prevents data leakage while maintaining consistent noise patterns.

---

## Summary Comparison

### Method Selection Guide

| Aspect | Baseline | Method 1 (No Overlap) | Method 2 (Overlap) |
|--------|----------|----------------------|-------------------|
| Templates | 7 simple | 44 (24 train / 20 dev) | 44 (all shared) |
| STT Noise | 9 parameters | 12 parameters | 12 parameters |
| Template Overlap | Possible | None | Complete |
| Tests NLU | Limited | Strong | None |
| Tests STT Noise | Basic | Full | Full |
| Production Ready | Good | Best | Overfits |
| Training Difficulty | Easy | Hard | Medium |
| Assignment Use | Recommended | Research | Upper Bound |

### Parameter Complexity

```
BASELINE:
[████████████████░░░░] 9/12 noise parameters

METHOD 1 & 2:
[████████████████████] 12/12 noise parameters
```

### Dataset Statistics

All methods generate:
- Training: 900 samples
- Development: 150 samples
- Test: 150 samples

Total: 1,200 labeled utterances per method

---

## Future Work: Scaling with Large-Scale PII Datasets

### Approach with More Resources and Time

Given additional resources and time, the optimal approach would be to combine large-scale general PII detection datasets with our specialized STT noise simulation methodology.

### Proposed Method

#### 1. Use Large-Scale Base Dataset

Instead of generating from scratch with 44 templates, leverage existing large-scale PII datasets such as:

**AI4Privacy PII-Masking-200k** ([https://huggingface.co/datasets/ai4privacy/pii-masking-200k](https://huggingface.co/datasets/ai4privacy/pii-masking-200k))
- 209,000 examples with 649k PII tokens
- 54 PII classes (far more diverse than our 7 classes)
- 229 discussion subjects across business, education, psychology, and legal domains
- 4 languages: English, French, German, Italian
- Human-in-the-loop validated quality
- Multiple interaction styles (casual, formal, email, etc.)

**Microsoft Presidio Research** ([https://github.com/microsoft/presidio-research](https://github.com/microsoft/presidio-research))
- Academic-quality PII recognition research
- Benchmark datasets and evaluation frameworks
- Production-tested entity patterns

#### 2. Apply STT Noise Simulation Pipeline

Take the clean text from these datasets and apply our 12-parameter STT noise simulation:

```python
# Pseudocode
for example in ai4privacy_dataset:
    clean_text = example['source_text']
    pii_labels = example['privacy_mask']
    
    # Apply our STT noise transformation
    noisy_text = apply_stt_noise(
        text=clean_text,
        noise_config=STTNoiseConfig(
            digit_to_word_ratio=0.5,
            filler_ratio=0.3,
            extra_space_ratio=0.15,
            missing_space_ratio=0.1,
            at_spacing_variation=0.3,
            # ... all 12 parameters
        )
    )
    
    # Adjust entity spans for noise-induced position shifts
    adjusted_labels = adjust_entity_positions(pii_labels, clean_text, noisy_text)
```

#### 3. Benefits of This Approach

**Scale:**
- 200,000+ training examples vs our 900
- 100x more data for robust model training

**Diversity:**
- 54 PII classes vs our 7 entity types
- Covers IBAN, IP addresses, SSN, passport numbers, medical records
- More realistic conversation patterns from human validation

**Domain Coverage:**
- Business contracts, legal documents, medical records
- Educational transcripts, customer support, emails
- Multiple languages and cultural contexts

**STT Specialization:**
- Our unique 12-parameter noise simulation adds STT-specific challenges
- Preserves the quality labels from AI4Privacy
- Tests model robustness to speech recognition errors

**Pre-trained Models:**
- Leverage existing models like [SoelMgd/bert-pii-detection](https://huggingface.co/SoelMgd/bert-pii-detection)
- Fine-tune on STT-noised version for domain adaptation
- Transfer learning from 66.4M parameter BERT models

#### 4. Implementation Strategy

**Phase 1: Data Preparation**
- Download AI4Privacy dataset (209k examples)
- Implement character-position-aware STT noise pipeline
- Validate entity span alignment after noise injection

**Phase 2: Dataset Generation**
- Apply STT noise with multiple preset levels (clean, realistic, noisy)
- Create train/dev/test splits maintaining original proportions
- Generate multiple noise variants per example for augmentation

**Phase 3: Model Training**
- Start from bert-pii-detection checkpoint
- Fine-tune on STT-noised data
- Evaluate on both clean and noisy test sets

**Phase 4: Evaluation**
- Benchmark against clean-text models
- Measure robustness to varying noise levels
- Compare with rule-based STT-specific approaches

### Why This Wasn't Done in Assignment

**Time Constraints:**
- Assignment duration: 2 hours
- Dataset preparation alone would require several days
- Character-position tracking during noise injection is complex

**Resource Limitations:**
- AI4Privacy dataset is 381 MB download
- Training on 200k examples requires GPU resources
- Validation and quality control need extensive compute

**Scope:**
- Assignment focuses on demonstrating ML approach, not production scale
- 900 training samples sufficient to prove concept
- Custom generation allows full control over noise parameters

### Expected Performance Improvement

Based on the literature and dataset scale:

**Current Approach (900 samples):**
- PII Precision: 0.62-0.69 (Baseline/Method1)
- Best: 0.96-0.98 (Method2, but overfits)

**Expected with AI4Privacy + STT Noise (200k samples):**
- PII Precision: 0.85-0.92 (robust generalization)
- Better recall on rare PII types
- Improved handling of novel phrasings
- More consistent performance across domains

### Conclusion

The current synthetic generation approach with 44 templates successfully demonstrates the PII NER concept within assignment constraints. However, for production deployment, combining large-scale curated datasets like AI4Privacy with our STT noise simulation methodology would yield:

1. Superior model performance through scale
2. Better generalization across domains and PII types
3. Maintained STT-specific robustness through our noise pipeline
4. Reduced development time by leveraging existing validated data

This hybrid approach represents the optimal balance between leveraging community resources and addressing the specific challenges of PII detection in speech-to-text transcripts.
