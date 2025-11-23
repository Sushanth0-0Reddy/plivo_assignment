# Configuration Guide

## Dataset Generation Configuration

All dataset generation parameters are controlled via `data_generation.yaml`.

### Quick Start

**Default (Realistic STT noise):**
```bash
python scripts/generate_data.py
```

**Use different noise preset:**
```bash
python scripts/generate_data.py --preset clean    # Minimal noise
python scripts/generate_data.py --preset noisy    # High noise
```

**Override specific parameters:**
```bash
python scripts/generate_data.py --train_size 1000 --dev_size 200 --preset noisy
```

---

## Configuration Structure

### 1. Dataset Size

```yaml
dataset:
  train_size: 900        # 500-1000 per assignment
  dev_size: 150          # 100-200 per assignment
  test_size: 150
```

### 2. STT Noise Parameters

Control how "realistic" the STT transcripts are:

| Parameter | Range | Description | Example |
|-----------|-------|-------------|---------|
| `digit_to_word_ratio` | 0.0-1.0 | How often digits stay numeric vs spoken | `0.5` = "123" vs "one two three" |
| `filler_ratio` | 0.0-1.0 | Add conversational fillers | "uh", "hmm", "like" |
| `lowercase_ratio` | 0.0-1.0 | Lowercase entity names | "John" → "john" |
| `spoken_card_ratio` | 0.0-1.0 | Credit cards spoken | "4321" → "four three two one" |
| `spoken_phone_ratio` | 0.0-1.0 | Phone numbers spoken | Same as above |
| `email_semantic_link` | 0.0-1.0 | Email matches name | "john smith" → "john.smith@..." |

###3. Noise Presets

**Clean** (Minimal noise - easier for model):
- 20% digit-to-word
- 10% fillers
- 50% lowercase

**Realistic** (Default - moderate noise):
- 50% digit-to-word
- 30% fillers
- 100% lowercase

**Noisy** (Challenging):
- 80% digit-to-word
- 50% fillers
- 100% lowercase

### 4. Templates

- **44 total templates** (inspired by Microsoft Presidio)
- **24 templates for training** (55%)
- **20 different templates for dev/test** (45%)

This split prevents the model from memorizing template patterns.

---

## Sample Outputs

### Realistic Preset (Default)
```json
{
  "text": "my card four two three two 3 two 2 1 eight three 4 zero 6 2 9 0 four 2 nine",
  "entities": [{"start": 8, "end": 76, "label": "CREDIT_CARD"}]
}
```

### Clean Preset
```json
{
  "text": "my card 4232 3221 8340 6290 429",
  "entities": [{"start": 8, "end": 31, "label": "CREDIT_CARD"}]
}
```

### Noisy Preset
```json
{
  "text": "uh my card like four two three two three hmm two two one",
  "entities": [{"start": 20, "end": 57, "label": "CREDIT_CARD"}]
}
```

---

## Assignment Constraints

Per the assignment requirements:

✅ **Train**: 500-1000 examples (current: 900)  
✅ **Dev**: 100-200 examples (current: 150)  
✅ **Targets**: PII Precision ≥ 0.80, p95 Latency ≤ 20ms

---

## Advanced Usage

### Create Custom Preset

Edit `data_generation.yaml`:

```yaml
active_preset: "custom"

stt_noise:
  digit_to_word_ratio: 0.7
  filler_ratio: 0.4
  lowercase_ratio: 0.8
  # ... other params
```

### Change Seeds for Different Data

```bash
python scripts/generate_data.py --train_seed 99 --dev_seed 88 --test_seed 77
```

### Generate Smaller Dataset for Testing

```bash
python scripts/generate_data.py --train_size 100 --dev_size 20 --out_dir data_small
```

---

## Reproducibility

All generation parameters are saved to `{output_dir}/generation_config.json` for reproducibility.

