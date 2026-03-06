# OncoSegment Codabench Template

This repository is configured for the challenge:
`OncoSegment - Breast Tumor Mapping`.

Task scope in this template:
- Binary segmentation on breast ultrasound images.
- Only `benign` and `malignant` cases are kept from the source dataset.
- Metrics: Dice (primary) and Jaccard/IoU (secondary), both on public and private test sets.

## Dataset Source

Data is downloaded directly from Hugging Face in `tools/setup_data.py` using:

```python
load_dataset("gymprathap/Breast-Cancer-Ultrasound-Images-Dataset", split="train")
```

Link: https://huggingface.co/datasets/gymprathap/Breast-Cancer-Ultrasound-Images-Dataset

## Repository Layout

- `competition.yaml`: challenge metadata, tasks, phase, and leaderboard columns.
- `tools/setup_data.py`: downloads dataset, filters benign/malignant, creates train/test/private_test splits, writes Codabench input/reference folders.
- `ingestion_program/ingestion.py`: loads participant `submission.py`, trains model (optional), generates predicted masks as `.npz` files.
- `scoring_program/scoring.py`: computes Dice and Jaccard from predicted masks against hidden references.
- `solution/submission.py`: baseline Otsu-threshold segmentation starter.
- `dev_phase/`: generated split data used for local tests and bundle packaging.

## Submission API

Participants submit a `submission.py` with:

```python
def get_model():
    return model
```

Expected model methods:
- `fit(train_images, train_masks, train_labels=None)` (optional)
- `predict(test_images, sample_ids=None)` (required)

`predict` can return:
- List/tuple of masks in the same order as `test_images`, or
- Dict `{sample_id: mask}`

Each mask must be a binary or grayscale 2D array/image where non-zero means tumor.

## Prepare Data

```bash
python tools/setup_data.py --seed 42
```

Optional flags:
- `--dataset-id` (defaults to the Hugging Face repo above)
- `--split` (defaults to `train`)
- `--max-samples` (0 means all filtered samples)

## Local Test

Run ingestion:

```bash
python ingestion_program/ingestion.py --data-dir dev_phase/input_data/ --output-dir ingestion_res/ --submission-dir solution/
```

Run scoring:

```bash
python scoring_program/scoring.py --reference-dir dev_phase/reference_data/ --prediction-dir ingestion_res/ --output-dir scoring_res/
```

Expected score keys:
- `test_dice`
- `test_jaccard`
- `private_test_dice`
- `private_test_jaccard`
- `train_time`
- `test_time`

## Build Bundle

After generating `dev_phase` data:

```bash
python tools/create_bundle.py
```

Upload `bundle.zip` to Codabench:
https://www.codabench.org/competitions/upload/
