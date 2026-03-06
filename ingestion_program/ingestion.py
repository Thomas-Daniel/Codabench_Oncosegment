import csv
import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

EVAL_SETS = ["test", "private_test"]


def load_split(split_dir, with_masks=False):
    metadata_path = split_dir / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    sample_ids = []
    images = []
    masks = []
    labels = []

    with metadata_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_ids.append(row["sample_id"])
            image = Image.open(split_dir / row["image_path"]).convert("L")
            images.append(image)

            if with_masks:
                mask = Image.open(split_dir / row["mask_path"]).convert("L")
                masks.append(mask)
                labels.append(row.get("class_label", ""))

    if with_masks:
        return sample_ids, images, masks, labels
    return sample_ids, images


def to_binary_mask(mask):
    mask_array = np.asarray(mask)
    if mask_array.ndim == 3:
        mask_array = mask_array[..., 0]
    return (mask_array > 0).astype(np.uint8)


def normalize_predictions(predictions, sample_ids):
    if isinstance(predictions, dict):
        missing = [sid for sid in sample_ids if sid not in predictions]
        if missing:
            raise ValueError(
                f"Missing predictions for sample ids: {missing[:5]}"
            )
        return {sid: to_binary_mask(predictions[sid]) for sid in sample_ids}

    if len(predictions) != len(sample_ids):
        raise ValueError(
            f"Expected {len(sample_ids)} predictions, got {len(predictions)}."
        )
    return {
        sid: to_binary_mask(pred)
        for sid, pred in zip(sample_ids, predictions)
    }


def fit_model(model, train_images, train_masks, train_labels):
    if not hasattr(model, "fit"):
        return

    fit_sig = inspect.signature(model.fit)
    n_params = len(fit_sig.parameters)
    if n_params >= 3:
        model.fit(train_images, train_masks, train_labels)
    else:
        model.fit(train_images, train_masks)


def predict_masks(model, test_images, sample_ids):
    if not hasattr(model, "predict"):
        raise AttributeError("Submission model must define a `predict` method.")

    predict_sig = inspect.signature(model.predict)
    n_params = len(predict_sig.parameters)
    if n_params >= 2:
        predictions = model.predict(test_images, sample_ids)
    else:
        predictions = model.predict(test_images)
    return normalize_predictions(predictions, sample_ids)


def main(data_dir, output_dir):
    from submission import get_model

    train_ids, train_images, train_masks, train_labels = load_split(
        data_dir / "train",
        with_masks=True,
    )

    print(f"Loaded train split with {len(train_ids)} samples.")
    model = get_model()

    start = time.time()
    fit_model(model, train_images, train_masks, train_labels)
    train_time = time.time() - start

    print("Generating segmentation predictions...")
    start = time.time()
    predictions = {}
    for eval_set in EVAL_SETS:
        sample_ids, images = load_split(data_dir / eval_set, with_masks=False)
        prediction_map = predict_masks(model, images, sample_ids)
        predictions[eval_set] = prediction_map
        print(f"- {eval_set}: {len(sample_ids)} masks predicted")
    test_time = time.time() - start

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(
        json.dumps({"train_time": train_time, "test_time": test_time})
    )
    for eval_set in EVAL_SETS:
        np.savez_compressed(
            output_dir / f"{eval_set}_predictions.npz",
            **predictions[eval_set],
        )

    duration = train_time + test_time
    print(f"Completed prediction. Total duration: {duration:.3f}s")
    print("Ingestion program finished. Moving on to scoring.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingestion program for codabench"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/app/input_data",
        help="",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/app/output",
        help="",
    )
    parser.add_argument(
        "--submission-dir",
        type=str,
        default="/app/ingested_program",
        help="",
    )

    args = parser.parse_args()
    sys.path.append(args.submission_dir)
    sys.path.append(Path(__file__).parent.resolve())

    main(Path(args.data_dir), Path(args.output_dir))
