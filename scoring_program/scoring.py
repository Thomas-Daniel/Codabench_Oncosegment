import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

EVAL_SETS = ["test", "private_test"]


def load_reference_masks(reference_split_dir):
    metadata_path = reference_split_dir / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing reference metadata: {metadata_path}")

    sample_ids = []
    masks = {}

    with metadata_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = row["sample_id"]
            mask = Image.open(reference_split_dir / row["mask_path"]).convert("L")
            mask_array = (np.asarray(mask) > 0).astype(np.uint8)
            sample_ids.append(sample_id)
            masks[sample_id] = mask_array
    return sample_ids, masks


def dice_score(pred, gt):
    intersection = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return float((2.0 * intersection) / denom)


def jaccard_score(pred, gt):
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def score_split(reference_dir, prediction_dir, split_name):
    prediction_path = prediction_dir / f"{split_name}_predictions.npz"
    if not prediction_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {prediction_path}")

    sample_ids, gt_masks = load_reference_masks(reference_dir / split_name)
    npz_data = np.load(prediction_path)

    dices = []
    jaccards = []
    for sample_id in sample_ids:
        if sample_id not in npz_data:
            raise ValueError(f"Missing prediction for sample id '{sample_id}'.")

        pred = np.asarray(npz_data[sample_id])
        if pred.ndim == 3:
            pred = pred[..., 0]
        pred = (pred > 0).astype(np.uint8)

        gt = gt_masks[sample_id]
        if pred.shape != gt.shape:
            raise ValueError(
                f"Shape mismatch for sample '{sample_id}': "
                f"pred={pred.shape}, gt={gt.shape}"
            )

        dices.append(dice_score(pred, gt))
        jaccards.append(jaccard_score(pred, gt))

    return float(np.mean(dices)), float(np.mean(jaccards))


def main(reference_dir, prediction_dir, output_dir):
    scores = {}
    for eval_set in EVAL_SETS:
        print(f"Scoring {eval_set}")
        dice, jaccard = score_split(reference_dir, prediction_dir, eval_set)
        scores[f"{eval_set}_dice"] = dice
        scores[f"{eval_set}_jaccard"] = jaccard

    durations = json.loads((prediction_dir / "metadata.json").read_text())
    scores.update(**durations)
    print(scores)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "scores.json").write_text(json.dumps(scores))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Scoring program for codabench"
    )
    parser.add_argument(
        "--reference-dir",
        type=str,
        default="/app/input/ref",
        help="",
    )
    parser.add_argument(
        "--prediction-dir",
        type=str,
        default="/app/input/res",
        help="",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/app/output",
        help="",
    )

    args = parser.parse_args()

    main(
        Path(args.reference_dir),
        Path(args.prediction_dir),
        Path(args.output_dir)
    )
