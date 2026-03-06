"""Prepare OncoSegment data from Hugging Face for the Codabench template."""

import argparse
import csv
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import ClassLabel, Image as HFImage, load_dataset
from PIL import Image

PHASE = "dev_phase"

DATA_DIR = Path(PHASE) / "input_data"
REF_DIR = Path(PHASE) / "reference_data"
DATASET_ID = "gymprathap/Breast-Cancer-Ultrasound-Images-Dataset"
TARGET_CLASSES = {"benign", "malignant"}


def pick_column(columns, candidates, kind):
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(f"Could not find a {kind} column in dataset columns: {columns}")


def normalize_label(value, feature):
    if isinstance(feature, ClassLabel):
        return feature.int2str(int(value)).strip().lower()
    return str(value).strip().lower()


def split_stratified(records, seed, train_ratio=0.7, test_ratio=0.15):
    rng = np.random.default_rng(seed)
    per_label_indices = defaultdict(list)
    for idx, record in enumerate(records):
        per_label_indices[record["class_label"]].append(idx)

    split_indices = {"train": [], "test": [], "private_test": []}
    for indices in per_label_indices.values():
        idxs = np.array(indices)
        rng.shuffle(idxs)
        n_total = len(idxs)

        n_train = int(round(train_ratio * n_total))
        n_test = int(round(test_ratio * n_total))
        n_train = min(max(n_train, 0), n_total)
        n_test = min(max(n_test, 0), n_total - n_train)
        n_private = n_total - n_train - n_test

        if n_total >= 3:
            if n_train == 0:
                n_train = 1
            if n_test == 0:
                n_test = 1
            n_private = n_total - n_train - n_test
            if n_private <= 0:
                if n_train >= n_test and n_train > 1:
                    n_train -= 1
                elif n_test > 1:
                    n_test -= 1
                n_private = n_total - n_train - n_test

        split_indices["train"].extend(idxs[:n_train].tolist())
        split_indices["test"].extend(idxs[n_train : n_train + n_test].tolist())
        split_indices["private_test"].extend(idxs[n_train + n_test :].tolist())

    for split_name in split_indices:
        rng.shuffle(split_indices[split_name])

    return {
        split_name: [records[idx] for idx in idxs]
        for split_name, idxs in split_indices.items()
    }


def write_csv(rows, path, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_split(split_name, records):
    split_input_dir = DATA_DIR / split_name
    image_dir = split_input_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    input_rows = []
    ref_rows = []

    if split_name == "train":
        input_mask_dir = split_input_dir / "masks"
        input_mask_dir.mkdir(parents=True, exist_ok=True)
    else:
        ref_mask_dir = REF_DIR / split_name / "masks"
        ref_mask_dir.mkdir(parents=True, exist_ok=True)

    for record in records:
        sample_id = record["sample_id"]
        image_name = f"{sample_id}.png"
        mask_name = f"{sample_id}.png"

        image_path = image_dir / image_name
        record["image"].save(image_path)

        if split_name == "train":
            mask_path = input_mask_dir / mask_name
            record["mask"].save(mask_path)
            input_rows.append(
                {
                    "sample_id": sample_id,
                    "image_path": f"images/{image_name}",
                    "mask_path": f"masks/{mask_name}",
                    "class_label": record["class_label"],
                }
            )
        else:
            mask_path = ref_mask_dir / mask_name
            record["mask"].save(mask_path)
            input_rows.append(
                {
                    "sample_id": sample_id,
                    "image_path": f"images/{image_name}",
                }
            )
            ref_rows.append(
                {
                    "sample_id": sample_id,
                    "mask_path": f"masks/{mask_name}",
                    "class_label": record["class_label"],
                }
            )

    if split_name == "train":
        write_csv(
            input_rows,
            split_input_dir / "metadata.csv",
            ["sample_id", "image_path", "mask_path", "class_label"],
        )
    else:
        write_csv(
            input_rows,
            split_input_dir / "metadata.csv",
            ["sample_id", "image_path"],
        )
        write_csv(
            ref_rows,
            REF_DIR / split_name / "metadata.csv",
            ["sample_id", "mask_path", "class_label"],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and format the OncoSegment data for Codabench."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--dataset-id",
        type=str,
        default=DATASET_ID,
        help="Hugging Face dataset repository id",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to load from Hugging Face",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional cap on total number of filtered samples (0 means all).",
    )
    args = parser.parse_args()

    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    if REF_DIR.exists():
        shutil.rmtree(REF_DIR)

    print(f"Loading dataset: {args.dataset_id} (split={args.split})")
    dataset = load_dataset(args.dataset_id, split=args.split)

    columns = dataset.column_names
    image_col = pick_column(columns, ["image", "img"], "image")
    label_col = pick_column(
        columns,
        ["label", "class", "category", "tumor_type", "diagnosis"],
        "label",
    )
    mask_col = None
    for candidate in ["mask", "segmentation_mask", "label_mask", "annotation"]:
        if candidate in columns:
            mask_col = candidate
            break

    label_feature = dataset.features.get(label_col)
    records = []
    if mask_col is not None:
        for i, row in enumerate(dataset):
            class_label = normalize_label(row[label_col], label_feature)
            if class_label not in TARGET_CLASSES:
                continue

            image = row[image_col].convert("L")
            mask = row[mask_col].convert("L")
            if mask.size != image.size:
                mask = mask.resize(image.size, Image.NEAREST)

            mask_arr = (np.asarray(mask) > 0).astype(np.uint8) * 255
            mask = Image.fromarray(mask_arr)

            records.append(
                {
                    "sample_id": f"sample_{i:05d}",
                    "class_label": class_label,
                    "image": image,
                    "mask": mask,
                }
            )
    else:
        # Some exports contain only one image column where masks are separate files
        # named with the same stem + "_mask".
        dataset_paths = dataset.cast_column(image_col, HFImage(decode=False))
        paired = {}
        mask_regex = re.compile(r"_mask(?=\.[^.]+$)")
        for i, row in enumerate(dataset):
            class_label = normalize_label(row[label_col], label_feature)
            if class_label not in TARGET_CLASSES:
                continue

            path = dataset_paths[i][image_col]["path"]
            internal_path = path.split("::", maxsplit=1)[0]
            pair_key = mask_regex.sub("", internal_path)

            entry = paired.setdefault(pair_key, {"class_label": class_label})
            image = row[image_col].convert("L")
            if mask_regex.search(Path(internal_path).name):
                entry["mask"] = image
            else:
                entry["image"] = image
                entry["class_label"] = class_label

        for i, pair_key in enumerate(sorted(paired)):
            entry = paired[pair_key]
            if "image" not in entry or "mask" not in entry:
                continue

            image = entry["image"]
            mask = entry["mask"]
            if mask.size != image.size:
                mask = mask.resize(image.size, Image.NEAREST)
            mask_arr = (np.asarray(mask) > 0).astype(np.uint8) * 255
            mask = Image.fromarray(mask_arr)

            records.append(
                {
                    "sample_id": f"sample_{i:05d}",
                    "class_label": entry["class_label"],
                    "image": image,
                    "mask": mask,
                }
            )

    if args.max_samples > 0 and args.max_samples < len(records):
        rng = np.random.default_rng(args.seed)
        selected = rng.choice(len(records), size=args.max_samples, replace=False)
        records = [records[idx] for idx in selected]

    if not records:
        raise RuntimeError(
            "No samples found after filtering benign/malignant classes. "
            "Please check the dataset schema."
        )

    splits = split_stratified(records, seed=args.seed)
    for split_name, split_records in splits.items():
        if not split_records:
            raise RuntimeError(f"Split '{split_name}' is empty.")
        save_split(split_name, split_records)
        print(f"{split_name}: {len(split_records)} samples")

    print(f"Data prepared in: {DATA_DIR} and {REF_DIR}")
