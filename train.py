```python
"""
YOLO training script for ScN-based neuromorphic vision.

This script trains a YOLO object-detection model using a prepared
ScN-inspired dataset.

The dataset and model paths are supplied through command-line
arguments so that no machine-specific paths are stored in the
repository.
"""

import argparse
import os

from ultralytics import YOLO


# ============================================================
# TRAINING
# ============================================================

def train_model(
    data_yaml,
    model_name="yolov5s.pt",
    epochs=50,
    image_size=416,
    batch_size=32,
    device="0",
    workers=4,
    project="results",
    run_name="scn_yolo",
):
    """
    Train a YOLO object-detection model.

    Parameters
    ----------
    data_yaml : str
        Path to the YOLO dataset YAML file.

    model_name : str
        YOLO model/checkpoint to use for training.

    epochs : int
        Number of training epochs.

    image_size : int
        Input image size for training.

    batch_size : int
        Training batch size.

    device : str
        Training device.
        Examples:
            "0"        -> GPU 0
            "0,1"      -> GPUs 0 and 1
            "cpu"      -> CPU

    workers : int
        Number of dataloader workers.

    project : str
        Directory in which training results are stored.

    run_name : str
        Name of the training run.

    Returns
    -------
    object
        Ultralytics training results.
    """

    if not os.path.isfile(data_yaml):
        raise FileNotFoundError(
            f"Dataset YAML file not found:\n{data_yaml}"
        )

    print("\nLoading model...")
    model = YOLO(model_name)

    print(f"Model       : {model_name}")
    print(f"Dataset     : {data_yaml}")
    print(f"Epochs      : {epochs}")
    print(f"Image size  : {image_size}")
    print(f"Batch size  : {batch_size}")
    print(f"Device      : {device}")

    print("\nStarting training...\n")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        device=device,
        amp=True,
        project=project,
        name=run_name,
        workers=workers,
        save=True,
        pretrained=True,
    )

    print("\nTraining complete.")

    return results


# ============================================================
# COMMAND-LINE ARGUMENTS
# ============================================================

def parse_arguments():
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Train a YOLO model on an "
            "ScN-inspired object-detection dataset."
        )
    )

    parser.add_argument(
        "--data",
        required=True,
        help="Path to the YOLO dataset YAML file.",
    )

    parser.add_argument(
        "--model",
        default="yolov5s.pt",
        help="YOLO model/checkpoint.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--img-size",
        type=int,
        default=416,
        help="Training image size.",
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=32,
        help="Training batch size.",
    )

    parser.add_argument(
        "--device",
        default="0",
        help=(
            "Training device. "
            "Examples: 0, 0,1, cpu"
        ),
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of dataloader workers.",
    )

    parser.add_argument(
        "--project",
        default="results",
        help="Training output directory.",
    )

    parser.add_argument(
        "--name",
        default="scn_yolo",
        help="Training run name.",
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    args = parse_arguments()

    train_model(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        image_size=args.img_size,
        batch_size=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        run_name=args.name,
    )
```
