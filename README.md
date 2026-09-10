# ScN-Based Neuromorphic Vision

Computational framework for exploring **Scandium Nitride (ScN)-based neuromorphic vision** using experimentally inspired photoresponse transformation and YOLO-based infrared object detection.

## Dataset

This project uses the **LLVIP infrared dataset**. The dataset is not included in this repository.

## Repository Structure

```text
ScN-Based-Neuromorphic-Vision/
│
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── scn/
│   │   └── scn_transform.py
│   ├── dataset/
│   │   └── convert_annotations.py
│   ├── training/
│   │   └── train.py
│   └── inference/
│       └── video_detection.py
│
├── notebooks/
│   └── ScN_IR_YOLO_training.ipynb
│
├── configs/
│   └── example.yaml
│
├── data/
│   └── README.md
│
├── weights/
│   └── README.md
│
└── results/
    └── README.md
```

## Experimental Calibration

The transformation is based on experimentally measured ScN photoresponse. Device-specific calibration parameters are not included in this public repository.

## Installation

```bash
pip install -r requirements.txt
```

## License

See `LICENSE` for details.

