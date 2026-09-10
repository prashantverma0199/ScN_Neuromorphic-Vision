```python
"""

Converts bounding-box annotations from XML format
to YOLO TXT format.

Only the classes listed in CLASS_NAMES are retained.
"""

import argparse
import os
import xml.etree.ElementTree as ET


# ============================================================
# CLASS CONFIGURATION
# ============================================================

CLASS_NAMES = ["person"]


# ============================================================
# XML -> YOLO CONVERSION
# ============================================================

def convert_xml_to_yolo(xml_path, label_path):
    """
    Convert a Pascal VOC XML annotation file to YOLO format.

    Parameters
    ----------
    xml_path : str
        Path to the Pascal VOC XML file.

    label_path : str
        Path where the YOLO TXT annotation will be saved.

    Notes
    -----
    YOLO format:

        class_id center_x center_y width height

    All bounding-box coordinates are normalized to the range
    [0, 1] relative to the image dimensions.
    """

    root = ET.parse(xml_path).getroot()

    size = root.find("size")

    if size is None:
        raise ValueError(
            f"Image size information not found in: {xml_path}"
        )

    width = float(size.find("width").text)
    height = float(size.find("height").text)

    if width <= 0 or height <= 0:
        raise ValueError(
            f"Invalid image dimensions in: {xml_path}"
        )

    labels = []

    # --------------------------------------------------------
    # Process each annotated object
    # --------------------------------------------------------

    for obj in root.findall("object"):

        name_element = obj.find("name")

        if name_element is None:
            continue

        class_name = name_element.text.strip()

        # Ignore classes not included in the project
        if class_name not in CLASS_NAMES:
            continue

        class_id = CLASS_NAMES.index(class_name)

        bbox = obj.find("bndbox")

        if bbox is None:
            continue

        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        # ----------------------------------------------------
        # Clamp coordinates to image boundaries
        # ----------------------------------------------------

        xmin = max(0.0, min(xmin, width))
        xmax = max(0.0, min(xmax, width))

        ymin = max(0.0, min(ymin, height))
        ymax = max(0.0, min(ymax, height))

        # Ignore invalid bounding boxes
        if xmax <= xmin or ymax <= ymin:
            continue

        # ----------------------------------------------------
        # Pascal VOC -> YOLO
        # ----------------------------------------------------

        center_x = (
            (xmin + xmax) / 2.0
        ) / width

        center_y = (
            (ymin + ymax) / 2.0
        ) / height

        box_width = (
            xmax - xmin
        ) / width

        box_height = (
            ymax - ymin
        ) / height

        labels.append(
            f"{class_id} "
            f"{center_x:.6f} "
            f"{center_y:.6f} "
            f"{box_width:.6f} "
            f"{box_height:.6f}"
        )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    output_dir = os.path.dirname(label_path)

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True
        )

    # --------------------------------------------------------
    # Save YOLO annotation
    # --------------------------------------------------------

    with open(
        label_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "\n".join(labels)
        )


# ============================================================
# COMMAND-LINE INTERFACE
# ============================================================

def parse_arguments():
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Convert Pascal VOC XML annotations "
            "to YOLO TXT format."
        )
    )

    parser.add_argument(
        "--xml",
        required=True,
        help="Path to the Pascal VOC XML file."
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path for the YOLO TXT annotation."
    )

    return parser.parse_args()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    args = parse_arguments()

    convert_xml_to_yolo(
        xml_path=args.xml,
        label_path=args.output
    )

    print(
        f"Annotation converted successfully:\n"
        f"{args.output}"
    )
```
