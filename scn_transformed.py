```python
"""
ScN-inspired transformation of infrared images.

This module provides the framework used to transform an IR image
into a normalized ScN-inspired image representation.

The experimentally measured ScN device calibration parameters are
intentionally not included in this public repository.
"""

import numpy as np


def scn_transform_ir(ir_image, response_function):
    """
    Transform an infrared image using an externally supplied
    ScN photoresponse model.

    Parameters
    ----------
    ir_image : numpy.ndarray
        Single-channel infrared image.

    response_function : callable
        Experimentally calibrated ScN response function.

    Returns
    -------
    scn_image : numpy.ndarray
        Normalized ScN-inspired image.

    delta_I : numpy.ndarray
        ScN photoresponse corresponding to the input image.

    Notes
    -----
    The original implementation maps IR intensity to an optical
    excitation level and then to the experimentally measured ScN
    negative photoresponse. The device-specific calibration used
    for this mapping is not included here.
    """

    if ir_image is None:
        raise ValueError("IR image is None.")

    if ir_image.ndim != 2:
        raise ValueError(
            f"Expected single-channel IR image, "
            f"got shape {ir_image.shape}"
        )

    # Original IR intensity
    intensity = ir_image.astype(np.float32)

    # --------------------------------------------------------
    # Apply externally supplied ScN response model
    # --------------------------------------------------------

    delta_I = response_function(intensity)

    # --------------------------------------------------------
    # Normalize photoresponse to 8-bit image range
    # --------------------------------------------------------

    response_min = np.min(delta_I)
    response_max = np.max(delta_I)

    if response_max == response_min:
        scn_image = np.zeros_like(
            ir_image,
            dtype=np.uint8
        )

    else:
        scn_image = (
            (delta_I - response_min)
            / (response_max - response_min)
        ) * 255.0

        scn_image = np.clip(
            scn_image,
            0,
            255
        ).astype(np.uint8)

    return scn_image, delta_I
```
