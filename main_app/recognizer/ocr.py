# recognizer/ocr.py

import numpy as np

def extract_ocr_regions(
    image,
    reader
):

    w, h = image.size

    # ------------------------
    # NAME REGION
    # ------------------------

    name_crop = image.crop((
        0,
        0,
        w,
        int(h * 0.18)
    ))

    # ------------------------
    # BOTTOM REGION
    # ------------------------

    bottom_crop = image.crop((
        0,
        int(h * 0.82),
        w,
        h
    ))

    # ------------------------
    # OCR
    # ------------------------

    name_text = reader.readtext(
        np.array(name_crop),
        detail=0
    )

    bottom_text = reader.readtext(
        np.array(bottom_crop),
        detail=0
    )

    return {

        "name_text": name_text,

        "bottom_text": bottom_text
    }