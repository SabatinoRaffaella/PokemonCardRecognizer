from PIL import (
    ImageEnhance,
    ImageFilter
)

def augment_image(image):

    variations = []

    # ------------------------
    # ORIGINAL
    # ------------------------

    variations.append(
        ("original", image)
    )

    # ------------------------
    # DARKER
    # ------------------------

    dark = ImageEnhance.Brightness(
        image
    ).enhance(0.7)

    variations.append(
        ("dark", dark)
    )

    # ------------------------
    # BRIGHTER
    # ------------------------

    bright = ImageEnhance.Brightness(
        image
    ).enhance(1.3)

    variations.append(
        ("bright", bright)
    )

    # ------------------------
    # BLUR
    # ------------------------

    blur = image.filter(
        ImageFilter.GaussianBlur(1.5)
    )

    variations.append(
        ("blur", blur)
    )

    # ------------------------
    # ROTATE
    # ------------------------

    rotate = image.rotate(5)

    variations.append(
        ("rotate", rotate)
    )

    return variations