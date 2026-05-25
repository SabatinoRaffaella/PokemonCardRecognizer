# recognizer/embedding.py

import os
import json
import faiss
import torch
import numpy as np

from PIL import Image
from tqdm import tqdm

from recognizer.ocr import (
    extract_ocr_regions
)

from recognizer.augment import (
    augment_image
)

def build_embeddings(
    model,
    preprocess,
    reader,
    index,
    embeddings,
    metadata,
    existing_paths
):

    cards_root = "cards"

    new_embeddings = []
    new_metadata = []

    new_count = 0
    skip_count = 0

    for root, dirs, files in os.walk(cards_root):

        for file in tqdm(files):

            if not file.endswith((
                ".png",
                ".jpg",
                ".jpeg"
            )):
                continue

            path = os.path.join(
                root,
                file
            )

            # ------------------------
            # SKIP EXISTING
            # ------------------------

            if path in existing_paths:
                skip_count += 1
                continue

            try:

                image = Image.open(
                    path
                ).convert("RGB")

                # ------------------------
                # OCR
                # ------------------------

                ocr_data = extract_ocr_regions(
                    image,
                    reader
                )

                # ------------------------
                # AUGMENTATIONS
                # ------------------------

                variations = augment_image(
                    image
                )

                # ------------------------
                # EMBED EACH VARIATION
                # ------------------------

                for aug_name, aug_img in variations:

                    tensor = preprocess(
                        aug_img
                    ).unsqueeze(0)

                    with torch.no_grad():

                        features = model.encode_image(
                            tensor
                        )

                    features /= features.norm(
                        dim=-1,
                        keepdim=True
                    )

                    vector = (
                        features[0]
                        .cpu()
                        .numpy()
                        .astype("float32")
                    )

                    embeddings.append(
                        vector
                    )

                    new_embeddings.append(
                        vector
                    )

                    new_metadata.append({

                        "original_path": path,

                        "augmentation": aug_name,

                        "ocr_name": ocr_data[
                            "name_text"
                        ],

                        "ocr_bottom": ocr_data[
                            "bottom_text"
                        ]
                    })

                new_count += 1

            except Exception as e:

                print("ERROR:", path)
                print(e)

    # =====================================
    # UPDATE METADATA
    # =====================================

    metadata.extend(
        new_metadata
    )

    # =====================================
    # UPDATE FAISS
    # =====================================

    if len(new_embeddings) > 0:

        new_embeddings = np.array(
            new_embeddings
        ).astype("float32")

        index.add(
            new_embeddings
        )

        print(
            f"Added {len(new_embeddings)} vectors"
        )

    else:

        print("No new cards")
    # ============================================
    # SAVE EVERYTHING
    # ============================================

    embeddings = np.array(
        embeddings
    ).astype("float32")

    np.save(
        "embeddings.npy",
        embeddings
    )

    with open("metadata.json", "w") as f:

        json.dump(
            metadata,
            f,
            indent=2
        )

    faiss.write_index(
        index,
        "pokemon.index"
    )
    # =====================================
    # FINAL VALIDATION
    # =====================================

    assert len(metadata) == index.ntotal

    print("\n===== DONE =====")

    print(
        "New cards:",
        new_count
    )

    print(
        "Skipped:",
        skip_count
    )

    print(
        "Total metadata:",
        len(metadata)
    )

    print(
        "FAISS vectors:",
        index.ntotal
    )

    return (
        embeddings,
        metadata,
        index
    )