# loader/metadata_loader.py

import os
import json
import numpy as np

from loader.config import (
    EMBEDDINGS_PATH,
    METADATA_PATH
)

def load_embeddings():

    if os.path.exists(
        EMBEDDINGS_PATH
    ):

        embeddings = list(
            np.load(
                EMBEDDINGS_PATH
            )
        )

    else:

        embeddings = []

    return embeddings

def load_metadata():

    if os.path.exists(
        METADATA_PATH
    ):

        with open(
            METADATA_PATH,
            "r"
        ) as f:

            metadata = json.load(f)

    else:

        metadata = []

    return metadata

def build_existing_paths(
    metadata
):

    existing_paths = set()

    for item in metadata:

        existing_paths.add(
            item["original_path"]
        )

    return existing_paths