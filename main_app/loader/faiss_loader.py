# loader/faiss_loader.py

import os
import faiss

from loader.config import (
    FAISS_INDEX_PATH,
    FAISS_DIMENSION
)

def load_faiss():

    if os.path.exists(
        FAISS_INDEX_PATH
    ):

        index = faiss.read_index(
            FAISS_INDEX_PATH
        )

        print(
            "Loaded existing FAISS index"
        )

    else:

        index = faiss.IndexFlatIP(
            FAISS_DIMENSION
        )

        print(
            "Created new FAISS index"
        )

    return index