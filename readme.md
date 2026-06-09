This is a pokémon card recognizer that uses both Ocr and Embeddings to determine a given card.

## 🛠️ Tech Stack
* **Language:** Python
* **Model:** OpenCLIP (ViT-B-32 backbone, pretrained on laion2b_s34b_b79k)
* **Vector Database:** Meta's FAISS (Facebook AI Similarity Search)
* **Data Handling:** OpenCV, NumPy, JSON (Metadata)

## 📐 System Architecture

The project is structured following clean-code principles and modular software engineering, split into three main components:

### 1. 📂 Loader Module (The Data & Model Pipeline)
Responsible for bootstrapping the system environment and managing core assets:
* **Model Initialization:** Loads and configures the pre-trained OpenCLIP ViT-B-32 model and its required preprocessing transforms.
* **Vector Indexing:** Loads pre-calculated visual embeddings into a **FAISS index** for ultra-fast K-Nearest Neighbors (KNN) similarity lookup.
* **Metadata Alignment:** Loads JSON metadata mapping index positions back to explicit card details (Name, ID, Rarity).

### 2. 🗂️ Cards Module (The Dataset Registry)
Acts as the central configuration layer for tracking and organizing expansion sets:
* Manages references to the specific card sets currently registered in the system.
* Standardizes data directories, making it frictionless to append new releases or entire new card generations without breaking existing infrastructure.

### 3. 🧠 Recognizer Module (The Inference Engine)
The core algorithmic execution layer that processes user inputs:
* **Image Preprocessing:** Captures target images and subjects them to the exact OpenCLIP transformation matrix (normalization and scaling).
* **Embedding Extraction:** Passes the processed image through the Vision Transformer to generate a 512-dimensional feature vector.
* **Vector Querying:** Queries the FAISS index using Cosine Similarity / L2 Distance to find the best match in sub-millisecond time.

\-It's hosted on Google Colab using Gradio to fetch input images to test.

-Link to Colab Notebook: https://colab.research.google.com/drive/1Hxacxs1_Qh02n3b589_KrGknRjOe2Pad
