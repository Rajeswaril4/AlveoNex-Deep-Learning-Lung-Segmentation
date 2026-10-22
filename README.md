# AlveoNex: Automated Lung Field Segmentation from Chest X-rays Using Deep Learning

![Python](https://img.shields.io/badge/Python-3.x-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange) ![Architecture](https://img.shields.io/badge/Architecture-U--Net-purple) ![Platform](https://img.shields.io/badge/Platform-Google%20Colab-yellow) ![Web App](https://img.shields.io/badge/Web%20App-Streamlit-red)

**AlveoNex** is a deep learning project that automatically segments the left and right lung fields in chest X-ray images. It uses a U-Net convolutional neural network, image enhancement, binary segmentation masks, and morphological post-processing to produce lung-region masks and visual overlays.

> **Scope:** This project segments lung fields; it does **not** diagnose diseases or detect lung abnormalities. It is intended for learning and research, not clinical use.

## Features

- Matched chest X-ray and ground-truth lung mask loading
- CLAHE contrast enhancement and 256 × 256 image resizing
- Binary mask creation and train/validation/test splitting
- U-Net with encoder–decoder structure and skip connections
- Joint Binary Cross-Entropy (BCE) and Dice loss
- Paired image/mask augmentation during training
- Checkpointing, learning-rate scheduling, and training curves
- Horizontal-flip test-time augmentation (TTA)
- Morphological mask cleanup and connected-component filtering
- Single-image and batch prediction with visual overlays
- Ground-truth comparison using Dice and Intersection over Union (IoU)
- Optional learning-rate comparison experiment
- **Streamlit web app** for uploading X-rays, viewing lung masks and overlays, and downloading results

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| Deep learning | TensorFlow / Keras |
| Image processing | OpenCV, Pillow, SciPy |
| Data processing | NumPy, pandas |
| Visualization | Matplotlib |
| Dataset splitting | scikit-learn |
| Environment | Google Colab (GPU recommended) |
| Interactive web interface | Streamlit |

## Dataset

The development notebook uses **420 chest X-ray images and 420 corresponding lung segmentation masks** (420 matched pairs). The dataset is expected in a ZIP archive named `Data_Set.zip` with this structure:

```text
Data_Set.zip
└── Data_Set/
    ├── Model_Xrays/
    │   ├── image_001.png
    │   └── ...
    └── Masks/
        ├── image_001.png
        └── ...
```

Each mask should share the filename of its X-ray. Replace `image_001.png` with the actual filenames in your dataset. The notebook uploads and extracts the ZIP to Google Colab. Dataset files are excluded from Git using `.gitignore`; obtain and upload the dataset separately. **Dataset source/license:** add the original dataset citation and usage terms before publishing.

## Data Preparation

1. Match X-ray and mask filenames.
2. Convert X-rays to grayscale and resize to **256 × 256**.
3. Enhance contrast using **CLAHE**.
4. Standardize image intensities for model input.
5. Resize masks with nearest-neighbor interpolation and binarize them using a threshold of **127**.
6. Split the 420 image–mask pairs into **70% training / 15% validation / 15% testing** (294 / 63 / 63).

The notebook also displays sample X-rays, masks, and mask overlays for quality checking.

## Model Architecture

AlveoNex uses a **U-Net** architecture:

```text
Input X-ray (256 × 256 × 1)
           │
           ▼
Encoder: 32 → 64 → 128 → 256 filters
           │
           ▼
     Bottleneck: 512 filters
           │
           ▼
Decoder: 256 → 128 → 64 → 32 filters
       + encoder skip connections
           │
           ▼
  1-channel sigmoid output
           │
           ▼
  Lung segmentation probability map
```

The network contains approximately **7.77 million parameters** in the original project configuration. Sigmoid probabilities are converted into binary lung masks for visualization and evaluation.

## Training Configuration

| Setting | Value |
|---|---|
| Input size | 256 × 256 × 1 |
| Optimizer | Adam |
| Learning rate | 0.0003 in the main training run |
| Maximum epochs | 60 |
| Loss | BCE + Dice loss |
| Evaluation metrics | Dice score and Binary IoU |
| Checkpoint selection | Best validation Dice |
| Scheduler | ReduceLROnPlateau |

Training uses image/mask augmentation. An optional experiment compares learning rates **0.001, 0.0003, and 0.0001** with early stopping; it is separate from the main run.

## Results

The original project notebook reports the following result for one completed main training run:

| Metric | Test result |
|---|---:|
| Dice score | **97.47%** |
| Test loss | **0.0718** |

Another recorded run produced a **97.31% Dice score** and **0.0770 test loss**. Results may vary with dataset splits, random initialization, and runtime conditions. The displayed metrics are notebook-reported results, **not independently reproduced** here. For a fully reproducible benchmark, document the random seed, exact dataset version, package versions, and saved checkpoint.

### Segmentation Visualization

For an uploaded chest X-ray, the notebook can show:

- Original/preprocessed X-ray
- Predicted lung probability map
- Post-processed binary lung mask
- Lung-mask overlay on the X-ray

For images with ground-truth masks, the error overlay indicates:

- **Green:** correctly segmented lung region
- **Red:** missed lung region
- **Blue:** extra predicted region

> Add screenshots to an `assets/` folder if you want visual examples in this README. Do not commit patient-identifying images or data without authorization.

## How to Run in Google Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload `AlveoNex_Lung_Segmentation_UNet.ipynb` (or the included notebook filename).
3. Select **Runtime → Change runtime type → GPU** if available.
4. Run the cells from top to bottom.
5. When prompted, upload `Data_Set.zip` with the folder structure above.
6. Inspect preprocessing samples, then train the U-Net model.
7. Review training/validation curves and evaluate the test split.
8. Upload one or more X-ray images to generate masks and overlays.
9. Optionally upload corresponding ground-truth masks for Dice/IoU comparison.
10. Save and download the trained `.keras` model from the final notebook cell.

**Note:** The learning-rate search is optional and retrains models. Run it only when comparing configurations; keep track of which trained model you use for later predictions and exports.

## Run Locally (Optional)

The provided notebook is designed for **Google Colab**, including `google.colab.files.upload()` and `files.download()` calls. Running it locally requires replacing those upload/download calls with filesystem paths and installing dependencies:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

pip install tensorflow opencv-python numpy pandas matplotlib pillow scipy scikit-learn jupyter
jupyter notebook
```

GPU support and TensorFlow installation requirements depend on your operating system and hardware.

## Streamlit Web Application

AlveoNex also includes a **Streamlit interface** (`app.py`) to run inference on new chest X-ray images using the **trained U-Net model**. The web app does not train a model and does not provide a medical diagnosis.

### Streamlit Features

- Upload one or multiple PNG/JPG chest X-rays.
- Apply the **same 256 × 256, CLAHE, and z-score preprocessing** used during model training.
- Predict a lung probability map, optionally averaging horizontal-flip predictions (TTA).
- Clean the binary mask using morphological operations and the two largest connected components.
- Display the X-ray, probability map, lung mask, and colored overlay side by side.
- Adjust the mask threshold and download mask/overlay PNG files.

### Run Streamlit Locally

1. Train the model using the notebook and save the trained Keras model as `lung_unet_final.keras`.
2. Create a `models/` folder and copy the trained model into `models/lung_unet_final.keras` (**this file is not included in GitHub because `.gitignore` excludes model checkpoints**).
3. Install dependencies and start the app:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

4. Open the local address printed by Streamlit (usually `http://localhost:8501`), upload an X-ray, and inspect the segmentation.

If you saved your model elsewhere, enter its path in the app's sidebar. The app loads `.keras` models with `compile=False`, so training-only custom losses and metrics do not need to be registered for inference. Ensure the exported model is compatible with the installed TensorFlow/Keras version.

**Deployment note:** Hosting requires making the trained model accessible to the deployed app through a permitted model artifact source or secure deployment process. Do not commit large model weights or patient data to the public repository. TensorFlow can also require significant RAM on small free hosting instances.

## Repository Structure

```text
AlveoNex-Automated-Lung-Field-Segmentation/
├── README.md
├── .gitignore
├── requirements.txt
├── app.py                     # Streamlit inference web app
├── models/                    # Local model weights (ignored by Git)
├── AlveoNex_Lung_Segmentation_UNet.ipynb
└── assets/                  # Optional: project screenshots
```

> Rename your notebook to match the filename above, or update this README to match the actual committed notebook.

## Evaluation Metrics

**Dice Similarity Coefficient** measures overlap between the predicted lung region \(P\) and ground-truth region \(G\):

\[
\mathrm{Dice} = \frac{2|P \cap G|}{|P| + |G|}
\]

**Intersection over Union (IoU)** measures the ratio of overlap to union:

\[
\mathrm{IoU} = \frac{|P \cap G|}{|P \cup G|}
\]

Higher values indicate better segmentation overlap.

## Limitations and Future Work

- Results are based on a limited dataset; external validation is needed before broader claims.
- Segmentation performance may decrease on X-rays from different scanners or populations.
- Masks can contain small boundary errors, especially near the lung edges.
- Potential improvements include larger and more diverse datasets, cross-validation, uncertainty estimation, and comparison with other segmentation architectures.

## GitHub Setup

```bash
git init
git add .
git commit -m "Initial commit: AlveoNex U-Net lung segmentation"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/AlveoNex-Automated-Lung-Field-Segmentation.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username and create the empty repository before pushing. The `.gitignore` should exclude raw datasets, large checkpoints, secrets, and temporary outputs.

## Disclaimer

AlveoNex is an educational/research computer-vision project. Its segmentation output is **not a medical diagnosis** and should not be used for clinical decision-making without appropriate validation and regulatory review.
