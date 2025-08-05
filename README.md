# Automated Detection of MSX in Oyster Histology

This repository contains the source code and methodology for a project focused on detecting _Haplosporidium nelsoni_ (MSX) in Whole Slide Images (WSIs) of oyster tissue.

---

## Project Overview

The goal of this project is to develop a computational tool to augment the diagnostic workflow for MSX disease. By automating the detection of MSX plasmodia, this tool aims to help experts prioritize slides and rapidly locate areas of interest, thereby increasing the efficiency and throughput of oyster health surveillance.

The project is structured as a series of modular scripts that handle dataset creation, model training, inference, and results formatting.

### Key Features

- **Automated Dataset Creation:** Scripts to process QuPath annotations into a YOLO-compatible, patch-based dataset.
- **Model Training:** A streamlined training pipeline using the `ultralytics` framework with pre-trained YOLOv8 models.
- **Scalable Inference:** A robust inference script to run the trained model on new, full WSIs.
- **QuPath Integration:** A final script to convert model predictions back into a GeoJSON format for easy validation and review in QuPath.

---

## Project Structure

The repository is organized as follows:
```bash
oyster_msx_project/
├── data/                 # Data is gitignored; this shows the expected structure
│ ├── raw/wsis/           # Store original WSI files here
│ └── interim/qupath_exports/ # Store GeoJSON annotations from QuPath here
├── src/                  # All Python source code
│ ├── main_scripts/       # Core pipeline scripts (01 to 04)
│ └── utils/              # Helper modules for logging, file handling, etc.
├── config.yaml           # Central configuration file for all parameters
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/JonathanZul/oyster_msx_project.git
    cd oyster_msx_project
    ```
2.  **Create and activate a Python virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Add Data:**
    -   Place your Whole Slide Image files (`.tif`) into the `data/raw/wsis/` directory.
        - NOTE: The project uses Tifffile format for WSIs, which does not support `.vsi` files. If you have `.vsi` files or other format, convert them to `.tif` using QuPath or another tool.
    -   Export your annotations from QuPath as `.geojson` files into `data/interim/qupath_exports/`.

---

## The Pre-processing Pipeline: Oyster Segmentation

Before the main machine learning pipeline can run, we must first identify and separate the individual oyster sections on each Whole Slide Image.

### First Method: Classical Computer Vision

The primary, validated method for this task uses a classical computer vision script.

-   **Script:** `archive/00_segment_oysters_classical.py` (Archived)
-   **Purpose:** To automatically generate a precise binary mask for each of the two main oyster sections on a raw WSI.
-   **Method:** The script uses a series of standard computer vision techniques (thresholding, morphological operations) to create an initial tissue mask. To separate touching oysters, it employs the **Watershed Algorithm**. After separation, it intelligently merges significant tissue fragments (like separated gills) back to their closest parent oyster.
-   **Output:** The script saves one `.png` mask file for each detected oyster into the `data/interim/oyster_masks/` directory. These masks are used by subsequent scripts.

### Second Method: ML-Based Segmentation (U-Net)

As part of our research and development, we also implemented and evaluated a complete deep learning pipeline for the segmentation task.

-   **Summary:** This approach replaced the classical script with a three-part workflow to create a dataset (`00a`), train a model (`00b`), and run inference (`00`). It uses a U-Net architecture with a pre-trained ResNet34 encoder and employs advanced techniques like loss masking and two-phase fine-tuning.
-   **Outcome:** The pipeline was successfully built and initial model training was achieved. However, performance with the limited dataset (22 slides) did not yet surpass the classical method in boundary precision.
-   **Status:** All scripts (`00a`, `00b`, `00`, and tools) and utilities for this approach have been cleaned, documented, and are committed to the repository for future reference or development. The project is now proceeding with exploring a different ML architecture.

### Third Method: Segment Anything Model (SAM)
Third method, currently under development, uses the Segment Anything Model (SAM) to generate oyster masks. This method is still experimental and not yet integrated into the main pipeline.

---

## Usage: The Data Pipeline

The project is designed to be run as a sequence of scripts. Configure all parameters in `config.yaml` before running. The oyster segmentation step is not included in the main pipeline, as it is assumed that the oyster masks are already available in `data/interim/oyster_masks/`.

The project makes use of some Pytorch functions not yet available on macOS with MPS backend. If you are running on macOS, you can set the environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1` to use CPU fallback for these functions.

1.  **Create the Dataset:** Generate image patches and labels for YOLO.
    ```bash
    python -m src.main_scripts.01_create_dataset
    ```
2.  **Train the Model:** Train the YOLO model on the created dataset.
    ```bash
    python -m src.main_scripts.02_train_yolo
    ```
3.  **Run Inference:** Use the trained model to predict on a new WSI.
    ```bash
    python -m src.main_scripts.03_run_inference
    ```
4.  **Format Predictions:** Convert the raw predictions into a QuPath-compatible GeoJSON file.
    ```bash
    python -m src.main_scripts.04_format_predictions
    ```

---

## Acknowledgements
