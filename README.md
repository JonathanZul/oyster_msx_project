# Automated Detection of MSX in Oyster Histology

This repository contains the source code and methodology for a project focused on detecting _Haplosporidium nelsoni_ (MSX) in Whole Slide Images (WSIs) of oyster tissue.

---

## Project Overview

The goal of this project is to develop a computational tool to augment the diagnostic workflow for MSX disease. By automating the detection of MSX plasmodia, this tool aims to help experts prioritize slides and rapidly locate areas of interest, thereby increasing the efficiency and throughput of oyster health surveillance.

The project is structured as a series of modular scripts that handle the end-to-end machine learning pipeline.

### Key Features

-   **Robust Pre-processing:** A sophisticated, hybrid pipeline using the Segment Anything Model (SAM) to accurately segment the two main oyster sections on a WSI.
-   **Automated Dataset Creation:** Scripts to process QuPath annotations into a YOLO-compatible, patch-based dataset.
-   **Model Training:** A streamlined training pipeline using the `ultralytics` framework with pre-trained YOLOv8 models.
-   **Scalable Inference:** A robust inference script to run the trained model on new, full WSIs.
-   **QuPath Integration:** A final script to convert model predictions back into a GeoJSON format for easy validation and review in QuPath.

---

## Project Structure

The repository is organized for clarity and reproducibility:

```bash
oyster_msx_project/
├── data/             # (Gitignored) Contains all data, organized by processing stage
│ ├── raw/wsis/       # Original WSI files
│ └── interim/        # Intermediate data (QuPath exports, oyster masks)
├── src/              # All Python source code
│ ├── main_scripts/   # Core pipeline scripts (00 to 04)
│ ├── utils/          # Helper modules for logging, file handling, etc.
│ └── tools/          # Additional tools for evaluation and visualization.
├── archive/          # Older, deprecated experimental scripts
├── config.yaml       # Central configuration file for all parameters
├── requirements.txt  # Python dependencies
└── README.md         # This file
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
    -   Place your Whole Slide Image files (`.tif`, `.tiff`, etc.) into the `data/raw/wsis/` directory.
    -   Export your MSX annotations from QuPath as `.geojson` files into `data/interim/qupath_exports/`.

---

## The Pipeline: From WSI to Detections

The project is designed to be run as a sequence of five scripts, all controlled by the central `config.yaml` file.

### Stage 0: Oyster Instance Segmentation with SAM

The critical first step is to separate the two oyster sections on each slide. The current primary method is a robust, hybrid pipeline that leverages the Segment Anything Model (SAM).

-   **Script:** `src/main_scripts/00_segment_with_sam.py`
-   **Method:**
    1.  **Robust Prompt Generation:** The script first uses a classical computer vision pipeline to find all tissue fragments on a low-resolution overview of the WSI. It then uses **K-Means clustering** to intelligently group these fragments into two distinct clusters, representing the two oysters. This method is highly robust to variations in tissue placement and fragmentation (e.g., disconnected gills).
    2.  **SAM-Powered Segmentation:** For each oyster, it generates a "rich prompt package"—containing a bounding box, a positive point at the cluster center, and negative points—and feeds it to the pre-trained SAM.
    3.  **Intelligent Mask Selection:** The script instructs SAM to generate three candidate masks and then programmatically selects the best one by comparing each candidate against a coarse CV-generated tissue map, choosing the mask with the highest Dice score.
    4.  **Mask Repair:** A final post-processing step finds any tissue fragments that were missed by the initial SAM segmentation and intelligently merges them back into the closest parent mask, ensuring the final masks are complete.
-   **To Run:**
    ```bash
    python -m src.main_scripts.00_segment_with_sam
    ```

### Stages 1-4: The MSX Detection Pipeline

After the oyster masks have been generated, the main YOLO-based pipeline can be run.

1.  **`01_create_dataset.py`:** Generate image patches and labels for YOLO.
2.  **`02_train_yolo.py`:** Train the YOLOv8 model on the created dataset.
3.  **`03_run_inference.py`:** Use the trained model to predict on a new WSI.
4.  **`04_format_predictions.py`:** Convert the raw predictions into a QuPath-compatible GeoJSON file.

-   **To Run:**
    ```bash
    # Ensure PYTORCH_ENABLE_MPS_FALLBACK=1 is set if on macOS
    python -m src.main_scripts.01_create_dataset
    python -m src.main_scripts.02_train_yolo
    python -m src.main_scripts.03_run_inference
    python -m src.main_scripts.04_format_predictions
    ```

---

## Segmentation Method Comparison

Several methods for the initial oyster segmentation have been developed and evaluated. The current hybrid SAM approach (V8) provides the best balance of performance and robustness.

| Method                       | Avg. IoU Score | Avg. Dice Score | Avg. J&F Score | Notes |
|:-----------------------------|:--------------:|:---------------:| :---: | :--- |
| **Classical CV** (Watershed) |     x.xxxx     |     x.xxxx      | x.xxxx | Fast but brittle. Fails on complex slides and highly sensitive to parameters. |
| **U-Net** (ResNet34 Backend) |     x.xxxx     |     x.xxxx      | x.xxxx | Required training. Performance limited by the small dataset (22 slides), leading to overfitting. |
| **Hybrid SAM**               |   **0.8706**   |   **0.9287**    | **0.8996** | Slower but highly robust. Intelligently handles touching and fragmented tissue. **Current recommended method.** |

*Scores evaluated on a consistent set of 21 annotated slides using `src/tools/evaluate_segmentation.py`.*

---

## Acknowledgements
