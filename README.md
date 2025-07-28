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

```
oyster_msx_project/
├── data/                 # Data is gitignored; this shows the expected structure
│   ├── raw/wsis/         # Store original WSI files here
│   └── interim/qupath_exports/ # Store GeoJSON annotations from QuPath here
├── src/                  # All Python source code
│   ├── main_scripts/     # Core pipeline scripts (01 to 04)
│   └── utils/            # Helper modules for logging, file handling, etc.
├── config.yaml           # Central configuration file for all parameters
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## Setup & Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/JonathanZul/oyster_msx_project.git
   cd oyster_msx_project
   ```

2. **Create and activate a Python virtual environment:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the required dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Add Data:**
   - Place your Whole Slide Image files (`.tif`, `.vsi`, etc.) into the `data/raw/wsis/` directory.
   - Export your annotations from QuPath as `.geojson` files into `data/interim/qupath_exports/`.

---

## The Pre-processing Pipeline: Oyster Segmentation

Before the main machine learning pipeline can run, we must first identify the individual oyster sections on each Whole Slide Image. Some slides contain two oyster sections that are very close or touching, which can confuse simple detection methods.

This project includes a robust pre-processing script to handle this challenge.

### Step 0: `00_segment_oysters.py`

- **Purpose:** To automatically generate a precise binary mask for each of the two main oyster sections on a raw WSI.
- **Method:** The script uses a series of standard computer vision techniques (thresholding, morphological operations) to create an initial tissue mask. To separate touching oysters, it employs the **Watershed Algorithm**. This algorithm works by aggressively eroding the mask to find definite "cores" for each oyster and then intelligently flooding outwards to find the most likely separation boundary.
- **Intelligent Merging:** After separation, the script identifies all resulting tissue fragments. It then intelligently merges significant fragments (like separated gill pieces) back to their closest parent oyster based on centroid distance, while discarding small noise artifacts.
- **Output:** The script saves one `.png` mask file for each detected oyster into the `data/interim/oyster_masks/` directory. These masks are used by subsequent scripts.

### Tuning the Segmentation

The behavior of this script is controlled by the `oyster_segmentation` section in `config.yaml`. The most important parameters to tune are:

| Parameter                      | Type    | What it Does                                                                                                | How to Tune It                                                                                                                                                                                                                           |
| :----------------------------- | :------ | :---------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `use_watershed_separation`     | boolean | Enables or disables the Watershed algorithm.                                                                | Set to `true` for slides with touching oysters.                                                                                                                                                                                          |
| `watershed_erosion_iterations` | integer | **(Most Critical)** Controls how aggressively the algorithm erodes the mask to find separate "cores".       | **Increase** this value (e.g., from 3 to 6) if the oysters are still connected. **Decrease** it if the erosion is so strong that it's destroying one of the oyster cores entirely. Check `03_watershed_markers.png` in the debug output. |
| `enable_fragment_merging`      | boolean | Enables or disables the intelligent merging of separated tissue fragments back to the main oyster bodies.   | Set to `true` to ensure you don't lose important tissue pieces (like gills) that get disconnected during processing.                                                                                                                     |
| `min_fragment_area_percent`    | float   | The size threshold for merging. Fragments smaller than this percentage of the total image area are ignored. | **Increase** this value (e.g., from `0.001` to `0.005`) if small noise artifacts are being incorrectly merged. **Decrease** it if legitimate, smaller tissue fragments are being incorrectly discarded.                                  |

---

## Usage: The Data Pipeline

The project is designed to be run as a sequence of scripts. Configure all parameters in `config.yaml` before running.

1. **Create the Dataset:** Generate image patches and labels for YOLO.

   ```bash
   python src/main_scripts/01_create_dataset.py
   ```

2. **Train the Model:** Train the YOLOv8 model on the created dataset.

   ```bash
   python src/main_scripts/02_train_yolo.py
   ```

3. **Run Inference:** Use the trained model to predict on a new WSI.

   ```bash
   python src/main_scripts/03_run_inference.py
   ```

4. **Format Predictions:** Convert the raw predictions into a QuPath-compatible GeoJSON file.

   ```bash
   python src/main_scripts/04_format_predictions.py
   ```

---

## Acknowledgements
