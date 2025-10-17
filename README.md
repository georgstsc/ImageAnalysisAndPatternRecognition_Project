# Chocolate Recognition Project — IAPR (EPFL) 🍫🔬

Overview
--------
This repository contains the final assignment for the Introduction to Image Analysis and Pattern Recognition (IAPR) course at EPFL. The task is chocolate recognition: instance segmentation + classification over 13 chocolate classes from product images (Kaggle challenge dataset). The solution explores hybrid pipelines and end‑to‑end models (Mask R‑CNN) to handle overlapping instances, variable lighting, and precise boundary detection — useful for automated quality control in confectionery manufacturing.

Key highlights
- Instance segmentation + classification: isolate individual chocolates, then label among 13 categories. 🧩
- Robust data handling: preprocessing and augmentations for lighting/angle variation. 🔄
- Evaluation & delivery: notebooks produce plots, metrics and Kaggle submission file(s). 📈

Authors & Acknowledgements 🙏
- Alessio Zazo (SCIPER: 328450) — segmentation pipeline & model experiments  
- Gautier Demierre (SCIPER: 340423) — data preprocessing & Kaggle submission  
- Georg Schwabedal (SCIPER: 328434) — classification fine‑tuning & visualizations

Thanks to the IAPR teaching team for guidance, the LTS5 lab for dataset access, and our group collaborators for many productive discussions. 💙

Quick start — run the project locally 🚀
--------------------------------------
Prerequisites
- Python 3.11+ (used for the notebook)  
- (Optional but recommended) NVIDIA GPU with CUDA for training / faster inference. If using GPU, install PyTorch with the CUDA version that matches your system (see https://pytorch.org/get-started/locally/).

1) Clone the repository
```bash
git clone https://github.com/georgstsc/ImageAnalysisAndPatternRecognition_Project.git
cd ImageAnalysisAndPatternRecognition_Project
```

2) (Recommended) Create a conda environment
```bash
conda create -n chocolate-iapr python=3.11 -y
conda activate chocolate-iapr
```

3) Install Python dependencies
- Quick install (minimal):
```bash
pip install torch torchvision opencv-python scikit-learn matplotlib pandas numpy jupyterlab
```
- If you need GPU‑enabled PyTorch, follow PyTorch's selector and install the appropriate wheel:
https://pytorch.org/get-started/locally/

- (Optional) If the project provides a requirements.txt or environment.yml, prefer:
```bash
pip install -r requirements.txt
# or
conda env create -f environment.yml
conda activate <env-name>
```

4) Get the Kaggle dataset
- Either download via the Kaggle website and place files under `data/`, or use the Kaggle API:
```bash
pip install kaggle
# place your kaggle.json at ~/.kaggle/kaggle.json
kaggle competitions download -c <competition-name> -p data/
# or kaggle datasets download -d <dataset-owner>/<dataset-name>
unzip data/<downloaded-file>.zip -d data/
```
Make sure the folder structure matches what the notebook expects (the notebook includes helper cells to locate/prepare data if needed).

5) Launch the notebook
```bash
jupyter lab
# or
jupyter notebook
```
Open `report.ipynb` and choose Kernel → Restart & Run All to execute the analysis end-to-end (recommended for a clean run).

Notes on runtime & resources ⏱️
- Notebook runtime depends on dataset size and whether training is performed. Expect ~30–60 minutes for evaluation + inference on a mid-range GPU; CPU-only runs will be significantly slower.
- Some cells may save trained checkpoints and generated outputs to `results/`. Keep an eye on disk usage.

How the notebook is organized (what runs)
----------------------------------------
report.ipynb contains:
- Data loading & preprocessing (image normalization, instance grouping)
- Augmentation pipeline used during training
- Segmentation approach(s) — Mask R‑CNN based experiments and hybrid pipelines
- Classification fine‑tuning and evaluation
- Plots, per‑class metrics and confusion matrices
- Generation of Kaggle submission CSV (final cell)

Project structure
-----------------
- report.ipynb — full project report, code and results (single notebook entrypoint)  
- data/ — (optional) place dataset files here (not checked into the repo)  
- results/ — auto‑generated plots, model checkpoints and submission files (created by the notebook)  
- (optional) requirements.txt / environment.yml — environment specs (if provided)

Generating a Kaggle submission
-----------------------------
- The notebook contains the final cell to create `submission.csv` in `results/`. After running it, upload `results/submission.csv` to Kaggle.
- If you change preprocessing or model weights, re-run the inference cells to regenerate the submission.

Customization & reproducibility 🔧
- Hyperparameters (batch size, learning rate, epochs) are grouped near the top of the notebook (Section 3). Edit there and re-run the training/eval cells.
- For reproducible runs, set the random seed variables in the notebook and run cells top-to-bottom.
- Save/export results via `jupyter nbconvert`:
```bash
jupyter nbconvert --to html report.ipynb
```

Results & takeaways
-------------------
Our end‑to‑end Mask R‑CNN approach achieved strong segmentation + classification performance (insert official test score here if available, e.g., 0.85 mAP). Key lesson: integrating segmentation and classification in a single pipeline yields better real‑world robustness than siloed stages; however multi‑chocolate merging and occlusions remain the main challenges.

If you reuse or cite this work
------------------------------
Course: IAPR — Introduction to Image Analysis & Pattern Recognition (EPFL)  
Due date: May 21, 2025

Contact / support
-----------------
For questions, reach out via EPFL Moodle (Group ID: 01, Team: "group 1"), or contact the authors listed above.

License
-------
If you need to reuse code or data, please check/add a LICENSE file to clarify reuse terms. If none exists, contact the project owners.

Change log & next steps
-----------------------
- This README was prepared to make the notebook immediately runnable locally. Replace the placeholder test score and any dataset-specific paths if you change the data layout.
- If you want, I can also: (a) create a requirements.txt from the notebook imports, (b) add a short run script to build the environment and launch the notebook, or (c) help push this README to the repository and open a PR — tell me which and I will prepare the exact commands or the commit content for you.
