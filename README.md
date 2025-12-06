# CV Project: Skin Disease Classification Through Residual Networks And Vision Transformers

This repository contains the code for training and evaluating deep learning models, mainly **ResNet** and **ViT (Vision Transformer)** architectures.  
All experiments — **training and evaluations** — are orchestrated from the Jupyter notebook `main.ipynb`.

> ⚠️ **Note on data and models**  
> Datasets and trained model checkpoints are **not** tracked in this repository.  
> They are excluded via `.gitignore` because they are too large.  
> You’ll need to download / generate them separately and place them in the correct folders before running the notebook or scripts.

---

## Repository Structure

```text
.
├── main.ipynb          # Central notebook: data loading, training & evaluation pipeline
├── resnet_v2.py        # ResNet model definition(s)
├── VIS_V1.py           # Vision Transformer (ViT) model definition(s)
├── data_process_v4.py  # Data processing & dataset utilities
├── eval_resnet.py      # Standalone evaluation script for ResNet models
├── eval_vis.py         # Standalone evaluation script for ViT models
├── utils.py            # Shared helper function for mld to train on apple silicon M4
├── legacy/             # Legacy code and final models from previous iterations
├── __pycache__/        # Python bytecode cache (auto-generated)
├── .env                # Optional environment variables (not tracked)
├── .gitignore          # Excludes data, models, and other heavy / local files
```
# License

Dataset & Licensing (ISIC 2018 Challenge)
This project uses data from the ISIC 2018: Skin Lesion Analysis Towards Melanoma Detection Challenge.
The ISIC 2018 training data are distributed under the Creative Commons Attribution–NonCommercial 4.0 International (CC BY-NC 4.0) license.
In practical terms:
You must give appropriate credit to the dataset creators.
You may not use the dataset for commercial purposes.
You must include a link to the license and indicate if changes were made.
For full legal details, see the official license page: https://creativecommons.org/licenses/by-nc/4.0/
When publishing work based on this repository and the ISIC 2018 data, please cite at least the following:
N. Codella et al., “Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the International Skin Imaging Collaboration (ISIC)”, 2018.
P. Tschandl, C. Rosendahl, H. Kittler, “The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions”, Scientific Data 5, 180161 (2018).
If you redistribute or build on this work, ensure your use of the ISIC 2018 data complies with the CC BY-NC 4.0 terms and any additional conditions stated on the official ISIC challenge data page.
