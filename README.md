# CRISP: A Causal Relationships-Guided Deep Learning Framework for Advanced ICU Mortality Prediction

This repository is a PyTorch implementation of our paper "CRISP: A Causal Relationships-Guided Deep Learning Framework for Advanced ICU Mortality Prediction".

## Overview

CRISP (Causal Relationships-guided framework for ICU mortality prediction using deep learning) is an advanced deep learning framework designed for predicting mortality in Intensive Care Unit (ICU) patients. The framework incorporates causal inference techniques to improve prediction accuracy and robustness by modeling the complex relationships between various clinical features.

## Key Features

- **Causal Inference Integration**: Utilizes causal relationships between diagnoses to enhance model performance
- **Transformer-based Architecture**: Employs a Tabular Transformer model to capture complex feature interactions
- **Multi-modal Data Processing**: Handles diverse clinical data types including diagnoses, procedures, medications, and time-series indicators
- **Data Balancing Techniques**: Implements various sampling methods (CMG, SMOTE, ADASYN, etc.) to address class imbalance
- **MIMIC Dataset Support**: Compatible with both MIMIC-III and MIMIC-IV datasets

## Repository Structure

```
├── Code/
│   ├── Main.py              # Main training script
│   ├── ourModels.py         # Model definitions (Tabular Transformer)
│   ├── CMG.py               # Causal Matching Generator for data balancing
│   ├── Data_process/        # Jupyter notebooks for data preprocessing
│   │   ├── 0_Get_Patients.ipynb
│   │   ├── 1_Get_D_P_M.ipynb
│   │   └── 2_get_TS_data.ipynb
├── input/                   # Input data files
│   ├── 0_diag_9_10.csv      # Diagnosis codes
│   ├── drug-atc.csv         # Drug-ATC code mappings
│   ├── RXCUI2atc4.csv       # RXCUI to ATC4 code mappings
│   ├── ATC_SMIL.pkl         # ATC code to SMILES representations
│   └── rxnorm2RXCUI.txt     # RxNorm to RXCUI mappings
├── LICENSE                  # MIT License
└── README.md                # This file
```

## Requirements

- Python 3.7+
- PyTorch
- Transformers (Hugging Face)
- Scikit-learn
- Pandas
- NumPy
- Psmpy (Propensity Score Matching)
- Imbalanced-learn

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/CRISP.git
cd CRISP
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Data Preparation

1. Obtain access to the MIMIC-III and/or MIMIC-IV datasets from [https://mimic.mit.edu/](https://mimic.mit.edu/)
2. Place the required drug data files from [https://github.com/ycq091044/SafeDrug](https://github.com/ycq091044/SafeDrug) in the `input/` directory
3. Run the data processing notebooks in sequential order:
   - `Code/Data_process/0_Get_Patients.ipynb`
   - `Code/Data_process/1_Get_D_P_M.ipynb`
   - `Code/Data_process/2_get_TS_data.ipynb`

## Usage

Train the CRISP model using the following command:

```bash
python Code/Main.py
```

### Command Line Arguments

- `--CI`: Enable causal inference loss (default: True)
- `--Dataset`: Dataset to use ('III' or 'III-IV') (default: 'III-IV')
- `--Label`: Target label ('DIEINHOSPITAL') (default: 'DIEINHOSPITAL')
- `--Balance`: Data balancing method ('CMG', 'SMOTE', 'ADASYN', 'SMOTEENN', 'SMOTETomek', 'None') (default: 'None')
- `--dim`: Model dimension (default: 64)
- `--epoch`: Number of training epochs (default: 200)
- `--batch_size`: Training batch size (default: 16)
- `--lr`: Learning rate (default: 0.0001)
- `--w_ci`: Weight of causal inference loss (default: 0.2)
- `--k_ci`: Number of diagnoses for causal inference loss (default: 10)

## Model Architecture

The CRISP framework utilizes a Tabular Transformer architecture with the following components:

1. **Feature Embedding**: Separate embedding layers for diagnoses, procedures, medications, and other clinical indicators
2. **Transformer Layers**: Multi-head self-attention mechanism to capture feature interactions
3. **Causal Inference Module**: Introduces counterfactual samples to enhance model robustness
4. **Output Layer**: Fully connected layers with sigmoid activation for binary classification

## Data Balancing Methods

The framework supports multiple data balancing techniques:

- **CMG (Causal Matching Generator)**: Our proposed method that combines propensity score matching with synthetic data generation
- **SMOTE**: Synthetic Minority Oversampling Technique
- **ADASYN**: Adaptive Synthetic Sampling
- **SMOTEENN**: Combination of SMOTE and Edited Nearest Neighbors
- **SMOTETomek**: Combination of SMOTE and Tomek Links

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite our paper:

```
@article{your-paper-reference,
  title={CRISP: A Causal Relationships-Guided Deep Learning Framework for Advanced ICU Mortality Prediction},
  author={Your Names},
  journal={Your Journal},
  year={2024}
}
```