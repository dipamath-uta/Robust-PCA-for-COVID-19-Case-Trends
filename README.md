# Robust PCA for COVID-19 Case Trends  
### Comparing Convex (PCP) and Non-Convex (IRCUR) Methods Under Different Temporal Aggregation Scales

This repository contains all Python code used to preprocess COVID-19 case data,
run convex and non-convex Robust Principal Component Analysis (RPCA)
decompositions, and generate all figures presented in the project report.


## 📁 Repository Structure
covid-vax-project/
│
├── data/ # Cleaned weekly & daily matrices
├── preprocessing/ # Scripts for OWID loading & weekly aggregation
├── robustpca/ # PCP and IRCUR algorithm implementations copied from https://github.com/sverdoot
│ ├── pcp.py
│ ├── ircur.py
│ ├── utils.py
│
├── plots/ # Generated figures (low-rank and sparse components)
│ ├── continents_daily_lowrank_pcp_ircurr.png
│ ├── continents_sparse_weeklymean_pcp_ircurr.png
│ └── ...
│
├── run_pcp_covid.py # Driver script for PCP on continents & WHO
├── run_ircur_covid.py # Driver script for IRCUR (non-convex) RPCA
│
└── README.md # This document


## ⚙️ Requirements

All code is tested using:

- Python 3.10–3.12  
- NumPy  
- SciPy  
- Pandas  
- Matplotlib  

## Install dependencies:

```bash
pip install -r requirements.txt

🚀 Running the Experiments
1. Preprocess the data
python preprocessing/build_weekly_matrices.py

2. Run convex RPCA (PCP)
python run_pcp_covid.py


This generates:

*_lowrank_pcp.csv

*_sparse_pcp.csv

corresponding plots under plots/.

3. Run non-convex RPCA (IRCUR)
python run_ircur_covid.py


This produces:

*_lowrank_ircurr.csv

*_sparse_ircurr.csv

IRCUR-based figures.

📊 Figures

The repository includes:

Daily vs Weekly low-rank trends

Daily vs Weekly sparse components

PCP vs IRCUR comparisons

Vaccination vs case-trend visualizations (context only)

All figures used in the report are reproducible from the included scripts.

🔍 Notes on the Non-Convex Method (IRCUR)

This project uses an IRCUR implementation adapted from a public robust PCA
repository. IRCUR behaves as a non-convex RPCA algorithm that approximates
low-rank structure using CUR-based sampling and iterative thresholding.

Although theoretically capable of sharper rank selection, IRCUR showed
instability on noisy epidemiological data, as documented in the report.

📑 Citation (not available right now)

## Contact:
Feel free to open issues or contact me if you want to extend or adapt the code.
Sadia Afrin Dipa
Email: dipacoumath@gmail.com









