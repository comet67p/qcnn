# Code for A Novel Quantum-based model of the Cortical Canonical Microcircuit
This repository contains code for the paper "A Novel Quantum-based model of the Cortical Canonical Microcircuit" presented at TEXAS JUNIOR SCIENCE & HUMANITIES SYMPOSIUM 2024

## Pre-requisites

- Python 3.11.7 (code was testing using this version, other versions may also work)
- VSCode Editor

## Setup Instructions

1. Create a virtual environment, and install the packages as shown below

```
python -m venv .canonical

.canonical/bin/activate

pip install torch qiskit qiskit_algorithms qiskit_machine_learning pylatexenc torchvision ipywidgets matplotlib qiskit_aer pandas tensorboard torchmetrics scikit-image

```

2. Download and extract the data

```

download_data.sh

```

3. Run the notebook "sampler-qnn-gpu.ipynb"

