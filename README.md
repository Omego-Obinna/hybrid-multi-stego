# Hybrid Multichannel Steganography

Research code, experiment configurations, and compact reproducibility material for the hybrid multichannel steganography framework developed by **Obinna Omego** and **Michał Bosy**.

The repository supports the protocol implementation and the main experimental workflow used to evaluate the proposed hybrid steganographic model, including adaptive/QIM-based embedding, image-quality analysis, classical steganalysis, CNN-based steganalysis, robustness testing, and linkability probes.

> **Repository:** `Omego-Obinna/hybrid-multi-stego`  
> **Licence:** Apache-2.0  
> **Primary dataset:** BOSSbase 1.01  
> **Main experimental seed:** `20260605`

---

## 1. Scope

The repository is intended to provide a **compact reproducibility package** rather than an archive of every intermediate experiment.

It contains, or is designed to contain:

- the core hybrid steganographic protocol;
- the proposed embedding and extraction implementation;
- experiment launchers and configuration files;
- deterministic dataset splits;
- compact CSV summaries supporting the reported tables and figures;
- scripts for regenerating the main plots and tables.

Large datasets, feature caches, trained checkpoints, and full generated stego-image collections are intentionally excluded.

---

## 2. Research Overview

The framework combines **cover synthesis** and **cover modification** within a multichannel protocol.

At protocol level, the method uses independently transmitted cover information together with a stego-object and cryptographic masking/integrity mechanisms. At image level, the current experimental implementation evaluates a region-constrained, direction-adaptive QIM-based embedding strategy with fused distortion information.

The current experimental method is referred to as:

```text
QIM-Fused-CF
```

The evaluation considers both embedding fidelity and resistance to steganalysis.

### Main evaluation components

- embedding fidelity and extraction correctness;
- payloads including `0.10`, `0.20`, and `0.40` bpp where applicable;
- maxSRM + Ensemble Classifier;
- Ye-Net;
- Yedroudj-Net;
- robustness/resynchronisation analysis;
- linkability analysis;
- generation of publication tables and figures.

---

## 3. Repository Layout

The layout should be structured as it is uploaded. It can be structured alternatively. However, an alternative recommended layout should be:

```text
hybrid-multi-stego/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── .gitignore
│
│
├── src/
│   ├── hybrid_stego.py
│   ├── qim_embedding.py
│   ├── cover_generation.py
│   └── utils.py
│
├── experiments/
│   ├── run_embedding.py
│   ├── run_srm_ec.py
│   ├── run_yenet.py
│   ├── run_yedroudj.py
│   └── run_linkability.py
│
│
├── results/
│   ├──
```

The exact filenames may differ slightly from the local research environment. The important requirement is that the repository preserves the **code, configurations, dataset split, compact result summaries, and figure-generation logic** needed to reproduce the reported experiments.

---

## 4. Installation

### 4.1 Clone the repository

```bash
git clone https://github.com/Omego-Obinna/hybrid-multi-stego.git
cd hybrid-multi-stego
```

### 4.2 Create a virtual environment

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Upgrade `pip`:

```bash
python -m pip install --upgrade pip
```

### 4.3 Install dependencies

```bash
pip install -r requirements.txt
```

### PyTorch and GPU support

The generic `requirements.txt` includes the Python packages needed by the experimental workflow. For CUDA-enabled Ye-Net or Yedroudj-Net training, install the PyTorch build appropriate to the local CUDA version using the official PyTorch installation selector before running the CNN experiments.

---

## 5. Dataset

### BOSSbase 1.01

The image experiments use **BOSSbase 1.01**, which contains 10,000 grayscale images and is widely used for image-steganography and steganalysis research.

The dataset is **not redistributed in this repository**.

Obtain BOSSbase from the Digital Data Embedding Laboratory at Binghamton University and place it locally, for example:

```text
data/
└── BOSSbase_1.01/
    ├── 1.pgm
    ├── 2.pgm
    ├── ...
    └── 10000.pgm
```

The `data/` directory should remain excluded from Git.

### Reproducible split

Experiments use the fixed seed:

```text
20260605
```

Where provided, `splits/bossbase_seed20260605.csv` records the exact train/validation/test allocation used in the experiments. Use this split rather than generating a new random partition when reproducing reported values.

---

## 6. Proposed Embedding Experiments

The main embedding configuration is based on:

```text
Ours-QIM-v3.2-Fused-CF
```

The implementation combines distortion information derived from S-UNIWARD and MiPOD with the QIM-based embedding procedure and region/direction constraints used in the study.

A typical experiment follows the form:

```bash
python experiments/run_embedding.py \
    --payload 0.20 \
    --seed 20260605
```

Repeat for the payloads required by the manuscript, for example:

```bash
0.10
0.20
0.40
```

Generated stego images should normally be written to a local `outputs/` directory and should **not** be committed in bulk.

---

## 7. Steganalysis

### 7.1 SRM + Ensemble Classifier

Run the classical steganalysis experiment using the required payload:

```bash
python experiments/run_srm_ec.py \
    --payload 0.20 \
    --seed 20260605
```

NOTE: The SRM feature cache can be expensive to generate.

Recommended metrics include:

- Accuracy;
- Balanced Accuracy;
- AUC;
- MCC;
- test-set error/detection performance.

---

### 7.2 Ye-Net

The Ye-Net experiments use a fixed configuration, and the representative settings used in the study include:

```text
epochs: 80
batch_size: 8
optimizer: SGD
momentum: 0.9
weight_decay: 1e-4
learning_rate: 0.003
milestones: [40, 60]
gamma: 0.8
seed: 20260605
```

Run with:

```bash
python experiments/run_yenet.py \
    --config configs/yenet.yaml \
    --payload 0.20
```

---

### 7.3 Yedroudj-Net

The Yedroudj-Net representative settings include:

```text
learning_rate: 0.001
gamma: 0.3
batch_size: 16
image_size: 512
augmentation: true
seed: 20260605
```

---

## 8. Linkability Probes

The linkability experiments evaluate whether an informed warden can distinguish linked observations from shuffled or independently paired observations.

Run the compact experiment wrapper using the seed:

```bash
    --seed 20260605
```

The repository should contain only the scripts, configuration, and final compact result tables. Large generated pair datasets should remain local.

---

## 9. Results Included in the Repository

Only compact numerical outputs needed to support the manuscript should be committed.

Recommended files include for example:

```text
results/
├── embedding_fidelity.csv
├── detectors_payloads_test.csv
├── qim_resync_robustness.csv
├── linkability_binary_results.csv
└── linkability_multiclass_results.csv
```

Per-epoch CNN logs may also be included when they are directly used to generate supplementary learning curves.

The repository should avoid committing:

- thousands of generated cover/stego images;
- SRM feature caches;
- `.npy` feature matrices;
- complete checkpoint histories;
- temporary files;
- virtual environments;
- duplicate figure formats unless specifically needed for publication.

---

## 10. Reproducibility Notes

For comparable results:

1. use the same BOSSbase source images;
2. use the provided split;
3. keep `seed = 20260605`;
4. use the configuration files supplied with each detector;
5. do not mix results obtained from different preprocessing pipelines;
6. report the exact payload and test split;
7. preserve cover/stego pairing during detector evaluation;
8. record package and GPU/CUDA versions for CNN experiments.

For archival reproducibility, an exact environment snapshot can additionally be generated after verification:

```bash
pip freeze > requirements-lock.txt
```

This lock file is optional and should complement, not replace, the cleaner human-maintained `requirements.txt`.

---

## 11. Citation

Please, the citation will be provided when final journal bibliographic information when the peer-reviewed version becomes available.

---

## 12. Licence

This repository is distributed under the **Apache License 2.0**. See `LICENSE` for details.

Third-party datasets, external steganalysis implementations, and derived detector architectures remain subject to their respective licences and terms of use.

---

## 13. Authors

**Obinna Omego**  
School of Computer Science and Mathematics  
Kingston University London  

---

## 14. Contact and Issues

For reproducibility questions, implementation problems, or corrections, please use the GitHub **Issues** section of this repository.

Contributions that improve reproducibility, documentation, or experiment portability are welcome.
