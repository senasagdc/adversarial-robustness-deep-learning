# Reproducing Madry et al. (ICLR 2018) on MNIST

CMP4509 Deep Learning, Fall 2026 term project.

We reproduce the central MNIST result of *Towards Deep Learning Models Resistant to Adversarial Attacks* (Madry, Makelov, Schmidt, Tsipras, Vladu, ICLR 2018): PGD adversarial training gives high accuracy under ℓ∞ attacks with ε = 0.3, while a normally trained network drops to 0%.

## Main result (3 seeds, mean ± std)

| Attack | Natural training | PGD training (ours) | Paper (Table 1) |
|---|---|---|---|
| none (clean) | 99.09 ± 0.09 | 98.43 ± 0.09 | 98.8 |
| FGSM | 9.04 ± 1.91 | 93.98 ± 0.24 | 95.6 |
| PGD-40 | 0.00 | 91.84 ± 0.34 | 93.2 |
| PGD-100 | 0.00 | 89.60 ± 0.42 | 91.8 |
| PGD-40, 20 restarts* | 0.00 | 89.47 ± 0.23 | 90.4 |
| PGD-100, 20 restarts* | 0.00 | 87.83 ± 0.21 | 89.3 |

\* first 1000 test images. The claim holds; the 1–2 point gap is consistent with our ~4× shorter training (20 epochs instead of ~83).

## Repository contents

| Path | What it is |
|---|---|
| `madry.py` | All code: data loading, network (`MadryCNN`), FGSM, PGD with projection, adversarial training with per-epoch checkpoints, evaluation, transfer attack, loss-landscape study |
| `notebooks/madry_reproduction.ipynb` | Runs every experiment and produces every number and figure in the report |
| `figures/` | Figures used in the report |
| `results/` | JSON result files written by the notebook |

## Experiments

- **E1 – Central claim:** natural vs PGD adversarial training, full network (32-64-1024), 20 epochs, 3 seeds; FGSM, PGD-40, PGD-100 and 20-restart evaluation.
- **E2 – Capacity:** {natural, FGSM, PGD} training × network scale {1, 2, 4, 8, 16} × 3 seeds (45 models, 5 epochs each).
- **E3 – Loss concentration:** 1000 random-start PGD-40 runs on 5 test images.
- **E4 – Gradient-masking checks:** accuracy vs ε, accuracy vs number of PGD steps, black-box transfer attack.

Settings (same as the paper unless noted): ε = 0.3, pixels in [0, 1]; Adam, lr 1e-4, batch 50; training attack PGD-40, step 0.01, random start; final model is used (no checkpoint selection).

## How to run

1. Open the notebook in Google Colab:
   https://colab.research.google.com/github/senasagdc/adversarial-attacks-dl-test/blob/main/notebooks/madry_reproduction.ipynb
2. Select a GPU runtime (Runtime → Change runtime type → T4 GPU).
3. Run all cells (Runtime → Run all). The notebook mounts Google Drive and writes checkpoints, results and figures to `MyDrive/madry_repro/full/`.
4. If the session disconnects or the GPU quota runs out, simply run all cells again: finished runs are skipped and unfinished runs resume from their last epoch checkpoint.

Runtime on a free Colab T4: one PGD-training epoch ≈ 76 s, one natural-training epoch ≈ 4 s; one full E1 PGD model ≈ 25 min. The full notebook needs a few hours of GPU time in total and can be spread over several sessions.

Requirements: Python 3 with PyTorch, torchvision, numpy, matplotlib (all preinstalled on Colab).

## Report

The report (LaTeX, course template) describes the method, every change from the paper, all results and their discussion.
