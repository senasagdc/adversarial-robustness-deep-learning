# Reproducing Madry et al. (ICLR 2018) on MNIST

CMP4509 term project. We reproduce the central claim of
*Towards Deep Learning Models Resistant to Adversarial Attacks* (Madry, Makelov, Schmidt, Tsipras, Vladu, ICLR 2018):
training on PGD adversarial examples, i.e. solving the saddle-point problem

$$\min_\theta \; \mathbb{E}_{(x,y)}\Big[\max_{\|\delta\|_\infty \le \epsilon} L(\theta, x+\delta, y)\Big],$$

gives MNIST models that remain accurate under strong first-order L∞ attacks with ε = 0.3.

## Repository

| File | Content |
|---|---|
| `madry.py` | all code: data, model, FGSM/PGD attacks, training, evaluation (~250 lines) |
| `notebooks/madry_reproduction.ipynb` | runs every experiment and produces every table and figure of the report |

## How to run

1. Open the notebook in Google Colab (File → Open notebook → GitHub → this repo) or Kaggle (internet on).
2. Select a GPU runtime (T4).
3. Set `REPO_URL` in the first code cell to this repository.
4. Run all cells with `QUICK = True` (~15 min) to test the pipeline, then with `QUICK = False`.

Checkpoints are written after every epoch, and results are saved as JSON files (on Google Drive in Colab).
Re-running the notebook skips finished work and resumes unfinished training runs, so the experiments
can be split over several free sessions.

## Experiments

| | Setting | Paper reference |
|---|---|---|
| E1 | natural vs. PGD-40 adversarial training, paper architecture, seeds 0–2; clean / FGSM / PGD-40 / PGD-100 / 20-restart / transfer accuracy | Table 1 |
| E2 | natural, FGSM and PGD training at capacity scales 1, 2, 4, 8, 16 | Sec. 4, Fig. 4 |
| E3 | distribution of the loss reached by 1000 random-restart PGD runs | Sec. 3, Fig. 1 |
| E4 | accuracy vs. ε and vs. number of PGD steps (gradient-masking checks) | — |

## Deviations from the paper

* Training budget: 20 epochs (≈ 24k iterations of batch 50) instead of 100k iterations; E2 uses 5 epochs.
* 20-restart evaluations use the first 1000 test images; single-start attacks use the full test set.
* Everything else (architecture, Adam lr 1e-4, batch 50, ε = 0.3, PGD-40 with step 0.01 and random start) follows the paper.
