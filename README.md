# Knowledge Distillation from BERT to BiLSTM for AG News Classification

This project investigates whether **Knowledge Distillation (KD)** can transfer useful information from a fine-tuned **BERT teacher** to a smaller **BiLSTM student** for AG News topic classification.

The experiments use the AG News dataset and compare a supervised BiLSTM baseline with BiLSTM students trained using teacher-generated soft predictions under different distillation temperatures and loss-weight settings. The distillation configuration is selected on a held-out validation split, and the test set is used only to report final results.

---

## 1. Overview

The project follows a teacher-student setup:

```text
AG News
   │
   ├── BERT Teacher
   │      └── Fine-tuned on AG News
   │             └── Cached teacher logits
   │
   ├── BiLSTM Baseline
   │      └── Hard-label supervision
   │
   └── BiLSTM + Knowledge Distillation
          ├── T = 2, α = 0.5
          ├── T = 4, α = 0.5
          ├── T = 6, α = 0.5
          ├── T = 2, α = 0.25
          └── T = 2, α = 0.75
```

The BiLSTM baseline and distilled students use the **same architecture, preprocessing, vocabulary, and training configuration**. The main difference is the training objective: the baseline uses ground-truth labels, while the distilled students additionally learn from the BERT teacher's soft predictions.

The BERT teacher is trained first and then frozen. Its logits are precomputed and stored so that the teacher does not need to be evaluated repeatedly during student training.

Two ablation experiments are conducted:

* **Temperature ablation:** $T \in \{2, 4, 6\}$ with $\alpha = 0.5$.
* **Loss-weight ablation:** $\alpha \in \{0.25, 0.5, 0.75\}$ with $T = 2$.

The $\alpha = 0.5$, $T = 2$ configuration is shared between the two experiments.

---

## 2. Dataset

The project uses the [AG News dataset](https://huggingface.co/datasets/fancyzhx/ag_news), a four-class news topic classification dataset.

| Property               |   Value |
| ---------------------- | ------: |
| Classes                |       4 |
| Training examples      | 120,000 |
| Test examples          |   7,600 |
| Examples per class     |  30,000 |
| Training split         | 108,000 |
| Validation split       |  12,000 |
| Train/validation split |   90/10 |
| Random seed            |      42 |

The four classes are:

* World
* Sports
* Business
* Sci/Tech

The original test set is held out from training. Within each run, the best epoch is chosen by validation accuracy. The distillation settings ($T$, $\alpha$) are selected on the validation split as well (see Section 4); the test set is used only to report final results.

---

## 3. Models

### BERT Teacher

The teacher is based on `bert-base-uncased` and is fine-tuned for four-class AG News classification.

Main configuration:

* Maximum sequence length: 128
* Batch size: 16
* Learning rate: `2e-5`
* Weight decay: `0.01`
* Training epochs: 2

After training, the selected BERT model is frozen and used to generate cached logits for Knowledge Distillation.

### BiLSTM Baseline

The baseline is a two-layer bidirectional LSTM with:

* 300-dimensional word embeddings
* Hidden size: 256 per direction
* Dropout: 0.3
* Packed sequence processing
* Linear classification layer
* Batch size: 64
* Learning rate: `1e-3`
* Weight decay: `1e-5`
* Adam optimizer
* 10 epochs

### Knowledge Distillation

The distilled BiLSTM uses the **same architecture and training configuration** as the baseline.

The training loss combines hard-label cross-entropy with the KL divergence between the teacher and student predictions, both softened with temperature $T$:

$$
\mathcal{L} = \alpha \cdot \mathrm{CE}(z_s, y) + (1 - \alpha) \cdot T^2 \cdot \mathrm{KL}\left(\mathrm{softmax}(z_t / T) \,\|\, \mathrm{softmax}(z_s / T)\right)
$$

where $z_s$ and $z_t$ are the student and teacher logits and $y$ is the ground-truth label.

The main temperature experiment uses:

```text
T = 2, α = 0.5
T = 4, α = 0.5
T = 6, α = 0.5
```

A separate loss-weight experiment fixes the temperature at $T = 2$ and evaluates:

```text
α = 0.25
α = 0.50
α = 0.75
```

Each experimental configuration is trained from a fresh BiLSTM initialization.

---

## 4. Results

### Model selection (validation split)

The distillation configuration was selected using accuracy on the 12,000-example validation split, without looking at the test set.

| Configuration   | Validation accuracy (%) | Validation macro F1 (%) |
| --------------- | ----------------------: | ----------------------: |
| T=2, α=0.25     |                **92.73** |                **92.68** |
| T=4, α=0.5      |                   92.60 |                   92.56 |
| T=2, α=0.5      |                   92.47 |                   92.43 |
| T=6, α=0.5      |                   92.43 |                   92.38 |
| T=2, α=0.75     |                   92.43 |                   92.40 |

The selected configuration is **T=2, α=0.25**. The validation scores of all five settings lie within about 0.3 percentage points of each other, which is close to the standard error of a 12,000-example validation set (about 0.25 pp). The ranking among distillation settings should therefore be read as indicative rather than conclusive.

### Test results

Performance is measured on the AG News test set (7,600 examples).

#### Temperature ablation ($\alpha = 0.5$)

| Model             | Accuracy (%) | Macro F1 (%) |
| ----------------- | -----------: | -----------: |
| BERT Teacher      |        94.41 |        94.40 |
| BiLSTM Baseline   |        90.86 |        90.81 |
| KD BiLSTM         |        92.62|         92.60 |

Knowledge Distillation improves the BiLSTM baseline.

#### Headline result

The validation-selected student ($T=2$, $\alpha=0.25$) improves accuracy from **90.86% to 92.62%**, an absolute gain of **1.76 percentage points**, and macro F1 from **90.81% to 92.60%**. It recovers approximately **49.6% of the accuracy gap between the BiLSTM baseline and the BERT teacher**. All five distilled students outperform the baseline (91.97% to 92.63% accuracy versus 90.86%), so the improvement does not depend on which configuration is selected.

#### Per-class F1

| Class    |  BERT | BiLSTM | KD ($T=2$, $\alpha=0.5$) |
| -------- | ----: | -----: | -----------------------: |
| World    | 95.36 |  91.19 |                    93.46 |
| Sports   | 98.54 |  96.06 |                    96.91 |
| Business | 91.66 |  87.88 |                    89.52 |
| Sci/Tech | 92.06 |  88.13 |                    90.59 |

The per-class table shows the $T=2$, $\alpha=0.5$ student. Per-class results for every configuration, including the selected one, are stored in `results/`. For this student, the largest F1 improvement occurs for **Sci/Tech**, from 88.13% to 90.59% (+2.46 percentage points).

### Limitations

* **Single seed.** Every configuration was trained once. Differences between distillation settings (a few tenths of a percentage point) are within the likely run-to-run variation, so only the gap between the baseline and the distilled students should be treated as a robust finding.
* **No efficiency measurements.** The repository does not currently report parameter counts or inference speed for BERT versus the BiLSTM.

---

## 5. Repository Structure

```text
KnowledgeDistillation/
│
├── models/
│   └── BiLSTM_KD/
│
├── notebooks/
│   ├── 1.Data_exploration.ipynb
│   ├── 2.Bert_teacher.ipynb
│   └── 3.Results_comparision.ipynb
│
├── results/
│   ├── bert_confusion_matrix.png
│   ├── bert_test_results.json
│   ├── bilstm_confusion_matrix.png
│   ├── bilstm_baseline_results.json
│   ├── bilstm_kd2(0.25)_confusion_matrix.png
│   ├── bilstm_kd2(0.25)_test_results.json
│   ├── kd_validation_selection.json
│   └── plots/
│       ├── model_comparison.png
│       ├── per_class_f1_comparison.png
│       └── temperature_comparison.png
│
├── training/
│   ├── Bert/
│   │   ├── bert_teacher.py
│   │   ├── train_bert.py
│   │   └── test_bert.py
│   │
│   ├── BiLSTM/
│   │   ├── dataset.py
│   │   ├── evaluation_BiLSTM.py
│   │   ├── model.py
│   │   ├── train.py
│   │   └── vocabulary.py
│   │
│   └── KD/
│       ├── dataset.py
│       ├── evaluation_KD.py
│       ├── precompute_teacher.py
│       ├── select_on_validation.py
│       └── train.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

Large model checkpoints, optimizer states, virtual environments, and Python cache files are excluded from version control.

---

## 6. How to Run

Run all commands from the repository root.

### Clone the repository

```bash
git clone <repository-url>
cd KnowledgeDistillation
```

### Create the environment

```bash
python -m venv kd-env
source kd-env/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train the BERT teacher

```bash
python training/Bert/train_bert.py
```

### Generate teacher logits

```bash
python training/KD/precompute_teacher.py
```

### Train the BiLSTM baseline

```bash
python training/BiLSTM/train.py
```

### Train the KD students

```bash
python training/KD/train.py
```

The KD training configuration supports the temperature and loss-weight ablations described above. The temperature experiment evaluates $T = 2, 4, 6$ with $\alpha = 0.5$, while the loss-weight experiment evaluates $\alpha = 0.25, 0.5, 0.75$ at $T = 2$.

### Select the KD configuration on the validation split

```bash
python -m training.KD.select_on_validation
```

This evaluates every trained KD checkpoint on the validation split only and writes the ranking to `results/kd_validation_selection.json`. It never loads the test set.

### Evaluate the models on the test set

```bash
python training/Bert/test_bert.py
```

```bash
python training/BiLSTM/evaluation_BiLSTM.py
```

```bash
python training/KD/evaluation_KD.py
```

The evaluation scripts produce **JSON test-result files and confusion matrices**.

---

## 7. References

* Hinton, G., Vinyals, O., & Dean, J. (2015). *Distilling the Knowledge in a Neural Network.*
* Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.*
* Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter.*
* Jiao, X., et al. (2020). *TinyBERT: Distilling BERT for Natural Language Understanding.*
* Zhang, X., Zhao, J., & LeCun, Y. (2015). *Character-level Convolutional Networks for Text Classification.*
