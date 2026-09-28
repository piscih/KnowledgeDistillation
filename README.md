# Knowledge Distillation from BERT to BiLSTM for AG News Classification

This project investigates whether **Knowledge Distillation (KD)** can transfer useful information from a fine-tuned **BERT teacher** to a smaller **BiLSTM student** for AG News topic classification.

The experiments use the AG News dataset and compare a supervised BiLSTM baseline with BiLSTM students trained using teacher-generated soft predictions under different distillation temperatures and loss-weight settings.

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

* **Temperature ablation:** $T \in {2,4,6}$ with $\alpha=0.5$.
* **Loss-weight ablation:** $\alpha \in {0.25,0.5,0.75}$ with $T=2$.

The $\alpha=0.5$ configuration is shared between the two experiments.

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

The original test set is kept unchanged and is used only for final evaluation.

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

The training loss combines hard-label cross-entropy with KL divergence between the teacher and student predictions.

The main temperature experiment uses:

```text
T = 2, α = 0.5
T = 4, α = 0.5
T = 6, α = 0.5
```

A separate loss-weight experiment fixes the temperature at $T=2$ and evaluates:

```text
α = 0.25
α = 0.50
α = 0.75
```

Each experimental configuration is trained from a fresh BiLSTM initialization.

---

## 4. Results

Performance is measured on the untouched AG News test set.

### Temperature Ablation

| Model             | Accuracy (%) | Macro F1 (%) |
| ----------------- | -----------: | -----------: |
| BERT Teacher      |        94.41 |        94.40 |
| BiLSTM Baseline   |        90.86 |        90.81 |
| KD BiLSTM ($T=2$) |    **92.63** |    **92.62** |
| KD BiLSTM ($T=4$) |        92.09 |        92.10 |
| KD BiLSTM ($T=6$) |        91.97 |        91.95 |

Knowledge Distillation improves the BiLSTM baseline for all three tested temperatures.

The $T=2$ student achieves the highest performance among the distilled models, improving accuracy from **90.86% to 92.63%**, an absolute improvement of **1.77 percentage points**.

### Alpha Ablation

At fixed temperature $T=2$:

| $\alpha$ | Accuracy (%) | Macro F1 (%) |
| -------: | -----------: | -----------: |
|     0.25 |        92.62 |        92.60 |
|     0.50 |    **92.63** |    **92.62** |
|     0.75 |        92.09 |        92.09 |

The $\alpha=0.25$ and $\alpha=0.5$ configurations produce nearly identical performance, while increasing $\alpha$ to 0.75 results in lower performance under the evaluated settings.

### Per-Class F1

| Class    |  BERT | BiLSTM | KD ($T=2$) |
| -------- | ----: | -----: | ---------: |
| World    | 95.36 |  91.19 |      93.46 |
| Sports   | 98.54 |  96.06 |      96.91 |
| Business | 91.66 |  87.88 |      89.52 |
| Sci/Tech | 92.06 |  88.13 |      90.59 |

The largest F1 improvement from Knowledge Distillation occurs for **Sci/Tech**, increasing from 88.13% to 90.59%, an improvement of **2.46 percentage points** over the BiLSTM baseline.

The best distilled model recovers approximately **49.9% of the accuracy gap between the BiLSTM baseline and the BERT teacher**.

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
│   ├── bilstm_kd2_confusion_matrix.png
│   ├── bilstm_kd2_test_results.json
│   ├── bilstm_kd4_confusion_matrix.png
│   ├── bilstm_kd4_test_results.json
│   ├── bilstm_kd6_confusion_matrix.png
│   ├── bilstm_kd6_test_results.json
│   ├── bilstm_kd2(0.25)_confusion_matrix.png
│   ├── bilstm_kd2(0.25)_test_results.json
│   ├── bilstm_kd2(0.75)_confusion_matrix.png
│   ├── bilstm_kd2(0.75)_test_results.json
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
│       └── train.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

Large model checkpoints, optimizer states, virtual environments, and Python cache files are excluded from version control.

---

## 6. How to Run

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

The KD training configuration supports the temperature and loss-weight ablations described above. The temperature experiment evaluates $T=2,4,6$ with $\alpha=0.5$, while the alpha experiment evaluates $\alpha=0.25,0.5,0.75$ at $T=2$.

### Evaluate the models

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
