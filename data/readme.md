# AG News Dataset

This directory contains the data used for the knowledge distillation experiments.

## Dataset

**AG News** is a text classification dataset consisting of news articles grouped into four categories:

| Label | Category |
| ----: | -------- |
|     0 | World    |
|     1 | Sports   |
|     2 | Business |
|     3 | Sci/Tech |

The standard dataset contains:

* **120,000** training examples
* **7,600** test examples
* **4** classification categories

### Source

The dataset is available through Hugging Face:

**AG News:** https://huggingface.co/datasets/fancyzhx/ag_news

## Download / Loading

The dataset does not need to be committed to this repository. It can be downloaded automatically using the Hugging Face `datasets` library.

```python
from datasets import load_dataset

dataset = load_dataset("fancyzhx/ag_news")

print(dataset)
```

This will download the dataset locally and provide the following splits:

```text
DatasetDict({
    train: Dataset(...)
    test: Dataset(...)
})
```

## Repository Policy

The raw dataset is **not stored in this GitHub repository** because of its size.

The dataset should be downloaded locally when running the experiments. This directory contains documentation only.

