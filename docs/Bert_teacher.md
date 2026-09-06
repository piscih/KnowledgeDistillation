### Bert_teacher


#### Objective 
Fine tune a pretrained BERT model on the AG news dataset text classification to obtain a high performing teacher model.
The result of the teacher provides the soft logits to a smaller BiLSTM student.

#### Teacher Model
- Model : google-bert/bert-base-uncased
- Architecture : Bert-base
- Task ; sequence classification
- No of classes : 4
- No of parameters: ~110 million

#### Why BERT 
BERT was chosen because it is a pretrained Transformer-based language model capable of producing contextual representations of text. Fine-tuning allows the pretrained model to adapt these representations to the AG News classification task. Its has relatively high capacity which makes it suitable as a teacher for transferring knowledge to a smaller BiLSTM student.

#### Dataset 
- Dataset : AG news dataset
- Hugging Face identifier: fancyzhx/ag_news
- Task: 4-class news classification
- Classes : World,sports,Business,Sci/Tech
- Split : Original train has 120,000 so 90% training ( 108,000) and 10% validation(12,000), test (7600)

#### Tokenization 
- Tokenizer: BERT tokenizer corresponding to google-bert/bert-base-uncased
- Maximum sequence length: 128 tokens
- Truncation: enabled
- Padding: enabled

#### Loss function 
- The model uses cross-entropy loss

#### Training hyperparameters
- No of labels : 4
- Max seq length : 128
- Batch size : 16
- learning rate : 2 * 10e -5 
- weight decay : 0.01
- epochs : 2
- random seed : 42
- optimizer : adamW


- learning rate : A relatively small learning rate is appropriate for fine-tuning a pretrained Transformer because the model already contains useful language representations.
- Batch size : based on available hardware and memory.
- weight decay : regularization to reduce overfitting.
- max length : Controls computational cost while retaining the majority of the information needed for this classification task.

#### validation 
- at end of each epoch
- It is used to determine which checkpoint performs best.
- The checkpoint with the lowest val loss is the best teacher model selected.

#### For testing :
- Test Accuracy
- Macro Precision
- Macro Recall
- Macro F1

- per class metrics 
- confusion matrix



#### Final results 

{'eval_loss': '0.1902', 'eval_accuracy': '0.9486', 'eval_runtime': '150.9', 'eval_samples_per_second': '79.5', 'eval_steps_per_second': '4.969', 'epoch': '2'} 

- Accuracy: 94.41%
- Macro Precision: 94.40%
- Macro Recall: 94.41%
- Macro F1: 94.40%

Per-class F1:
- World: 95.36%
- Sports: 98.54%
- Business: 91.66%
- Sci/Tech: 92.06%

##### confusion matrix

1. Sports is the easiest class.
1,884 of 1,900 Sports articles are correctly classified, giving 99.2% recall. 
Only 16 are misclassified.

2. The main difficulty is Business ↔ Sci/Tech.
This is the most important pattern in your teacher:

118 Business → Sci/Tech
103 Sci/Tech → Business

So there are 221 errors between these two classes.

This makes sense semantically: technology/business news can contain very similar vocabulary, especially around companies, products, markets, acquisitions, and technology developments.

3. World is relatively strong.
1,799/1,900 World articles are correct. Its largest confusion is with Business:

49 World → Business
35 World → Sci/Tech

4. Sports is clearly separated from the other categories.
There are very few Sports confusions:

7 Sports → World
6 Sports → Business
3 Sports → Sci/Tech

That supports the 98.54% F1 you obtained for Sports.

