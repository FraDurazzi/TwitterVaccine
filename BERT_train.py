import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from tqdm.auto import tqdm
import re, os
import torch
import torch.nn.functional as Functional
from torch.utils.data import DataLoader
from datasets import load_dataset
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForSequenceClassification, TrainingArguments, Trainer, get_scheduler
from multimodal_transformers.model import AutoModelWithTabular
from multimodal_transformers.model import TabularConfig
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

os.chdir('/home/work/applicata/francesco.durazzi2/vaccines/')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TRANSFORMERS_CACHE'] = '/home/work/applicata/francesco.durazzi2/'
device='cuda'
phase='inference'

#metric = evaluate.load("accuracy")
def compute_metrics(logits_and_labels):
    logits, labels = logits_and_labels
    #predictions = np.argmax(logits, axis=-1)
    acc = np.mean(predictions == labels)
    f1 = f1_score(labels, predictions, average = 'weighted')
    f1s= f1_score(labels, predictions, average = None)
    return {'accuracy': acc, 'f1_score': f1, 'f1_scores': f1s}



#with open('df_full.pkl', 'rb') as f:
#    df = pickle.load(f)

#df_anno=df[df['annotation'].notna()]
#df_anno.loc[:,'text']=df_anno['text'].apply(lambda x: x.replace('\n',' ').replace('\t',''))
## Remove other unsafe characters
##df_anno['text']=df_anno['text'].apply(lambda x: re.sub(r'[^\x00-\x7F]+',' ', x))
##df_anno[['text','annotation']].rename(columns={'text':'sentence'}).to_csv('annotated_dataset.csv', index=True, sep='\t',encoding='utf-8')
## For loop to write a csv file
#with open('annotated_dataset.csv', 'w',encoding='utf-8') as f:
#    f.write('id\tsentence\tlabel\n')
#    for i in range(len(df_anno)):
#        f.write(str(df_anno.index[i])+'\t'+df_anno.iloc[i]['text']+'\t'+df_anno.iloc[i]['annotation']+'\n')

ds = load_dataset('csv', data_files='annotated_dataset.csv', split='train',encoding='utf-8',delimiter='\t')
id2label = {0:'AntiVax', 1:'Neutral', 2:'ProVax'}
label2id = {'AntiVax':0, 'Neutral':1, 'ProVax':2}
# Remove entries not in the label set
ds = ds.filter(lambda e: e['label'] in label2id)
# Convert labels to ids
ds = ds.map(lambda e: {'labels': label2id[e['label']]}, remove_columns=['label'])


# Split in train, validation and test
ds_train = ds.train_test_split(test_size=0.3, seed=42)
#print(ds_train)
ds_valid = ds_train['test'].train_test_split(test_size=0.5, seed=42)
#print(ds_valid)

tokenizer = AutoTokenizer.from_pretrained('m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0')
def tokenize_fn(batch):
    return tokenizer(batch['sentence'],padding='max_length', max_length=512, truncation=True)

tokenized_dataset = ds_train.map(tokenize_fn, batched = True)
tokenized_validation = ds_valid.map(tokenize_fn, batched = True)
tokenized_dataset = tokenized_dataset.remove_columns(["sentence", "id"])
tokenized_validation = tokenized_validation.remove_columns(["sentence", "id"])

tokenized_dataset.set_format("torch")
tokenized_validation.set_format("torch")
# Detect max length
#max_len = max([len(x['input_ids']) for x in tokenized_dataset['train']])
#print(max_len)
#exit()

train_dataloader = DataLoader(tokenized_dataset['train'], shuffle=True, batch_size=64)
eval_dataloader = DataLoader(tokenized_validation['train'], batch_size=128)

if phase=='train':
    model= AutoModelForSequenceClassification.from_pretrained('m-polignano-uniba/bert_uncased_L-12_H-768_A-12_italian_alb3rt0',num_labels=3).to(device)

    
    optimizer=torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    progress_bar = tqdm(range(num_training_steps))
    # Set seed for reproducibility
    torch.manual_seed(42)
    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: torch.from_numpy(np.asarray(v).astype('long')).to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
    
    # Save the model
    model.save_model('alb3rt0_3epochs')


if phase=='inference':
    model= AutoModelForSequenceClassification.from_pretrained('alb3rt0_3epochs',num_labels=3).to(device)

predictions = []
labels = []
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions.append(torch.argmax(logits, dim=-1))
    labels.append(batch["labels"])

predictions = torch.cat(predictions).cpu().numpy()
labels = torch.cat(labels).cpu().numpy()
print(compute_metrics((predictions, labels)))


    

exit()

training_args = TrainingArguments(output_dir='training_dir',
                                  evaluation_strategy='epoch',
                                  #save_strategy='epoch',
                                  num_train_epochs=3,
                                  per_device_train_batch_size=64,
                                  per_device_eval_batch_size=128,
                                  )



trainer = Trainer(model,
                  training_args,
                  train_dataset = tokenized_dataset["train"],
                  eval_dataset = tokenized_validation["train"],
                  tokenizer=tokenizer,
                  optimizers=(torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01),
                              None ),
                  compute_metrics=compute_metrics)

trainer.train()

# Evaluate the model
trainer.evaluate()

# Save the model
trainer.save_model('alb3rt0_3epochs')
