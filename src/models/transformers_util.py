import transformers 
from transformers import BertModel

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np

"""Code adapted from W207 Week 13 Session Notebook on Sentiment Analysis using Transformers"""

class TextDataset(Dataset):
    """Class that holds the input text, corresponding labels, and a tokenizer/encoder"""
    def __init__(self, texts, targets, tokenizer, max_len):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            truncation=True,
            pad_to_max_length=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt',
            )
        
        return {
            'text_text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
            }

    
class ViewCategoryClassifier(nn.Module):
    """The classifier model built from a pretrained Bert model"""
    def __init__(self, n_classes, model_name):
        super(ViewCategoryClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name, return_dict=False)
        self.drop = nn.Dropout(p=0.3)
        #self.out1 = nn.Linear(self.bert.config.hidden_size, 128)
        #self.drop1 = nn.Dropout(p=0.4)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

        self.softmax = nn.Softmax()
        
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        output = self.drop(pooled_output)
        #output = self.out1(output)
        #output = self.drop1(output)
        output = self.out(output)

        return self.softmax(output)


def create_data_loader(texts, labels, tokenizer, max_len, batch_size):
    """Function for creating a DataLoader instance for each input & output dataset"""
    ds = TextDataset(
        texts=texts.to_numpy(),
        targets=labels,
        tokenizer=tokenizer,
        max_len=max_len
    )

    return DataLoader(ds,
                      batch_size=batch_size,
                      num_workers=0)


def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    """Function that executes each training epoch"""
    model = model.train()
    losses = []
    correct_predictions = 0

    for data in data_loader:
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        targets = data['targets'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
            )
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs,targets)

        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples):
    """Function that evaluates the model's accuracy and loss given an input dataset"""
    model = model.eval()
    
    losses = []
    correct_predictions = 0
    
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())
    
    return correct_predictions.double() / n_examples, np.mean(losses)


def get_predictions(model, data_loader, device):
    """Function that uses the model to generate predictions from a given input dataset"""
    model = model.eval()
    text_texts = []
    predictions = []
    prediction_probs = []
    real_values = []
    
    with torch.no_grad():
        for d in data_loader:
            texts = d["text_text"]
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)
            text_texts.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(outputs)
            real_values.extend(targets)
            
    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    
    return text_texts, predictions, prediction_probs, real_values