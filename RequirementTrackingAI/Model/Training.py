import torch
from torch import Tensor
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import KFold
import numpy as np
import os
from transformers import AutoConfig, AutoModel
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from datasets import Dataset

def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

# Custom model class
class CustomModel(nn.Module):
    def __init__(self, model_name, num_labels, additional_layers=2):
        super(CustomModel, self).__init__()

        # 모델 종류를 확인하여 인스턴스 변수에 저장
        self.is_qwen_model = 'qwen' in model_name.lower()

        self.config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
        self.model = AutoModel.from_pretrained(model_name,
                                               config=self.config)
        self.dropout = nn.Dropout(0.1)
        self.additional_layers = self._make_additional_layers(additional_layers, self.model.config.hidden_size,
                                                              num_labels)

    def _make_additional_layers(self, num_layers, hidden_size, num_labels):
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, num_labels))
        return nn.Sequential(*layers)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None, class_weights=None):
        if token_type_ids is not None:
            outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        else:
            outputs = self.model(input_ids, attention_mask=attention_mask)

        # ===== 변경된 부분: 모델 종류에 따라 다른 풀링 방식 적용 =====
        if self.is_qwen_model:
            # Qwen 모델일 경우: 마지막 토큰 풀링
            pooled_output = last_token_pool(outputs.last_hidden_state, attention_mask)
        else:
            # 다른 모델일 경우: CLS 토큰 풀링 (기존 방식)
            pooled_output = outputs.last_hidden_state[:, 0]
        # ========================================================

        pooled_output = self.dropout(pooled_output)
        logits = self.additional_layers(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(weight=class_weights)
            loss = loss_fct(logits.view(-1, self.config.num_labels),
                            labels.view(-1).long())

        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}


# Preprocess function
def preprocess_function(dataset):
    encoded = tokenizer(dataset['src_sentence'], dataset['tar_sentence'], truncation=True, padding=True)
    encoded['labels'] = [int(label) for label in dataset['Label']]  # Convert true/false to 1/0
    return encoded

if __name__ == '__main__':

    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset and tokenizer
    model_name = r'C:\Users\DTaQ\PycharmProjects\RequirementTrackingAI\SentenceBert\distiluse_sbert_v2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load dataset using pandas
    dataset_path = r'C:\Users\DTaQ\PycharmProjects\RequirementTrackingAI\Data\filtered_first_dataset.csv'
    df = pd.read_csv(dataset_path)

    # Convert pandas DataFrame to datasets Dataset
    dataset = Dataset.from_pandas(df)

    # Apply preprocessing
    encoded_dataset = dataset.map(preprocess_function, batched=True)
    encoded_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    # Split the dataset
    train_size = int(0.9 * len(encoded_dataset))
    train_dataset, eval_dataset = random_split(encoded_dataset, [train_size, len(encoded_dataset) - train_size])

    # DataLoader with padding collator
    data_collator = DataCollatorWithPadding(tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=data_collator)
    eval_loader = DataLoader(eval_dataset, batch_size=16, collate_fn=data_collator)

    # Initialize the custom model
    num_labels = 2
    custom_model = CustomModel(model_name=model_name, num_labels=num_labels, additional_layers=2)
    custom_model.to(device)  # Move model to GPU
    custom_model.train()

    # Optimizer
    optimizer = optim.AdamW(custom_model.parameters(), lr=2e-5)

    # Training loop
    epochs = 3
    for epoch in range(epochs):
        total_loss = 0
        custom_model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = custom_model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs['loss']
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1}, Loss: {avg_train_loss}')

        custom_model.eval()
        eval_preds, eval_labels = [], []
        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = custom_model(input_ids, attention_mask=attention_mask, labels=labels)
                logits = outputs['logits']
                preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
                label_ids = labels.detach().cpu().numpy()
                eval_preds.extend(preds)
                eval_labels.extend(label_ids)

        eval_preds = np.array(eval_preds)
        eval_labels = np.array(eval_labels)
        acc = accuracy_score(eval_labels, eval_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(eval_labels, eval_preds, average='binary')
        print(f'Validation Accuracy: {acc}, F1: {f1}, Precision: {precision}, Recall: {recall}')

    # Save the model and tokenizer
    model_save_path = './fine_tuned_model'
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    torch.save(custom_model.state_dict(), os.path.join(model_save_path, 'pytorch_model.bin'))
    tokenizer.save_pretrained(model_save_path)
