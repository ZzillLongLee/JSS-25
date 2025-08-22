from transformers import AutoTokenizer, AutoConfig, AutoModel
import torch
import torch.nn as nn
import os


# Custom model class definition (minimal for loading purposes)
class CustomModel(nn.Module):
    def __init__(self, model_name, num_labels, additional_layers=2):
        super(CustomModel, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
        self.model = AutoModel.from_pretrained(model_name, config=self.config)
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

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs.last_hidden_state[:, 0]  # Use the first token ([CLS] token)
        pooled_output = self.dropout(pooled_output)
        logits = self.additional_layers(pooled_output)
        return {"logits": logits}


def preprocess_single_instance(sentence1, sentence2):
    encoded = tokenizer(sentence1, sentence2, truncation=True, padding=True, return_tensors='pt')
    return encoded

    # Define a function to predict the label for a single data instance


def predict_single_instance(model, tokenizer, sentence1, sentence2):
    encoded = preprocess_single_instance(sentence1, sentence2)
    with torch.no_grad():
        outputs = model(**encoded)
    logits = outputs['logits']  # Extract logits from the dictionary
    predicted_label = torch.argmax(logits, dim=1).item()
    return predicted_label


if __name__ == '__main__':
    # Load the fine-tuned model and tokenizer
    model_save_path = './fine_tuned_model'
    tokenizer = AutoTokenizer.from_pretrained(model_save_path)
    model = CustomModel(model_name='jhgan/ko-sroberta-multitask', num_labels=2, additional_layers=2)
    model.load_state_dict(torch.load(os.path.join(model_save_path, 'pytorch_model.bin')))
    model.eval()  # Set the model to evaluation mode

    # Example usage
    sentence1 = "Example sentence 1."
    sentence2 = "Example sentence 2."
    predicted_label = predict_single_instance(model, tokenizer, sentence1, sentence2)
    print(f'Predicted label: {predicted_label}')
