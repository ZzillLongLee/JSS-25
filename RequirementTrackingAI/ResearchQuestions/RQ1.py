import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, fbeta_score, balanced_accuracy_score
from torch.amp import autocast, GradScaler
from sklearn.model_selection import StratifiedKFold
import json
import time
import numpy as np

from sklearn.utils.class_weight import compute_class_weight

# Import existing modules
from Dict.DictionaryBuilder import DapaKomoranLoader
from Model.Training import CustomModel
import torch
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from datasets import Dataset
import re


class ComprehensiveModelPreprocessingExperiment:
    """Comprehensive experiment testing multiple models and preprocessing strategies"""

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

        # Define models to test
        self.models = {
            'KR-SBERT': 'snunlp/KR-SBERT-V40K-klueNLI-augSTS',
            'Ko-SRoBERTa': 'jhgan/ko-sroberta-multitask',
            'KoSimCSE': 'BM-K/KoSimCSE-roberta-multitask',
            'DistilUSE': 'sentence-transformers/distiluse-base-multilingual-cased-v2',
            'MiniLM': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
            'Qwen': 'Qwen/Qwen3-Embedding-0.6B',
            'BGE-M3': 'BAAI/bge-m3'
        }

        # Define preprocessing strategies
        self.strategies = ['full_text', 'noun_verb_only', 'noun_verb_adj']

        # Define layers to test
        self.test_layers = [1, 2, 3, 4, 5]

        # Check GPU availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.use_amp = torch.cuda.is_available()
        if self.use_amp:
            print("Automatic Mixed Precision (AMP) is enabled for CUDA devices.")
        else:
            print("AMP is disabled (running on CPU or disabled manually).")

        # Initialize preprocessing components
        self.dapa_loader = None
        self.dapa_available = self._init_dapa_loader()

    def _init_dapa_loader(self):
        """Initializes the DAPA loader safely"""
        try:
            import os
            os.environ['JAVA_OPTS'] = '-Xmx2g -Xms1g -XX:+UseG1GC'

            self.dapa_loader = DapaKomoranLoader()
            print("DAPA dictionary loaded successfully")
            return True

        except Exception as e:
            print(f"DAPA dictionary loading failed: {e}")
            print("Falling back to alternative POS tagging...")
            self._init_fallback_pos_tagger()
            return False

    def _init_fallback_pos_tagger(self):
        """Initializes a fallback POS tagger if DAPA fails"""
        try:
            from konlpy.tag import Okt
            self.fallback_tagger = Okt()
            print("Fallback POS tagger (Okt) loaded")
        except ImportError:
            print("KoNLPy not available. Using simple tokenization.")
            self.fallback_tagger = None

    def run_full_experiment(self):
        """Run comprehensive experiment across all models, strategies, and layers"""
        print("Starting Comprehensive Model-Preprocessing-Layer Experiment")
        print("=" * 80)

        total_experiments = len(self.models) * len(self.strategies) * len(self.test_layers) * 3
        print(f"Total experiments to run: {total_experiments}")
        print(f"Estimated time: {total_experiments * 10 / 60:.1f} hours")
        print()

        all_results = {}
        start_time = time.time()

        for model_name, model_path in self.models.items():
            print(f"\n{'=' * 20} TESTING MODEL: {model_name} {'=' * 20}")

            try:
                model_experiment = self._create_model_experiment(model_name, model_path)

                if model_experiment is None:
                    print(f"Skipping {model_name} due to initialization failure")
                    all_results[model_name] = {'error': 'Model initialization failed'}
                    continue

                model_results, model_analysis = model_experiment.run_comprehensive_comparison(self.test_layers)

                all_results[model_name] = {
                    'model_path': model_path,
                    'results': model_results,
                    'analysis': model_analysis
                }

                self._show_model_summary(model_name, model_analysis)

            except Exception as e:
                print(f"Error with model {model_name}: {e}")
                all_results[model_name] = {'error': str(e)}
                continue

        self._analyze_across_models(all_results)
        global_optimal = self._find_global_optimal_configurations(all_results)
        self._save_comprehensive_results(all_results, global_optimal)

        end_time = time.time()
        print(f"\nFull experiment completed in {(end_time - start_time) / 3600:.1f} hours")

        return all_results, global_optimal

    def _create_model_experiment(self, model_name, model_path):
        """Create a model-specific experiment instance"""

        class ModelSpecificExperiment:
            def __init__(self, dataset_path, model_name, tokenizer, device, dapa_loader, dapa_available, use_amp):
                self.dataset_path = dataset_path
                self.model_name = model_name
                self.tokenizer = tokenizer
                self.device = device
                self.dapa_loader = dapa_loader
                self.dapa_available = dapa_available
                self.use_amp = use_amp

            def _safe_pos_tagging(self, text):
                try:
                    if self.dapa_available and self.dapa_loader is not None:
                        return self.dapa_loader.get_pos_tags(text)
                except Exception as e:
                    print(f"DAPA tagging failed for text: {text[:50]}... Error: {e}")
                words = text.split()
                return [(word, 'UNKNOWN') for word in words]

            def preprocess_full_text(self, text):
                try:
                    if self.dapa_available and self.dapa_loader is not None:
                        return self.dapa_loader.process_compound_terms(text).strip()
                except:
                    pass
                return re.sub(r'\s+', ' ', text.strip())

            def preprocess_noun_verb_only(self, text):
                try:
                    pos_tags = self._safe_pos_tagging(text)
                    selected_words = [word for word, pos in pos_tags if
                                      pos.startswith('NN') or pos.startswith('VV') or pos == 'NNP' or
                                      pos.startswith('Noun') or pos.startswith('Verb')]
                    result = ' '.join(selected_words)
                    return result if result.strip() else text
                except Exception as e:
                    print(f"Error in noun_verb_only preprocessing: {e}")
                    return text

            def preprocess_noun_verb_adj(self, text):
                try:
                    pos_tags = self._safe_pos_tagging(text)
                    selected_words = [word for word, pos in pos_tags if
                                      pos.startswith('NN') or pos.startswith('VV') or pos.startswith('JJ') or
                                      pos == 'NNP' or pos.startswith('Noun') or pos.startswith('Verb') or
                                      pos.startswith('Adjective')]
                    result = ' '.join(selected_words)
                    return result if result.strip() else text
                except Exception as e:
                    print(f"Error in noun_verb_adj preprocessing: {e}")
                    return text

            def prepare_dataset(self, strategy='full_text'):
                print(f"Loading dataset: {self.dataset_path}")
                df = pd.read_excel(self.dataset_path)

                print(f"Applying {strategy} preprocessing...")

                if strategy == 'full_text':
                    preprocess_func = self.preprocess_full_text
                elif strategy == 'noun_verb_only':
                    preprocess_func = self.preprocess_noun_verb_only
                elif strategy == 'noun_verb_adj':
                    preprocess_func = self.preprocess_noun_verb_adj
                else:
                    raise ValueError(f"Unknown preprocessing strategy: {strategy}")

                processed_data = []
                for idx, row in df.iterrows():
                    src_processed = preprocess_func(row['src_sentence'])
                    tar_processed = preprocess_func(row['tar_sentence'])
                    processed_data.append({
                        'src_sentence': src_processed,
                        'tar_sentence': tar_processed,
                        'Label': int(row['Label'])
                    })
                return processed_data

            def run_comprehensive_comparison(self, test_layers=None):
                if test_layers is None: test_layers = [1, 2, 3, 4, 5]
                strategies = ['full_text', 'noun_verb_only', 'noun_verb_adj']
                all_results = {}
                for strategy in strategies:
                    print(f"\n--- Testing {strategy} with {self.model_name} ---")
                    data = self.prepare_dataset(strategy)
                    strategy_results = {}
                    for layer in test_layers:
                        print(f"Layer {layer}...")
                        layer_results = self._train_and_evaluate_with_layers(data, strategy, layer, k_folds=3)
                        strategy_results[f'layer_{layer}'] = layer_results
                        best_f1 = layer_results['avg_results']['f1']
                        balanced_acc = layer_results['avg_results']['balanced_accuracy']
                        print(f"   F1: {best_f1:.3f}, Balanced Accuracy: {balanced_acc:.3f}")
                    all_results[strategy] = strategy_results
                optimal_configs = self._find_optimal_configurations(all_results)
                return all_results, optimal_configs

            def _train_and_evaluate_with_layers(self, data, strategy_name, additional_layers, k_folds=3):
                X = np.array(data)
                y = np.array([item['Label'] for item in data])
                skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
                fold_results = []
                for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
                    train_data = X[train_idx].tolist()
                    test_data = X[test_idx].tolist()
                    fold_result = self._train_fold(train_data, test_data, additional_layers)
                    fold_results.append(fold_result)
                avg_results = {}
                metrics = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 'f2']
                for metric in metrics:
                    values = [fold[metric] for fold in fold_results]
                    avg_results[metric] = np.mean(values)
                return {'avg_results': avg_results, 'fold_results': fold_results}

            def _train_fold(self, train_data, test_data, additional_layers):
                train_dataset = Dataset.from_list(train_data)
                test_dataset = Dataset.from_list(test_data)

                def preprocess_function(examples):

                    if 'qwen' in self.model_name.lower():
                        src_texts = [f"Query: {text}" for text in examples['src_sentence']]

                        tar_texts = [f"Passage: {text}" for text in examples['tar_sentence']]
                    else:
                        src_texts = examples['src_sentence']
                        tar_texts = examples['tar_sentence']

                    encoded = self.tokenizer(src_texts, tar_texts, truncation=True, padding=True, max_length=256)
                    encoded['labels'] = examples['Label']
                    return encoded

                train_encoded = train_dataset.map(preprocess_function, batched=True)
                test_encoded = test_dataset.map(preprocess_function, batched=True)
                train_encoded.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
                test_encoded.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

                if 'qwen' in self.model_name.lower():
                    batch_size = 8 if additional_layers <= 3 else 4
                else:
                    batch_size = 16 if additional_layers <= 3 else 8

                data_collator = DataCollatorWithPadding(self.tokenizer)
                train_loader = DataLoader(train_encoded, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
                test_loader = DataLoader(test_encoded, batch_size=batch_size, collate_fn=data_collator)

                try:
                    model = CustomModel(self.model_name, num_labels=2, additional_layers=additional_layers)
                    model.to(self.device)
                except Exception as e:
                    print(f"Model initialization failed: {e}")
                    return {'accuracy': 0.5, 'balanced_accuracy': 0.5, 'precision': 0.5, 'recall': 0.5, 'f1': 0.5, 'f2': 0.5}

                if 'qwen' in self.model_name.lower():
                    learning_rate = 1e-6
                else:
                    learning_rate = 2e-5

                optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
                scaler = GradScaler() if self.use_amp else None

                train_labels = [item['Label'] for item in train_data]
                class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
                class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)

                for epoch in range(1):
                    model.train()
                    total_loss = 0
                    for batch in train_loader:
                        try:
                            input_ids = batch['input_ids'].to(self.device)
                            attention_mask = batch['attention_mask'].to(self.device)
                            labels = batch['labels'].to(self.device)
                            optimizer.zero_grad()

                            if self.use_amp and scaler is not None:
                                with autocast(device_type=self.device.type):

                                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels,
                                                    class_weights=class_weights)
                                    loss = outputs['loss']
                                scaler.scale(loss).backward()
                                # <--- FIX: Add gradient clipping
                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                                scaler.step(optimizer)
                                scaler.update()
                            else:

                                outputs = model(input_ids, attention_mask=attention_mask, labels=labels,
                                                class_weights=class_weights)
                                loss = outputs['loss']
                                loss.backward()
                                # <--- FIX: Add gradient clipping
                                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                                optimizer.step()

                            total_loss += loss.item()

                        except RuntimeError as e:
                            if "out of memory" in str(e):
                                print("GPU OOM in training, skipping batch")
                                torch.cuda.empty_cache()
                                continue
                            else:
                                raise e
                    avg_train_loss = total_loss / len(train_loader)
                    print(f"Epoch {epoch + 1} completed, Average Loss: {avg_train_loss:.4f}")

                model.eval()
                predictions, true_labels = [], []
                with torch.no_grad():
                    for batch in test_loader:
                        try:
                            input_ids, attention_mask, labels = batch['input_ids'].to(self.device), batch[
                                'attention_mask'].to(self.device), batch['labels'].to(self.device)
                            outputs = model(input_ids, attention_mask=attention_mask)
                            logits = outputs['logits']
                            preds = torch.argmax(logits, dim=1)
                            predictions.extend(preds.cpu().numpy())
                            true_labels.extend(labels.cpu().numpy())
                        except RuntimeError as e:
                            if "out of memory" in str(e):
                                print("GPU OOM in evaluation, skipping batch"); torch.cuda.empty_cache(); continue
                            else:
                                raise e
                del model
                torch.cuda.empty_cache()

                if not predictions or not true_labels:
                    return {'accuracy': 0.5, 'balanced_accuracy': 0.5, 'precision': 0.5, 'recall': 0.5, 'f1': 0.5, 'f2': 0.5}

                try:
                    accuracy = accuracy_score(true_labels, predictions)
                    balanced_acc = balanced_accuracy_score(true_labels, predictions)
                    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions,
                                                                               average='binary', zero_division=0)
                    f2 = fbeta_score(true_labels, predictions, beta=2, average='binary', zero_division=0)
                except Exception as e:
                    print(f"Metric calculation failed: {e}")
                    return {'accuracy': 0.5, 'balanced_accuracy': 0.5, 'precision': 0.5, 'recall': 0.5, 'f1': 0.5, 'f2': 0.5}

                return {'accuracy': accuracy, 'balanced_accuracy': balanced_acc, 'precision': precision, 'recall': recall, 'f1': f1, 'f2': f2}

            def _find_optimal_configurations(self, all_results):
                strategies = ['full_text', 'noun_verb_only', 'noun_verb_adj']
                layers = [1, 2, 3, 4, 5]
                best_configs = {'f1': {'score': 0}, 'f2': {'score': 0}, 'weighted': {'score': 0}}
                for strategy in strategies:
                    if strategy in all_results:
                        for layer in layers:
                            layer_key = f'layer_{layer}'
                            if layer_key in all_results[strategy]:
                                results = all_results[strategy][layer_key]['avg_results']
                                weighted_score = (0.25 * results['f1'] + 0.35 * results['f2'] + 0.15 * results['accuracy'] + 0.15 * results['balanced_accuracy'] + 0.1 * results['precision'])
                                if results['f1'] > best_configs['f1']['score']: best_configs['f1'] = {
                                    'strategy': strategy, 'layer': layer, 'score': results['f1']}
                                if results['f2'] > best_configs['f2']['score']: best_configs['f2'] = {
                                    'strategy': strategy, 'layer': layer, 'score': results['f2']}
                                if weighted_score > best_configs['weighted']['score']:
                                    best_configs['weighted'] = {'strategy': strategy, 'layer': layer,
                                                                'score': weighted_score, **results}
                return best_configs

        try:
            # ===== 수정된 부분: Qwen 모델일 경우 padding_side='left' 옵션 추가 =====
            if 'qwen' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side='left')
                print(f"Tokenizer for Qwen loaded with padding_side='left'")
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                print(f"Tokenizer loaded for {model_name}")
            # ====================================================================
        except Exception as e:
            print(f"Failed to load tokenizer for {model_name}: {e}")
            return None
        return ModelSpecificExperiment(self.dataset_path, model_path, tokenizer, self.device, self.dapa_loader,
                                       self.dapa_available, self.use_amp)

    def _show_model_summary(self, model_name, analysis):
        print(f"\n{model_name} Summary:")
        if 'weighted' in analysis and analysis['weighted'].get('strategy'):
            best = analysis['weighted']
            print(
                f"   Best Config: {best['strategy']} + Layer {best['layer']} -> F1: {best['f1']:.3f} | F2: {best['f2']:.3f} | Recall: {best['recall']:.3f} | Balanced Accuracy: {best['balanced_accuracy']:.3f}")

    def _analyze_across_models(self, all_results):
        print(f"\n{'=' * 20} CROSS-MODEL ANALYSIS {'=' * 20}")
        # Placeholder for future implementation
        pass

    def _find_global_optimal_configurations(self, all_results):
        print("\nFINDING GLOBAL OPTIMAL CONFIGURATIONS")
        # Placeholder for future implementation
        return {'top_weighted': [], 'top_f1': [], 'top_f2': [], 'top_recall': [], 'all_configs': []}

    def _save_comprehensive_results(self, all_results, global_optimal):
        output = {'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'), 'dataset': self.dataset_path,
                  'models_tested': list(self.models.keys()), 'results_by_model': all_results,
                  'global_optimal_configurations': global_optimal}
        filename = 'comprehensive_model_preprocessing_analysis.json'
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\nComprehensive results saved to: {filename}")


if __name__ == "__main__":
    dataset_path = r'/home/teddy/PycharmProjects/RequirementTrackingAI/Dataset.xlsx'
    comprehensive_experiment = ComprehensiveModelPreprocessingExperiment(dataset_path)
    all_results, global_optimal = comprehensive_experiment.run_full_experiment()
    print("\nExperiment completed successfully!")