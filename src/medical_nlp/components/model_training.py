from transformers import (AutoModelForSequenceClassification,
                          AutoTokenizer, AutoModel,
                          TrainingArguments, Trainer)
import numpy as np
from datasets import load_dataset, Split
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import mlflow
import os
from datetime import datetime
from urllib.parse import urlparse
from medical_nlp.entity.config_entity import TrainingConfig
from dotenv import load_dotenv
load_dotenv()




class ModelTrainerHF(object):
    def __init__(self, config:TrainingConfig, device='cpu'):
        self.config = config
        self.dataset = self.load_dataset()
        self.tokenizer, self.model = self.load_model()
        # self.set_loaders()
        self.device = device if device != '' else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.set_seed()
        
    def load_model(self):
        # mapping integer labels to string labels and vv
        id2label = {0: 'Medical Necessity', 1: 'Experimental/Investigational', 2: 'Urgent Care'}
        label2id = {'Medical Necessity': 0, 'Experimental/Investigational': 1, 'Urgent Care': 2}
        tokenizer = AutoTokenizer.from_pretrained(self.config.params_model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
                    self.config.params_model_name, num_labels=self.config.params_classes, id2label=id2label, label2id=label2id
                )
        print(model.classifier)
        return tokenizer, model
    
    def _label_prep(self, row):
        label_dict = {'Medical Necessity': 0, 'Experimental/Investigational': 1, 'Urgent Care': 2}
        label_type = label_dict[row['Type']]
        return {'labels': label_type}
    
    def _tokenize(self, row):
        return self.tokenizer(row['Findings'],
                            truncation=True,
                            padding='max_length',
                            max_length=50)
                        
    
    def load_dataset(self):
        dataset = dataset = load_dataset(path='csv',data_files=self.config.training_data + 'Independent_Medical_Reviews_Custom.csv',split=Split.TRAIN)
        return dataset
    
    def set_seed(self, seed=42):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    def set_loaders(self):
        self.dataset = self.dataset.map(self._label_prep)
        self.dataset = self.dataset.shuffle(seed=42)
        self.dataset = self.dataset.train_test_split(test_size=0.2)
        train_dataset = self.dataset['train']
        test_dataset = self.dataset['test']
        tokenized_train_dataset = train_dataset.map(self._tokenize, batched=True)
        tokenized_test_dataset = test_dataset.map(self._tokenize, batched=True)

        tokenized_train_dataset.set_format(
                            type='torch',
                            columns=['input_ids', 'attention_mask', 'labels'])

        tokenized_test_dataset.set_format(
                            type='torch',
                            columns=['input_ids', 'attention_mask', 'labels'])
        
        return tokenized_train_dataset, tokenized_test_dataset
    
    def load_trainer(self):
        
        tokenized_train_dataset, tokenized_test_dataset = self.set_loaders()
        args = TrainingArguments(
                self.config.nlp_trained_model_path + self.config.params_model_name,
                # save_strategy="epoch",
                evaluation_strategy="epoch",
                learning_rate=self.config.params_learning_rate,
                per_device_train_batch_size=self.config.params_batch_size,
                per_device_eval_batch_size=self.config.params_batch_size*2,
                num_train_epochs=self.config.params_epochs,
                # load_best_model_at_end=True,
                metric_for_best_model="accuracy",
                logging_dir='logs',
                remove_unused_columns=False,
                gradient_accumulation_steps=8,
            )
        args.report_to = ["mlflow"]
        
        def _compute_metrics(eval_pred):
            predictions = eval_pred.predictions
            labels = eval_pred.label_ids
            predictions = np.argmax(predictions, axis=1)
            return {"accuracy": (predictions == labels).mean()}
        
        trainer = Trainer(model=self.model,
                args=args,
                train_dataset=tokenized_train_dataset,
                eval_dataset=tokenized_test_dataset,
                compute_metrics=_compute_metrics)
        
        return trainer
    
    def train(self):
        now = datetime.now()
        time_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        os.environ["MLFLOW_EXPERIMENT_NAME"] = "medical_nlp_" + time_str
        os.environ["MLFLOW_FLATTEN_PARAMS"] = "1"
        
        trainer = self.load_trainer()
        trainer.train()
        self.save_checkpoint(trainer)
        mlflow.end_run()
    
    
    def evaluation(self):
        self.load_checkpoint()
        trainer = self.load_trainer()
        outputs = trainer.evaluate()
        print(outputs)
            
    def save_checkpoint(self, trainer):
        trainer.save_model(self.config.nlp_trained_model_path + self.config.params_model_name)
        
    def load_checkpoint(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(self.config.nlp_trained_model_path + self.config.params_model_name)
        self.model.to(self.device)
    
    
    def predict(self, text):
        self.load_checkpoint()
        tokenized_text = self.tokenizer(text, return_tensors='pt')
        tokenized_text.to(self.device)
        outputs = self.model(**tokenized_text)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        
        return self.model.config.id2label[predicted_class_idx]
    
    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics({'train_loss': np.mean(self.losses),'val_loss': np.mean(self.val_losses), 'train_accuracy': np.mean(self.accuracy), 'val_accuracy': np.mean(self.val_accuracy)})
        
            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.pytorch.log_model(self.model, "model", registered_model_name=self.config.params_model_name)
            else:
                mlflow.pytorch.log_model(self.model, "model")
                
                


class ModelTrainerPyTorch(object):
    def __init__(self, config:TrainingConfig, loss_fn=None, optimizer=None):
        self.config = config
        self.model = self.load_model()
        self.loss_fn = loss_fn if loss_fn else nn.BCEWithLogitsLoss()
        self.optimizer = optimizer if optimizer else optim.Adam(self.model.parameters(), lr=self.config.params_learning_rate)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        
        self.train_loader, self.val_loader = self.set_loaders()
        
        self.losses = []
        self.val_losses = []
        self.accuracy = []
        self.val_accuracy = []
        self.total_epoches = 0
        
        self.train_step_fn = self._make_train_step_fn()
        self.val_step_fn = self._make_val_step_fn()
        
    def load_model(self):
        # return torch.load(self.config.nlp_updated_base_model_path)
        bert_model = AutoModel.from_pretrained(self.config.params_model_name)
        return BERTClassifier(bert_model, 128, n_outputs=1)
    
    def set_seed(self, seed=42):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    def label_prep(self, row):
        label_dict = {'Medical Necessity': 0, 'Experimental/Investigational': 1, 'Urgent Care': 2}
        label_type = label_dict[row['Type']]
        return {'labels': label_type}

    
    def set_loaders(self):
        dataset = load_dataset(path='csv',data_files=self.config.training_data + 'Independent_Medical_Reviews_Custom.csv',split=Split.TRAIN)
        dataset = dataset.map(self.label_prep)
        shuffled_dataset = dataset.shuffle(seed=42)
        split_dataset = shuffled_dataset.train_test_split(test_size=0.2)
        train_dataset = split_dataset['train']
        test_dataset = split_dataset['test']
        auto_tokenizer = AutoTokenizer.from_pretrained(self.config.params_model_name)
        tokenizer_kwargs = dict(truncation=True,
                                padding=True,
                                max_length=30,
                                add_special_tokens=True)
        
        train_dataset_float = train_dataset.map(
            lambda row: {'labels': [float(row['labels'])]}
        )
        test_dataset_float = test_dataset.map(
            lambda row: {'labels': [float(row['labels'])]}
        )
        train_tensor_dataset = self._tokenize_dataset(train_dataset_float,
                                                'Findings',
                                                'labels',
                                                auto_tokenizer,
                                                **tokenizer_kwargs)
        test_tensor_dataset = self._tokenize_dataset(test_dataset_float,
                                                'Findings',
                                                'labels',
                                                auto_tokenizer,
                                                **tokenizer_kwargs)
        generator = torch.Generator()
        train_loader = DataLoader(
            train_tensor_dataset, batch_size=4,
            shuffle=True, generator=generator
        )
        test_loader = DataLoader(test_tensor_dataset, batch_size=8)
        
        return train_loader, test_loader
    
    # higher order function to be set and built globally and constructed the inner fuction without knowning x and y before hand
    def _make_train_step_fn(self):
        # single batch operation
        def perform_train_step_fn(x,y):
            # set the train mode
            self.model.train()
            
            # step 1: compute model output
            yhat = self.model(x)
            
            # step 2: compute the loss  
            loss= self.loss_fn(yhat, y)
            
            # step 2': compute accuracy 
            yhat = torch.argmax(yhat,1)
            total_correct = (yhat ==y).sum().item()
            total = y.shape[0]
            acc = total_correct/total
            
            # step 3: compute the gradient
            loss.backward()
            
            #step4: update parameters
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            #step 5: return the loss
            return loss.item() , acc
        return perform_train_step_fn
    
    def _make_val_step_fn(self):
        # single batch operation
        def perform_val_step_fn(x,y):
            # set the model in val mode
            self.model.eval()
            
            #step 1: compute the prediction
            yhat = self.model(x)
            
            #step 2: compute the loss
            loss = self.loss_fn(yhat, y)
            # step 2': compute accuracy 
            yhat = torch.argmax(yhat,1)
            total_correct = (yhat ==y).sum().item()
            total = y.shape[0]
            acc = total_correct/total
            
            return loss.item(), acc
        return perform_val_step_fn
    
    def _mini_batch(self, validation=False):
        # one epoch operation 
        if validation:
            data_loader = self.val_loader
            step_fn = self.val_step_fn
            
        else: 
            data_loader = self.train_loader
            step_fn = self.train_step_fn
            
        if data_loader is None:
            return None
        
        mini_batch_losses = []
        mini_batch_accs = []
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            mini_batch_loss, mini_batch_acc = step_fn(x_batch,y_batch)
            
            mini_batch_losses.append(mini_batch_loss)
            mini_batch_accs.append(mini_batch_acc)
        
        loss = np.mean(mini_batch_losses)
        acc = np.mean(mini_batch_accs)
        return loss, acc
    
    def train(self, seed=42):
        self.set_seed(seed)
        
        for epoch in range(self.config.params_epochs):
            self.total_epoches +=1
            
            # perform training on mini batches within 1 epoch
            loss, acc = self._mini_batch(validation=False)
            self.losses.append(loss)
            self.accuracy.append(acc)
            # now calc validation
            with torch.no_grad():
                val_loss, val_acc = self._mini_batch(validation=True)
                self.val_losses.append(val_loss)
                self.val_accuracy.append(val_acc)
                
            print(
                f'\nEpoch: {epoch+1} \tTraining Loss: {loss:.4f} \tValidation Loss: {val_loss:.4f}'
            )
            print(
                f'\t\tTraining Accuracy: {100 * acc:.2f}%\t Validation Accuracy: {100 * val_acc:.2f}%'
            )
        self.save_checkpoint()
            
    def save_checkpoint(self):
        checkpoint = {'epoch': self.total_epoches,
                      'model_state_dict': self.model.state_dict(),
                      'optimizer_state_dict': self.optimizer.state_dict(),
                      'loss': self.losses,
                      'accuracy': self.accuracy,
                      'val_loss': self.val_losses,
                      'val_accuracy': self.val_accuracy
                      }
        torch.save(checkpoint, self.config.nlp_trained_model_path)
        
    def load_checkpoint(self):
        checkpoint = torch.load(self.config.nlp_trained_model_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_epoches = checkpoint["epoch"]
        self.losses = checkpoint["loss"]
        self.accuracy = checkpoint['accuracy']
        self.val_accuracy = checkpoint['val_accuracy']
        self.val_losses = checkpoint["val_loss"]
        self.model.train() # always use train for resuming traning
    
    
    # HF’s Dataset to Tokenized TensorDataset
    def _tokenize_dataset(self, hf_dataset, sentence_field,
        label_field, tokenizer, **kwargs):
        sentences = hf_dataset[sentence_field]
        token_ids = tokenizer(
        sentences, return_tensors='pt', **kwargs
        )['input_ids']
        labels = torch.as_tensor(hf_dataset[label_field])
        dataset = TensorDataset(token_ids, labels)
        return dataset
    
    
    def predict(self, text):
        self.load_checkpoint()
        self.model.eval()
        auto_tokenizer = AutoTokenizer.from_pretrained(self.config.params_model_name)
        tokenizer_kwargs = dict(truncation=True,
                                padding=True,
                                max_length=30,
                                add_special_tokens=True)
        tokenize_text = self._tokenize_dataset(text,
                                                'sentence',
                                                'labels',
                                                auto_tokenizer,
                                                **tokenizer_kwargs)
        x_tensor = torch.as_tensor(tokenize_text).float()
        y_hat_tensor = self.model(x_tensor.to(self.device))
        
        # set it back to the train mode
        self.model.train()
        labels = {0: 'Medical Necessity', 1: 'Experimental/Investigational', 2: 'Urgent Care'}
        
        return labels[np.argmax(y_hat_tensor.detach().cpu().numpy())]
    
    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics({'train_loss': np.mean(self.losses),'val_loss': np.mean(self.val_losses), 'train_accuracy': np.mean(self.accuracy), 'val_accuracy': np.mean(self.val_accuracy)})
        
            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.pytorch.log_model(self.model, "model", registered_model_name="nlp18Model")
            else:
                mlflow.pytorch.log_model(self.model, "model")
                
                
class BERTClassifier(nn.Module):
    def __init__(self, bert_model, ff_units, n_outputs, dropout=0.3):
        super().__init__()
        self.d_model = bert_model.config.dim
        self.n_outputs = n_outputs
        self.encoder = bert_model
        self.mlp = nn.Sequential(
            nn.Linear(self.d_model, ff_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_units, n_outputs)
        )
    def encode(self, source, source_mask=None):
        states = self.encoder(
        input_ids=source, attention_mask=source_mask)[0]
        cls_state = states[:, 0]
        return cls_state
    def forward(self, X):
        source_mask = (X > 0)
        # Featurizer
        cls_state = self.encode(X, source_mask)
        # Classifier
        out = self.mlp(cls_state)
        return out