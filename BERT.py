import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoFeatureExtractor, Trainer, TrainingArguments, TrainerCallback, AdamW
from datasets import load_dataset, DatasetDict, Audio, Value
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import torch.nn.functional as F 
import copy
import wandb,random
from torch.utils.data import DataLoader
from datetime import datetime

import os
model_name = "BERT"

# seed = 3
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# Load the dataset
train_path = '/home/kding@uni.federation.edu.au/ADReSS/Train'
test_path = '/home/kding@uni.federation.edu.au/ADReSS/Test'

train_ADReSSo_path = "/home/kding@uni.federation.edu.au/ADReSSo/Train"
test_ADReSSo_path = "/home/kding@uni.federation.edu.au/ADReSSo/Test"

# Initialize tokenizers and feature extractors
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-small")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def print_and_save(message, file_path=f"{model_name}.txt"):
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Format the message with timestamp
    message_with_timestamp = f"{timestamp} - {message}"
    
    # Print to console
    print(message_with_timestamp)
    
    # Save to file
    with open(file_path, "a") as file:
        file.write(message_with_timestamp + "\n")

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        normalized = self.norm(x)
        sublayer_output = sublayer(normalized)
        
        if isinstance(sublayer_output, tuple):
            sublayer_result, extra_output = sublayer_output
            return x + self.dropout(sublayer_result), extra_output
        else:
            return x + self.dropout(sublayer_output)

class LayerNorm(nn.Module):
    "Construct a layernorm module"

    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model=768, num_heads=12, batch_first=True, dropout=0.1, option=True):
        super().__init__()
        self.option = option  # Store the option flag
        self.cross_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=batch_first)
        
        # Conditionally initialize the feed forward layer and sublayer connection based on the option flag
        if option:
            self.feed_forward = PositionwiseFeedForward(d_model, d_ff=3072)
            self.sublayer = clones(SublayerConnection(d_model, dropout), 2)
        else:
            self.sublayer = clones(SublayerConnection(d_model, dropout), 1)

        self.d_model = d_model

    def forward(self, x, y):
        def apply_cross_attention(x, y):
            attn_output, attn_weights = self.cross_attention(x, y, y)
            return attn_output, attn_weights

        # Apply cross-attention and get the attention weights
        attn_output, attn_weights = apply_cross_attention(x, y)
        x = self.sublayer[0](x, lambda x: apply_cross_attention(x, y)[0])
        
        # Conditionally apply the second sublayer and feed-forward if option is True
        if self.option:
            x = self.sublayer[1](x, self.feed_forward)

        return x, attn_weights

class AudioTextClassifier(nn.Module):
    def __init__(self, num_labels=2, bert_model="bert-base-uncased", dropout=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model, output_hidden_states=True)
        self.embed_dim = self.bert.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.LayerNorm(self.embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(self.embed_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.num_labels = num_labels
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        # Enable gradient checkpointing for the transformer models
        self.bert.gradient_checkpointing_enable()

    def forward(self, text_input_ids, text_attention_mask,labels=None):

        text_output = self.bert(input_ids=text_input_ids, attention_mask=text_attention_mask)
        text_last_hidden_state = text_output.last_hidden_state
        text_pooled = text_last_hidden_state.mean(dim=1)
        logits = self.classifier(text_pooled)
        return logits # Outputs are probabilities after sigmoid

def preprocess_function(examples):
    # Tokenize text inputs
    text_inputs = tokenizer(
        examples["transcript"],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    # Convert labels to a list of integers
    labels = [int(label) for label in examples["label"]]

    # Ensure that everything is batched and returned in the correct format
    return {
        "labels": labels,
        "text_input_ids": text_inputs.input_ids,
        "text_attention_mask": text_inputs.attention_mask,
        'idx': examples["id"]
    }

# Define compute metrics function
def compute_metrics(eval_pred): 
    logits = eval_pred.predictions
    predictions = (logits >= 0.5).astype(int)  # Apply threshold
    labels = eval_pred.label_ids
 
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    tn, fp, fn, tp = cm.ravel()  # For binary classification, ravel returns [TN, FP, FN, TP]
    
    # Calculate metrics
    precision = precision_score(labels, predictions, pos_label=1)  # AD (class 1) as positive class
    recall = recall_score(labels, predictions, pos_label=1)        # AD (class 1) as positive class
    f1 = f1_score(labels, predictions, pos_label=1)                # AD (class 1) as positive class
    accuracy = accuracy_score(labels, predictions)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # TN / (TN + FP)

    precisions, recalls, f1s, _ = precision_recall_fscore_support(labels, predictions, average=None, labels=[0, 1])
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "specificity": specificity,
        "HC_precision": precisions[0],  # Precision for HC (class 0)
        "AD_precision": precisions[1],  # Precision for AD (class 1)
        "HC_recall": recalls[0],        # Recall for HC (class 0)
        "AD_recall": recalls[1],        # Recall for AD (class 1)
        "HC_f1": f1s[0],                # F1-score for HC (class 0)
        "AD_f1": f1s[1]                 # F1-score for AD (class 1)
    }

# Custom data collator
class MultimodalDataCollator:
    def __call__(self, features):
        batch = {
            "text_input_ids": torch.tensor(np.array([f["text_input_ids"] for f in features]), dtype=torch.long),
            "text_attention_mask": torch.tensor(np.array([f["text_attention_mask"] for f in features]), dtype=torch.float32),
            "labels": torch.tensor([f["labels"] for f in features], dtype=torch.long),
            "idx": [f["idx"] for f in features],
        }
        
        return batch
       
class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.001, num_classes=2):
        """
        Precise Label Smoothing Loss
        
        Args:
            smoothing (float): Smoothing parameter α
            num_classes (int): Number of classes K
        """
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.confidence = 1.0 - smoothing
        self.smoothing_value = smoothing / (num_classes - 1)

    def forward(self, inputs, targets):
        """
        Apply precise label smoothing
        
        Args:
            inputs (torch.Tensor): Model predictions (logits or probabilities)
            targets (torch.Tensor): Binary ground truth labels
        
        Returns:
            torch.Tensor: Label smoothed loss
        """
        # For binary classification, we'll adapt the multi-class formula
        # Reshape targets to ensure correct dimensionality
        targets = targets.view(-1, 1)
        
        # Create smoothed targets according to the formula:
        # yLSu_k = y_k * (1 - α) + α / K
        smoothed_targets = targets * self.confidence + self.smoothing_value
        
        # Compute cross-entropy loss with smoothed targets
        loss_fct = nn.BCELoss()
        loss = loss_fct(inputs.view(-1), smoothed_targets.view(-1))
        
        return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Focal Loss implementation
        
        Args:
            alpha (torch.Tensor, optional): Weight for each class. Useful for class imbalance
            gamma (float): Focusing parameter. Higher gamma means more focus on hard examples
            reduction (str): 'mean', 'sum' or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)  # Probabilities for the correct class
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Define custom trainer
class MultimodalTrainer(Trainer):
    def __init__(self, train_dataset=None, alpha=None, gamma=2.0, seed=42, label_smoothing=0.1, num_classes=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_dataset = train_dataset

        if alpha is not None:
            self.alpha = torch.tensor(alpha).to(self.args.device)
        else:
            self.alpha = None
        self.focal_loss = FocalLoss(alpha=self.alpha, gamma=gamma, reduction='mean')
        self.seed = seed
        self.epoch = 0  # Add epoch counter

        self.label_smoothing = label_smoothing
        self.label_smooth_loss = LabelSmoothingLoss(
            smoothing=label_smoothing, 
            num_classes=num_classes
        )

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        # Create a new random seed for each epoch
        epoch_seed = self.seed + self.epoch
        
        # Create a RandomSampler with the epoch-specific seed
        train_sampler = torch.utils.data.RandomSampler(
            self.train_dataset,
            generator=torch.Generator().manual_seed(epoch_seed)
        )
        
        # Create and return the DataLoader with the sampler
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            #sampler=train_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
        
    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = self.data_collator
        
        return DataLoader(
            eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
    
    def compute_loss(self, model, inputs, return_outputs=False):

        labels = inputs["labels"].float()  # Ensure labels are float for BCELoss    
        text_input_ids = inputs["text_input_ids"]
        text_attention_mask = inputs["text_attention_mask"]
        idx = inputs["idx"]
        
        outputs = model(text_input_ids, text_attention_mask)
        predictions = (outputs >= 0.5).long().view(-1)

        print(f"idx: {idx}")
        print(f"labels: {labels}")
        print(f"outputs: {outputs.view(-1)}")
        print(f"predictions: {predictions}")

        loss_fct = nn.BCELoss()
        loss = loss_fct(outputs.view(-1), labels)
        
        return (loss, outputs) if return_outputs else loss
        
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        labels = inputs["labels"].float()
        with torch.no_grad():
            loss, logits = self.compute_loss(model, inputs, return_outputs=True)
            preds = (logits >= 0.5).long()  # Apply threshold for binary predictions

        return (loss, preds, labels)

    def training_step(self, model, inputs):
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        
        if loss is None:
            raise ValueError("Loss is None. Check model outputs and loss calculation.")
        
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        
        loss.backward()

        # print(f" training_step loss: {loss}")
        return loss.detach()

class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, early_stopping_patience: int = 3):
        self.early_stopping_patience = early_stopping_patience
        self.best_loss = float('inf')
        self.no_improvement_count = 0

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if metrics['eval_loss'] < self.best_loss:
            self.best_loss = metrics['eval_loss']
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.early_stopping_patience:
                control.should_training_stop = True

wandb.login()

# Define different seeds for each run
seeds = [42, 2023, 2024, 88, 1234]
test_results_list = []

def objective(seed):
    # Load and shuffle the dataset with the current seed
    AD_dataset_train = load_dataset("audiofolder", data_dir=train_path, split="all").shuffle(seed=seed)
    AD_dataset_test = load_dataset("audiofolder", data_dir=test_path, split="all")

    # Split the dataset into train and validation (65% train, 35% val)
    train_val_split = AD_dataset_train.train_test_split(test_size=0.35)
    train_dataset = train_val_split['train']
    valid_dataset = train_val_split['test']
    AD_dataset = DatasetDict({"train": train_dataset, "eval": valid_dataset, "test": AD_dataset_test})

    # Cast audio column
    AD_dataset = AD_dataset.cast_column("audio", Audio(sampling_rate=feature_extractor.sampling_rate))

    # Preprocess dataset
    AD_dataset_encoded = AD_dataset.map(
        preprocess_function,
        batched=True,
        batch_size=200,
        remove_columns=AD_dataset["train"].column_names,
        num_proc=1,
    )

    # Initialize model and trainer
    model = AudioTextClassifier(dropout=0.1)
    data_collator = MultimodalDataCollator()

    training_args = TrainingArguments(
        run_name=model_name,
        output_dir=model_name,
        overwrite_output_dir=True,
        eval_strategy="steps",
        save_strategy="steps",
        learning_rate=2e-05,
        gradient_checkpointing=True,
        save_steps=1,              
        eval_steps=1,              
        save_total_limit=1, 
        max_grad_norm=1,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=8,
        num_train_epochs=50,
        logging_steps=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=seed,
        report_to="wandb"
    )

    trainer = MultimodalTrainer(
        model=model,
        args=training_args,
        train_dataset=AD_dataset_encoded["train"],
        eval_dataset=AD_dataset_encoded["eval"],
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=20)],
    )

    # Train the model
    trainer.train()

    # Evaluate on the Vallidation set
    val_results = trainer.evaluate(AD_dataset_encoded["eval"])
    print_and_save(f"Vallidation Result with seed {seed}: {val_results}")

    # Evaluate on the test set
    test_results = trainer.evaluate(AD_dataset_encoded["test"])
    print_and_save(f"Test Result with seed {seed}: {test_results}")

    wandb.finish()

    return test_results

# Run the experiment for each seed and store results
for seed in seeds:
    test_results = objective(seed)
    test_results_list.append(test_results)  # Assuming "eval_loss" is the metric of interest

avg_test_results = {key: np.mean([res[key] for res in test_results_list]) for key in test_results_list[0].keys()}
print_and_save(f"Average test results across all runs: {avg_test_results}")

std_test_results = {key: np.std([res[key] for res in test_results_list]) for key in test_results_list[0].keys()}
print_and_save(f"Std test results across all runs: {std_test_results}")