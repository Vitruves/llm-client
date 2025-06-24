#!/usr/bin/env python3

import os
import json
import argparse
import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

# CUDA compatibility settings - must be set before importing torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
# Disable expandable_segments to avoid compatibility issues
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
# Reduce tokenizer spam during multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import multiprocessing as mp
from collections import Counter
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

# Set multiprocessing start method and get CPU count
mp.set_start_method("spawn", force=True)
CPU_COUNT = mp.cpu_count()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Reduce spam from datasets library
logging.getLogger("datasets").setLevel(logging.WARNING)

# Optimize for CPU parallelism
os.environ["OMP_NUM_THREADS"] = str(CPU_COUNT)
os.environ["MKL_NUM_THREADS"] = str(CPU_COUNT)
os.environ["NUMEXPR_NUM_THREADS"] = str(CPU_COUNT)

# Set PyTorch thread count
torch.set_num_threads(CPU_COUNT)


class EarlyStoppingCallback:
    """Custom early stopping callback for training optimization"""

    def __init__(
        self,
        patience: int = 3,
        min_delta: float = 0.001,
        restore_best_weights: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float("inf")
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None

    def __call__(self, eval_loss: float, model, epoch: int):
        """Check if training should stop early"""
        if eval_loss < self.best_loss - self.min_delta:
            self.best_loss = eval_loss
            self.wait = 0
            if self.restore_best_weights:
                # Store best model state
                self.best_weights = {
                    k: v.clone() for k, v in model.state_dict().items()
                }
        else:
            self.wait += 1

        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            logger.info(
                f"Early stopping triggered at epoch {epoch} (patience: {self.patience})"
            )
            if self.restore_best_weights and self.best_weights:
                logger.info("Restoring best model weights")
                model.load_state_dict(self.best_weights)
            return True
        return False


class ParallelMetricsComputer:
    """Compute metrics using parallel processing"""

    @staticmethod
    def compute_metrics_chunk(args):
        """Compute metrics for a chunk of predictions"""
        pred_chunk, true_chunk = args
        if len(pred_chunk) == 0:
            return {"accuracy": 0.0, "f1": 0.0, "samples": 0}

        accuracy = accuracy_score(true_chunk, pred_chunk)
        f1 = f1_score(true_chunk, pred_chunk, average="macro", zero_division=0)

        return {"accuracy": accuracy, "f1": f1, "samples": len(pred_chunk)}

    @staticmethod
    def parallel_compute_metrics(pred_labels, true_labels, num_processes=None):
        """Compute metrics using parallel processing"""
        if num_processes is None:
            num_processes = min(CPU_COUNT, 32)  # Cap at 8 for efficiency

        if len(pred_labels) < 1000 or num_processes == 1:
            # Use single process for small datasets
            return ParallelMetricsComputer.compute_metrics_chunk(
                (pred_labels, true_labels)
            )

        # Split data into chunks
        chunk_size = max(100, len(pred_labels) // num_processes)
        chunks = []

        for i in range(0, len(pred_labels), chunk_size):
            pred_chunk = pred_labels[i : i + chunk_size]
            true_chunk = true_labels[i : i + chunk_size]
            chunks.append((pred_chunk, true_chunk))

        # Process chunks in parallel
        with mp.Pool(processes=num_processes) as pool:
            results = pool.map(ParallelMetricsComputer.compute_metrics_chunk, chunks)

        # Aggregate results
        total_samples = sum(r["samples"] for r in results)
        if total_samples == 0:
            return {"accuracy": 0.0, "f1": 0.0, "samples": 0}

        # Weighted average by sample count
        weighted_accuracy = (
            sum(r["accuracy"] * r["samples"] for r in results) / total_samples
        )
        weighted_f1 = sum(r["f1"] * r["samples"] for r in results) / total_samples

        return {
            "accuracy": weighted_accuracy,
            "f1": weighted_f1,
            "samples": total_samples,
        }


class UnslothFineTuner:
    def __init__(self, output_dir: str, label_mapping: Optional[Dict[str, str]] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.label_mapping = label_mapping or {}
        self.cpu_count = CPU_COUNT
        logger.info(f"Initialized with {self.cpu_count} CPU cores available")

    def load_excel_data(
        self,
        excel_path: str,
        test_col: str,
        control_col: str,
        max_examples: int = None,
        sampling_strategy: str = "balanced",
        eval_split: float = 0.1,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        logger.info(f"Loading data from {excel_path}")

        try:
            df = pd.read_excel(excel_path)
        except Exception as e:
            logger.error(f"Failed to load Excel file: {e}")
            raise

        if test_col not in df.columns or control_col not in df.columns:
            available_cols = list(df.columns)
            raise ValueError(
                f"Columns '{test_col}' or '{control_col}' not found. Available columns: {available_cols}"
            )

        original_size = len(df)
        df = df[[test_col, control_col]].dropna()
        logger.info(f"Removed {original_size - len(df)} rows with missing values")

        if self.label_mapping:
            df[control_col] = (
                df[control_col].map(self.label_mapping).fillna(df[control_col])
            )
            logger.info(f"Applied label mapping to {control_col} column")

        self._display_data_stats(df, test_col, control_col)

        if max_examples and len(df) > max_examples:
            df = self._sample_data(df, control_col, max_examples, sampling_strategy)

        # Split into train and eval
        if eval_split > 0:
            eval_size = int(len(df) * eval_split)
            df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
            eval_df = df_shuffled[:eval_size]
            train_df = df_shuffled[eval_size:]
            logger.info(
                f"Split into {len(train_df)} training and {len(eval_df)} evaluation samples"
            )
            return train_df, eval_df
        else:
            logger.info("No evaluation split created")
            return df, None

    def _display_data_stats(self, df: pd.DataFrame, test_col: str, control_col: str):
        logger.info(f"Dataset Statistics:")
        logger.info(f"  Total samples: {len(df)}")
        logger.info(f"  Unique labels: {df[control_col].nunique()}")

        label_counts = df[control_col].value_counts()
        logger.info(f"  Label distribution:")
        for label, count in label_counts.head(10).items():
            logger.info(f"    {label}: {count} ({count/len(df)*100:.1f}%)")

        if len(label_counts) > 10:
            logger.info(f"    ... and {len(label_counts) - 10} more labels")

        text_lengths = df[test_col].str.len()
        logger.info(f"  Text length stats:")
        logger.info(f"    Mean: {text_lengths.mean():.1f} chars")
        logger.info(f"    Median: {text_lengths.median():.1f} chars")
        logger.info(f"    Min: {text_lengths.min()} chars")
        logger.info(f"    Max: {text_lengths.max()} chars")

    def _sample_data(
        self, df: pd.DataFrame, control_col: str, max_examples: int, strategy: str
    ) -> pd.DataFrame:
        logger.info(f"Sampling {max_examples} examples using {strategy} strategy")

        if strategy == "balanced":
            label_counts = df[control_col].value_counts()
            samples_per_label = max(1, max_examples // len(label_counts))

            sampled_dfs = []
            for label in label_counts.index:
                label_df = df[df[control_col] == label]
                n_samples = min(len(label_df), samples_per_label)
                sampled_dfs.append(label_df.sample(n=n_samples, random_state=42))

            df = pd.concat(sampled_dfs, ignore_index=True)

            if len(df) > max_examples:
                df = df.sample(n=max_examples, random_state=42)

        elif strategy == "proportional":
            label_counts = df[control_col].value_counts()
            sampled_dfs = []

            for label, count in label_counts.items():
                proportion = count / len(df)
                n_samples = max(1, int(max_examples * proportion))
                label_df = df[df[control_col] == label]
                n_samples = min(n_samples, len(label_df))
                sampled_dfs.append(label_df.sample(n=n_samples, random_state=42))

            df = pd.concat(sampled_dfs, ignore_index=True)

        elif strategy == "random":
            df = df.sample(n=min(max_examples, len(df)), random_state=42)

        logger.info(f"Sampled dataset size: {len(df)}")
        return df

    def compute_class_weights(
        self,
        df: pd.DataFrame,
        control_col: str,
        class_weights: Union[str, List[float], Dict[str, float]] = "balanced",
    ) -> torch.Tensor:
        """
        Compute class weights for handling imbalanced datasets

        Args:
            df: DataFrame containing the data
            control_col: Name of the target column
            class_weights: Can be:
                - "balanced": Automatically compute inverse frequency weights
                - List[float]: Manual weights (e.g., [1.0, 0.1, 0.03])
                - Dict[str, float]: Manual weights mapped to class names
                - None: No weighting (uniform weights)

        Returns:
            torch.Tensor: Class weights tensor
        """
        unique_labels = sorted(df[control_col].unique())
        num_classes = len(unique_labels)

        if class_weights is None or class_weights == "none":
            weights = torch.ones(num_classes)
            logger.info("Using uniform class weights (no weighting)")

        elif class_weights == "balanced":
            # Compute balanced weights (inverse frequency)
            label_counts = df[control_col].value_counts()
            weights_dict = {}
            total_samples = len(df)

            for label in unique_labels:
                count = label_counts.get(label, 1)
                # Inverse frequency: total_samples / (num_classes * count)
                weight = total_samples / (num_classes * count)
                weights_dict[label] = weight

            weights = torch.tensor(
                [weights_dict[label] for label in unique_labels], dtype=torch.float32
            )
            logger.info("Computed balanced class weights (inverse frequency)")

        elif isinstance(class_weights, list):
            if len(class_weights) != num_classes:
                raise ValueError(
                    f"Number of provided weights ({len(class_weights)}) must match number of classes ({num_classes})"
                )
            weights = torch.tensor(class_weights, dtype=torch.float32)
            logger.info(f"Using manual class weights: {class_weights}")

        elif isinstance(class_weights, dict):
            weights_list = []
            for label in unique_labels:
                if label not in class_weights:
                    raise ValueError(
                        f"Weight for class '{label}' not provided in class_weights dictionary"
                    )
                weights_list.append(class_weights[label])
            weights = torch.tensor(weights_list, dtype=torch.float32)
            logger.info(f"Using manual class weights from dictionary: {class_weights}")

        else:
            raise ValueError(f"Unsupported class_weights type: {type(class_weights)}")

        # Log weight information
        logger.info("Class weight details:")
        for i, (label, weight) in enumerate(zip(unique_labels, weights)):
            count = df[control_col].value_counts().get(label, 0)
            percentage = (count / len(df)) * 100
            logger.info(
                f"  {label}: weight={weight:.4f}, samples={count} ({percentage:.1f}%)"
            )

        return weights

    def load_prompt_template(self, prompt_path: str) -> str:
        logger.info(f"Loading prompt template from {prompt_path}")
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            logger.error(f"Failed to load prompt template: {e}")
            raise

    def create_training_data(
        self, df: pd.DataFrame, prompt_template: str, test_col: str, control_col: str
    ) -> List[Dict[str, str]]:
        training_data = []

        for _, row in df.iterrows():
            user_content = prompt_template.format(
                text=row[test_col], control=row[control_col]
            )

            conversation = {
                "instruction": user_content,
                "output": str(row[control_col]),
            }
            training_data.append(conversation)

        logger.info(f"Created {len(training_data)} training examples")
        return training_data

    def compute_metrics(self, eval_preds):
        """Compute classification metrics for evaluation using parallel processing"""
        predictions, labels = eval_preds

        # Get predictions (argmax of logits)
        if len(predictions.shape) > 1:
            predictions = np.argmax(predictions, axis=-1)

        # Convert to string labels for easier interpretation
        pred_labels = [str(p) for p in predictions.flatten()]
        true_labels = [str(l) for l in labels.flatten()]

        # Filter out special tokens (typically -100)
        valid_indices = [i for i, label in enumerate(true_labels) if label != "-100"]
        pred_labels = [pred_labels[i] for i in valid_indices]
        true_labels = [true_labels[i] for i in valid_indices]

        if len(pred_labels) == 0:
            return {"accuracy": 0.0, "f1": 0.0}

        # Use parallel computation for metrics
        logger.info(
            f"Computing metrics for {len(pred_labels)} samples using {self.cpu_count} CPU cores"
        )

        result = ParallelMetricsComputer.parallel_compute_metrics(
            pred_labels, true_labels, num_processes=min(self.cpu_count, 32)
        )

        return {
            "accuracy": result["accuracy"],
            "f1_score": result["f1"],
            "eval_samples": result["samples"],
        }

    def _check_local_model_path(self, model_path: str) -> bool:
        """Check if local model path exists and contains required files"""
        path = Path(model_path)
        if not path.exists():
            return False

        # Check for common model files
        required_files = ["config.json"]
        has_model_file = any(
            [
                (path / "pytorch_model.bin").exists(),
                (path / "model.safetensors").exists(),
                any(path.glob("pytorch_model-*.bin")),
                any(path.glob("model-*.safetensors")),
            ]
        )

        has_config = (path / "config.json").exists()

        return has_config and has_model_file

    def fine_tune(
        self,
        training_data: List[Dict],
        train_df: pd.DataFrame,
        eval_data: Optional[List[Dict]] = None,
        epochs: int = 3,
        learning_rate: float = 2e-5,
        batch_size: int = 16,
        model_name: str = None,
        model_size: str = "1.5b",
        fast_training: bool = False,
        dropout_rate: float = 0.1,
        early_stopping_patience: int = 3,
        min_delta: float = 0.001,
        local_model_path: str = None,
        class_weights: Union[str, List[float], Dict[str, float]] = None,
        control_col: str = "DRUG",
    ):
        logger.info("Starting Unsloth fine-tuning...")
        logger.info(
            f"CPU optimization: Using {self.cpu_count} cores for parallel processing"
        )

        from unsloth import FastLanguageModel
        from datasets import Dataset
        from trl import SFTTrainer
        from transformers import TrainingArguments

        # Configure device placement for 4-bit quantized models
        # For 4-bit models, we need to use a single device to avoid training conflicts
        current_device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        device_map = {"": current_device} if torch.cuda.is_available() else "auto"
        
        if torch.cuda.device_count() > 1:
            logger.info(f"Multiple GPUs detected ({torch.cuda.device_count()}), but using single GPU ({current_device}) for 4-bit quantized model compatibility")
        else:
            logger.info(f"Using device: {current_device}")

        # Model selection logic with local model support
        if local_model_path:
            if not self._check_local_model_path(local_model_path):
                raise ValueError(
                    f"Local model path '{local_model_path}' does not exist or is missing required files"
                )
            model_to_use = str(Path(local_model_path).resolve())
            logger.info(f"Using local model: {model_to_use}")
        elif model_name:
            model_to_use = model_name
            logger.info(f"Using custom model: {model_name}")
        else:
            model_map = {
                "0.5b": "unsloth/qwen2.5-0.5b-instruct-bnb-4bit",
                "1.5b": "unsloth/qwen2.5-1.5b-instruct-bnb-4bit",
                "3b": "unsloth/qwen2.5-3b-instruct-bnb-4bit",
                "7b": "unsloth/qwen2.5-7b-instruct-bnb-4bit",
                "14b": "unsloth/qwen2.5-14b-instruct-bnb-4bit",
            }
            model_to_use = model_map.get(model_size, model_map["1.5b"])
            logger.info(f"Using preset model: {model_to_use}")

        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_to_use,
                max_seq_length=1024,
                dtype=None,
                load_in_4bit=True,
                device_map=device_map,
            )
        except Exception as e:
            logger.error(f"Failed to load model {model_to_use}: {e}")
            raise

        # Apply LoRA with configurable dropout
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=16,
            lora_dropout=dropout_rate,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )

        logger.info(f"Model loaded and LoRA applied with dropout rate: {dropout_rate}")

        # Compute class weights
        weights_tensor = None
        if class_weights is not None:
            weights_tensor = self.compute_class_weights(
                train_df, control_col, class_weights
            )
            if torch.cuda.is_available():
                weights_tensor = weights_tensor.to(current_device)

        def formatting_prompts_func(examples):
            instructions = examples["instruction"]
            outputs = examples["output"]
            texts = []
            for instruction, output in zip(instructions, outputs):
                text = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
                texts.append(text)
            return {"text": texts}

        # Use all CPU cores for training dataset
        dataset = Dataset.from_list(training_data)
        dataset = dataset.map(
            formatting_prompts_func,
            batched=True,
            num_proc=self.cpu_count,
        )

        eval_dataset = None
        early_stopping_callback = None

        if eval_data and not fast_training:
            eval_dataset = Dataset.from_list(eval_data)
            # Use fewer cores for evaluation to reduce spam and improve performance
            eval_proc_count = min(8, self.cpu_count)
            eval_dataset = eval_dataset.map(
                formatting_prompts_func,
                batched=True,
                num_proc=eval_proc_count,
            )
            logger.info(f"Created evaluation dataset with {len(eval_dataset)} examples")

            # Initialize early stopping callback
            early_stopping_callback = EarlyStoppingCallback(
                patience=early_stopping_patience,
                min_delta=min_delta,
                restore_best_weights=True,
            )
            logger.info(
                f"Early stopping enabled: patience={early_stopping_patience}, min_delta={min_delta}"
            )

        elif fast_training:
            logger.info("Fast training mode: Skipping evaluation for maximum speed")

        logger.info(f"Created training dataset with {len(dataset)} examples")

        total_steps = (len(dataset) // batch_size) * epochs
        warmup_steps = max(10, total_steps // 10)

        # Optimize evaluation settings
        if eval_dataset and not fast_training:
            eval_batch_size = min(
                32, len(eval_dataset)
            )  # Increase batch size for faster evaluation
            eval_steps = max(100, total_steps // 10)  # Reduce evaluation frequency
        else:
            eval_batch_size = 4
            eval_steps = None

        logger.info(f"Training configuration:")
        logger.info(
            f"  Training mode: {'FAST (no evaluation)' if fast_training else 'Standard with Early Stopping'}"
        )
        logger.info(f"  CPU cores available: {self.cpu_count}")
        logger.info(f"  Epochs: {epochs}")
        logger.info(f"  Training batch size: {batch_size}")
        logger.info(
            f"  Evaluation batch size: {eval_batch_size if eval_dataset else 'N/A'}"
        )
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  Dropout rate: {dropout_rate}")
        logger.info(
            f"  Class weighting: {'Enabled' if weights_tensor is not None else 'Disabled'}"
        )
        logger.info(f"  Total steps: {total_steps}")
        logger.info(f"  Warmup steps: {warmup_steps}")
        if eval_steps:
            logger.info(f"  Evaluation steps: {eval_steps}")
        if early_stopping_callback:
            logger.info(f"  Early stopping patience: {early_stopping_patience}")

        training_args = TrainingArguments(
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=eval_batch_size,
            gradient_accumulation_steps=1,
            warmup_steps=warmup_steps,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            fp16=False,
            bf16=True,
            logging_steps=max(1, total_steps // 50),
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=42,
            output_dir=str(self.output_dir),
            # Reduce CPU utilization for evaluation to speed it up
            dataloader_num_workers=4,  # Drastically reduce workers for faster evaluation
            dataloader_pin_memory=False,  # Disable pin memory for evaluation
            remove_unused_columns=False,
            save_strategy="steps" if eval_dataset else "epoch",
            save_steps=eval_steps if eval_dataset else 100,
            report_to=None,
            ddp_find_unused_parameters=False,
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=eval_steps,
            eval_delay=0,
            load_best_model_at_end=False,  # Disable for speed
            metric_for_best_model="eval_loss" if eval_dataset else None,
            greater_is_better=False,
            save_total_limit=2,  # Reduce saves
            eval_accumulation_steps=4,  # Accumulate evaluation batches
            dataloader_prefetch_factor=1,  # Minimal prefetch
            prediction_loss_only=True,  # Only compute loss, no metrics
        )

        # Disable compute_metrics for speed - only use loss
        compute_metrics_func = None  # Disable to speed up evaluation

        # Custom trainer class for early stopping and weighted loss
        class WeightedEarlyStoppingTrainer(SFTTrainer):
            def __init__(
                self, early_stopping_callback=None, class_weights=None, **kwargs
            ):
                super().__init__(**kwargs)
                self.early_stopping_callback = early_stopping_callback
                self.class_weights = class_weights

            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                """Compute weighted loss for imbalanced classes"""
                # Accept additional kwargs that Unsloth may pass (like num_items_in_batch)
                labels = inputs.get("labels")
                outputs = model(**inputs)

                if labels is not None and self.class_weights is not None:
                    logits = outputs.get("logits")
                    if logits is not None:
                        # Reshape for cross entropy
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = labels[..., 1:].contiguous()

                        # Flatten the tokens
                        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                        shift_labels = shift_labels.view(-1)

                        # Filter out -100 labels (padding)
                        valid_indices = shift_labels != -100
                        if valid_indices.any():
                            valid_logits = shift_logits[valid_indices]
                            valid_labels = shift_labels[valid_indices]

                            # Apply class weights to cross entropy loss
                            loss_fct = torch.nn.CrossEntropyLoss(
                                weight=self.class_weights, ignore_index=-100
                            )
                            loss = loss_fct(valid_logits, valid_labels)
                            outputs.loss = loss

                return (outputs.loss, outputs) if return_outputs else outputs.loss

            def evaluate(
                self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"
            ):
                # Call parent evaluate method with error handling for linter compatibility
                try:
                    eval_result = super().evaluate(
                        eval_dataset, ignore_keys, metric_key_prefix
                    )
                except AttributeError:
                    # Fallback if super().evaluate doesn't exist (should not happen in practice)
                    eval_result = {"eval_loss": float("inf")}
                    logger.warning("Parent evaluate method not found, using fallback")

                # Check early stopping condition
                if self.early_stopping_callback and "eval_loss" in eval_result:
                    should_stop = self.early_stopping_callback(
                        eval_result["eval_loss"], self.model, self.state.epoch
                    )
                    if should_stop:
                        logger.info("Early stopping condition met. Stopping training.")
                        self.control.should_training_stop = True

                return eval_result

        trainer = WeightedEarlyStoppingTrainer(
            early_stopping_callback=early_stopping_callback,
            class_weights=weights_tensor,
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            eval_dataset=eval_dataset,
            dataset_text_field="text",
            max_seq_length=1024,
            dataset_num_proc=self.cpu_count,  # Use all CPU cores for dataset processing
            packing=True,
            args=training_args,
            compute_metrics=compute_metrics_func,
        )

        logger.info("Starting training...")
        train_result = trainer.train()

        # Log early stopping info if applicable
        if early_stopping_callback and early_stopping_callback.stopped_epoch > 0:
            logger.info(
                f"Training stopped early at epoch {early_stopping_callback.stopped_epoch}"
            )
            logger.info(
                f"Best validation loss: {early_stopping_callback.best_loss:.4f}"
            )

        logger.info("Saving model...")
        model.save_pretrained(str(self.output_dir / "lora_model"))
        tokenizer.save_pretrained(str(self.output_dir / "lora_model"))

        logger.info("Converting to GGUF...")
        model.save_pretrained_gguf(
            str(self.output_dir), tokenizer, quantization_method="q6_k"
        )

        logger.info("Training completed successfully!")

        # Return training info including early stopping details
        training_info = {
            "train_runtime": train_result.metrics.get("train_runtime", 0),
            "train_samples_per_second": train_result.metrics.get(
                "train_samples_per_second", 0
            ),
            "early_stopped": early_stopping_callback
            and early_stopping_callback.stopped_epoch > 0,
            "stopped_epoch": (
                early_stopping_callback.stopped_epoch
                if early_stopping_callback
                else None
            ),
            "best_eval_loss": (
                early_stopping_callback.best_loss if early_stopping_callback else None
            ),
            "class_weights_used": (
                weights_tensor.tolist() if weights_tensor is not None else None
            ),
            "cpu_cores_used": self.cpu_count,
        }

        return training_info

    def save_training_metadata(
        self,
        args,
        df: pd.DataFrame,
        training_data: List[Dict],
        eval_data: Optional[List[Dict]] = None,
        training_info: Dict = None,
    ):
        metadata = {
            "system_info": {
                "cpu_cores": self.cpu_count,
                "torch_threads": torch.get_num_threads(),
            },
            "dataset_info": {
                "total_samples": len(df),
                "training_samples": len(training_data),
                "eval_samples": len(eval_data) if eval_data else 0,
                "unique_labels": df[args.control_col].nunique(),
                "label_distribution": df[args.control_col].value_counts().to_dict(),
            },
            "training_config": {
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "model": args.model
                or args.local_model_path
                or f"qwen2.5-{args.model_size}",
                "max_examples": args.max_examples,
                "sampling_strategy": args.sampling_strategy,
                "eval_split": args.eval_split,
                "dropout_rate": args.dropout_rate,
                "early_stopping_patience": args.early_stopping_patience,
                "min_delta": args.min_delta,
                "fast_training": args.fast_training,
                "local_model_path": args.local_model_path,
                "class_weights": args.class_weights,
            },
            "label_mapping": self.label_mapping,
            "training_results": training_info or {},
        }

        metadata_path = self.output_dir / "training_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Training metadata saved to {metadata_path}")


def load_label_mapping(mapping_path: str) -> Dict[str, str]:
    if not mapping_path:
        return {}

    try:
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        logger.info(f"Loaded label mapping with {len(mapping)} entries")
        return mapping
    except Exception as e:
        logger.error(f"Failed to load label mapping: {e}")
        raise


def parse_class_weights(weights_str: str) -> Union[str, List[float]]:
    """Parse class weights from command line argument"""
    if not weights_str or weights_str.lower() in ["none", "balanced"]:
        return weights_str.lower() if weights_str else None

    try:
        # Try to parse as a list of floats: "1.0,0.1,0.03"
        weights = [float(w.strip()) for w in weights_str.split(",")]
        return weights
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid class weights format: {weights_str}. Use 'balanced', 'none', or comma-separated floats like '1.0,0.1,0.03'"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Optimized Unsloth fine-tuning with CPU parallelization, early stopping, dropout, and class weighting"
    )
    parser.add_argument("--excel", required=True, help="Path to Excel file")
    parser.add_argument("--prompt", required=True, help="Path to prompt template file")
    parser.add_argument(
        "--test-col", default="REVIEW", help="Column name for test data"
    )
    parser.add_argument(
        "--control-col", default="DRUG", help="Column name for control data"
    )
    parser.add_argument(
        "--output-dir", default="./finetuned_optimized", help="Output directory"
    )
    parser.add_argument(
        "--epochs", type=int, default=15, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=2e-5, help="Learning rate"
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--max-examples", type=int, help="Maximum training examples")
    parser.add_argument("--model", help="Custom HuggingFace model name")
    parser.add_argument(
        "--model-size",
        choices=["0.5b", "1.5b", "3b", "7b", "14b"],
        default="1.5b",
        help="Preset model size",
    )
    parser.add_argument("--label-mapping", help="Path to JSON file with label mappings")
    parser.add_argument(
        "--sampling-strategy",
        choices=["balanced", "proportional", "random"],
        default="balanced",
        help="Sampling strategy for large datasets",
    )
    parser.add_argument(
        "--eval-split",
        type=float,
        default=0.1,
        help="Fraction of data to use for evaluation (0 = no evaluation)",
    )
    parser.add_argument(
        "--fast-training",
        action="store_true",
        help="Skip evaluation for maximum training speed",
    )

    # Enhanced arguments
    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=0.1,
        help="Dropout rate for LoRA layers (default: 0.1)",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=3,
        help="Number of evaluation steps with no improvement after which training will be stopped (default: 3)",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.001,
        help="Minimum change in loss to qualify as an improvement for early stopping (default: 0.001)",
    )
    parser.add_argument(
        "--local-model-path",
        type=str,
        help="Path to local model directory (alternative to HuggingFace models)",
    )
    parser.add_argument(
        "--class-weights",
        type=parse_class_weights,
        default=None,
        help="Class weights for imbalanced datasets. Options: 'balanced', 'none', or comma-separated floats like '1.0,0.1,0.03'",
    )

    args = parser.parse_args()

    # Log system information
    logger.info(f"System Information:")
    logger.info(f"  CPU cores: {CPU_COUNT}")
    logger.info(f"  PyTorch threads: {torch.get_num_threads()}")
    logger.info(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"  CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            try:
                device_name = torch.cuda.get_device_name(i)
                logger.info(f"    GPU {i}: {device_name}")
            except Exception as e:
                logger.info(f"    GPU {i}: <Unable to get device name: {e}>")
                # Set CUDA memory management to avoid compatibility issues
                try:
                    torch.cuda.set_per_process_memory_fraction(0.8, device=i)
                    logger.info(f"    Set memory fraction for GPU {i}")
                except Exception:
                    pass

    try:
        label_mapping = load_label_mapping(args.label_mapping)
        finetuner = UnslothFineTuner(args.output_dir, label_mapping)

        train_df, eval_df = finetuner.load_excel_data(
            args.excel,
            args.test_col,
            args.control_col,
            args.max_examples,
            args.sampling_strategy,
            args.eval_split,
        )

        prompt_template = finetuner.load_prompt_template(args.prompt)

        training_data = finetuner.create_training_data(
            train_df, prompt_template, args.test_col, args.control_col
        )

        eval_data = None
        if eval_df is not None:
            eval_data = finetuner.create_training_data(
                eval_df, prompt_template, args.test_col, args.control_col
            )

        training_info = finetuner.fine_tune(
            training_data,
            train_df,  # Pass train_df for class weight computation
            eval_data,
            args.epochs,
            args.learning_rate,
            args.batch_size,
            args.model,
            args.model_size,
            args.fast_training,
            args.dropout_rate,
            args.early_stopping_patience,
            args.min_delta,
            args.local_model_path,
            args.class_weights,
            args.control_col,
        )

        finetuner.save_training_metadata(
            args, train_df, training_data, eval_data, training_info
        )

        logger.info(f"Fine-tuning completed! Check {args.output_dir} for results.")
        logger.info(
            f"Training utilized {training_info.get('cpu_cores_used', CPU_COUNT)} CPU cores"
        )

    except Exception as e:
        logger.error(f"Error during fine-tuning: {e}")
        raise


if __name__ == "__main__":
    main()
