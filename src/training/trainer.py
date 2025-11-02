"""
Model training orchestration
"""
import os
import pandas as pd
import tensorflow as tf
from typing import Optional, List

from ..models.simple_cnn import build_simple_cnn
from ..models.densenet import build_densenet_model
from ..models.efficientnet import build_efficientnet_model
from ..models.deafnet import build_deafnet_model
from .losses import focal_loss
from .callbacks import get_callbacks


class ModelTrainer:
    """Handles model building, training, and evaluation"""
    
    def __init__(self, model_name: str, num_classes: int, config):
        """
        Initialize trainer
        
        Args:
            model_name: Name of model architecture
            num_classes: Number of output classes
            config: Configuration object
        """
        self.model_name = model_name
        self.num_classes = num_classes
        self.config = config
        
        self.model = None
        self.backbones = []
        self.history_stage1 = None
        self.history_stage2 = None
        
    def build_model(self):
        """Build the specified model architecture"""
        input_shape = (*self.config.IMG_SIZE, 3)
        
        if self.model_name == "SimpleCNN":
            self.model = build_simple_cnn(
                input_shape=input_shape,
                num_classes=self.num_classes
            )
            self.backbones = []
            
        elif self.model_name == "DenseNet121":
            self.model, base_model = build_densenet_model(
                input_shape=input_shape,
                num_classes=self.num_classes,
                dropout_rate=self.config.DROPOUT_RATE
            )
            self.backbones = [base_model]
            
        elif self.model_name == "EfficientNetB0":
            self.model, base_model = build_efficientnet_model(
                input_shape=input_shape,
                num_classes=self.num_classes,
                dropout_rate=self.config.DROPOUT_RATE
            )
            self.backbones = [base_model]
            
        elif self.model_name == "DEAFNet":
            self.model, densenet, effnet = build_deafnet_model(
                input_shape=input_shape,
                num_classes=self.num_classes,
                fusion_channels=self.config.FUSION_CHANNELS,
                dropout_rate=self.config.DROPOUT_RATE
            )
            self.backbones = [densenet, effnet]
            
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        
        print(f"\n{self.model_name} Model Architecture:")
        self.model.summary()
        
        return self.model
    
    def train(self, train_ds, val_ds, test_ds):
        """
        Train the model
        
        Args:
            train_ds: Training dataset
            val_ds: Validation dataset
            test_ds: Test dataset
            
        Returns:
            Training history
        """
        callbacks = get_callbacks(self.model_name, self.config)
        
        # SimpleCNN: Single-stage training
        if self.model_name == "SimpleCNN":
            return self._train_simple_cnn(train_ds, val_ds, callbacks)
        
        # Transfer Learning: Two-stage training
        else:
            return self._train_transfer_learning(train_ds, val_ds, callbacks)
    
    def _train_simple_cnn(self, train_ds, val_ds, callbacks):
        """Train SimpleCNN model (single stage)"""
        print("\n" + "="*60)
        print(f"TRAINING {self.model_name}")
        print("="*60)
        
        # Compile model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.config.LEARNING_RATE_SIMPLE_CNN
            ),
            loss=focal_loss(
                alpha=self.config.FOCAL_LOSS_ALPHA,
                gamma=self.config.FOCAL_LOSS_GAMMA
            ),
            metrics=[tf.keras.metrics.AUC(name='auc', multi_label=True)]
        )
        
        # Train
        self.history_stage1 = self.model.fit(
            train_ds,
            epochs=self.config.EPOCHS_STAGE2,  # Use stage2 epochs
            validation_data=val_ds,
            callbacks=callbacks
        )
        
        # Save history
        self._save_history(self.history_stage1, initial_epoch=0)
        
        return self.history_stage1
    
    def _train_transfer_learning(self, train_ds, val_ds, callbacks):
        """Train model with transfer learning (two stages)"""
        
        # Stage 1: Train head layers only
        print("\n" + "="*60)
        print(f"STAGE 1: TRAINING HEAD LAYERS ({self.model_name})")
        print("="*60)
        
        # Freeze backbones
        for backbone in self.backbones:
            backbone.trainable = False
        
        # Special handling for DEAFNet BatchNorm layers
        if self.model_name == 'DEAFNet':
            self._enable_batch_norm_layers()
        
        # Compile for stage 1
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.config.LEARNING_RATE_STAGE1
            ),
            loss=focal_loss(
                alpha=self.config.FOCAL_LOSS_ALPHA,
                gamma=self.config.FOCAL_LOSS_GAMMA
            ),
            metrics=[tf.keras.metrics.AUC(name='auc', multi_label=True)]
        )
        
        # Train stage 1
        self.history_stage1 = self.model.fit(
            train_ds,
            epochs=self.config.EPOCHS_STAGE1,
            validation_data=val_ds,
            callbacks=callbacks
        )
        
        # Save history
        self._save_history(self.history_stage1, initial_epoch=0)
        
        # Stage 2: Fine-tune full model
        print("\n" + "="*60)
        print(f"STAGE 2: FINE-TUNING FULL MODEL ({self.model_name})")
        print("="*60)
        
        # Unfreeze backbones
        for backbone in self.backbones:
            backbone.trainable = True
        
        # Compile for stage 2
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.config.LEARNING_RATE_STAGE2
            ),
            loss=focal_loss(
                alpha=self.config.FOCAL_LOSS_ALPHA,
                gamma=self.config.FOCAL_LOSS_GAMMA
            ),
            metrics=[tf.keras.metrics.AUC(name='auc', multi_label=True)]
        )
        
        # Train stage 2
        self.history_stage2 = self.model.fit(
            train_ds,
            epochs=self.config.EPOCHS_STAGE2,
            validation_data=val_ds,
            initial_epoch=len(self.history_stage1.epoch),
            callbacks=callbacks
        )
        
        # Save history
        self._save_history(
            self.history_stage2,
            initial_epoch=len(self.history_stage1.epoch)
        )
        
        return self.history_stage2
    
    def _enable_batch_norm_layers(self):
        """Enable BatchNormalization layers for DEAFNet"""
        for layer in self.model.layers:
            if hasattr(layer, 'layers'):
                for sub_layer in layer.layers:
                    if isinstance(sub_layer, tf.keras.layers.BatchNormalization):
                        sub_layer.trainable = True
            elif isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = True
    
    def _save_history(self, history, initial_epoch: int = 0):
        """Save training history to CSV"""
        history_df = pd.DataFrame(history.history)
        history_df['epoch'] = initial_epoch + history_df.index + 1
        history_df['model'] = self.model_name
        
        history_path = self.config.get_history_path()
        header = not os.path.exists(history_path)
        history_df.to_csv(history_path, mode='a', header=header, index=False)
        
        print(f"\nSaved training history to {history_path}")
    
    def evaluate(self, test_ds, labels_list):
        """
        Evaluate model on test set
        
        Args:
            test_ds: Test dataset
            labels_list: List of label names
        """
        from ..evaluation.evaluator import ModelEvaluator
        
        evaluator = ModelEvaluator(
            model=self.model,
            model_name=self.model_name,
            config=self.config
        )
        
        evaluator.evaluate(test_ds, labels_list)