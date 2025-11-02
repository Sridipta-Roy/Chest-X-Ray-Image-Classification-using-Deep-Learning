import tensorflow as tf
import os

def get_callbacks(model_name, config):
    """Create training callbacks"""
    
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(config.get_model_path(model_name)),
        monitor='val_auc',
        mode='max',
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_auc',
        mode='max',
        patience=config.EARLY_STOPPING_PATIENCE,
        verbose=1,
        restore_best_weights=True
    )
    
    return [model_checkpoint, early_stopping]