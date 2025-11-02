"""
Dense-Efficient Attention-Fusion Network (DEAF-Net)
"""
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121, EfficientNetB3
from tensorflow.keras.layers import (Input, Conv2D, Concatenate, GlobalAveragePooling2D, Dense, Dropout)
from tensorflow.keras.models import Model

from .layers import CrossAttention, ResizeToMatch


class DEAFNet:
    """
    Dense-Efficient Attention-Fusion Network
    
    Fusion architecture combining DenseNet121 and EfficientNetB3
    with cross-attention mechanism for enhanced feature learning.
    """
    
    def __init__(self, input_shape: tuple, num_classes: int,
                 fusion_channels: int = 256, dropout_rate: float = 0.5):
        """
        Initialize DEAF-Net
        
        Args:
            input_shape: Input image shape (height, width, channels)
            num_classes: Number of output classes
            fusion_channels: Number of channels for feature fusion
            dropout_rate: Dropout rate for regularization
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.fusion_channels = fusion_channels
        self.dropout_rate = dropout_rate
        
        self.model = None
        self.densenet_base = None
        self.efficientnet_base = None
    
    def build(self):
        """
        Build the DEAF-Net architecture
        
        Returns:
            Tuple of (model, densenet_base, efficientnet_base)
        """
        input_tensor = Input(shape=self.input_shape, name='input')
        
        # Stream 1: DenseNet121 Backbone
        self.densenet_base = DenseNet121(
            weights='imagenet',
            include_top=False,
            input_tensor=input_tensor
        )
        densenet_features = self.densenet_base.get_layer('conv5_block16_concat').output
        
        # Stream 2: EfficientNetB3 Backbone
        self.efficientnet_base = EfficientNetB3(
            weights='imagenet',
            include_top=False,
            input_tensor=input_tensor
        )
        efficientnet_features = self.efficientnet_base.get_layer(
            'block6a_expand_activation'
        ).output
        
        # Feature Alignment - Project to same channel dimension
        d_features_aligned = Conv2D(
            self.fusion_channels,
            (1, 1),
            padding='same',
            activation='relu',
            name='densenet_alignment'
        )(densenet_features)
        
        e_features_aligned = Conv2D(
            self.fusion_channels,
            (1, 1),
            padding='same',
            activation='relu',
            name='efficientnet_alignment'
        )(efficientnet_features)
        
        # Spatial Alignment - Resize to match dimensions
        e_features_resized = ResizeToMatch(name='spatial_alignment')(
            [e_features_aligned, d_features_aligned]
        )
        
        # Feature Fusion - Concatenate aligned features
        fused_features = Concatenate(axis=-1, name='feature_fusion')(
            [d_features_aligned, e_features_resized]
        )
        
        # Cross-Attention Module
        attention = CrossAttention(
            channels=2 * self.fusion_channels,
            name='cross_attention'
        )(fused_features)
        
        # Classification Head
        x = GlobalAveragePooling2D(name='global_pool')(attention)
        x = Dropout(self.dropout_rate, name='dropout')(x)
        outputs = Dense(
            self.num_classes,
            activation='sigmoid',
            dtype='float32',
            name='classifier'
        )(x)
        
        # Build model
        self.model = Model(
            inputs=input_tensor,
            outputs=outputs,
            name='DEAF_Net'
        )
        
        return self.model, self.densenet_base, self.efficientnet_base
    
    def freeze_backbones(self):
        """Freeze backbone networks for transfer learning"""
        if self.densenet_base:
            self.densenet_base.trainable = False
        if self.efficientnet_base:
            self.efficientnet_base.trainable = False
    
    def unfreeze_backbones(self):
        """Unfreeze backbone networks for fine-tuning"""
        if self.densenet_base:
            self.densenet_base.trainable = True
        if self.efficientnet_base:
            self.efficientnet_base.trainable = True
    
    def enable_batch_norm_fine_tuning(self):
        """
        Enable batch normalization layers during fine-tuning
        
        This helps maintain proper statistics during transfer learning.
        """
        for layer in self.model.layers:
            if hasattr(layer, 'layers'):
                for sub_layer in layer.layers:
                    if isinstance(sub_layer, tf.keras.layers.BatchNormalization):
                        sub_layer.trainable = True
            elif isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = True
    
    def get_model(self):
        """Get the built model"""
        if self.model is None:
            raise ValueError("Model not built. Call build() first.")
        return self.model
    
    def summary(self):
        """Print model summary"""
        if self.model:
            self.model.summary()
        else:
            raise ValueError("Model not built. Call build() first.")


def build_deafnet_model(input_shape: tuple, num_classes: int,
                       fusion_channels: int = 256, 
                       dropout_rate: float = 0.5):
    """
    Helper function to build DEAF-Net model
    
    Args:
        input_shape: Input image shape
        num_classes: Number of output classes
        fusion_channels: Channels for fusion layer
        dropout_rate: Dropout rate
        
    Returns:
        Tuple of (model, densenet_base, efficientnet_base)
    """
    deafnet = DEAFNet(
        input_shape=input_shape,
        num_classes=num_classes,
        fusion_channels=fusion_channels,
        dropout_rate=dropout_rate
    )
    return deafnet.build()