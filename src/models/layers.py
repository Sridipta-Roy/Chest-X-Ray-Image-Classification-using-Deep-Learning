"""
Custom layers for neural networks
"""
import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D


class CrossAttention(Layer):
    """
    Cross-Attention Layer for feature fusion
    
    Implements self-attention mechanism to capture long-range dependencies
    in spatial feature maps.
    """
    
    def __init__(self, channels: int, **kwargs):
        """
        Initialize Cross-Attention layer
        
        Args:
            channels: Number of channels for attention computation
        """
        super().__init__(**kwargs)
        self.channels = channels
        
        # Query, Key, Value convolutions
        self.query_conv = Conv2D(channels, 1, name="ca_query")
        self.key_conv = Conv2D(channels, 1, name="ca_key")
        self.value_conv = Conv2D(channels, 1, name="ca_value")
        
        # Learnable scaling parameter
        self.gamma = self.add_weight(
            name="ca_gamma",
            shape=[1],
            initializer="zeros",
            trainable=True
        )
    
    def call(self, feat):
        """
        Forward pass of Cross-Attention
        
        Args:
            feat: Input feature map (batch, height, width, channels)
            
        Returns:
            Attention-enhanced features
        """
        batch = tf.shape(feat)[0]
        h = tf.shape(feat)[1]
        w = tf.shape(feat)[2]
        c = tf.shape(feat)[3]
        
        # Generate Query, Key, Value
        Q = self.query_conv(feat)
        K = self.key_conv(feat)
        V = self.value_conv(feat)
        
        # Reshape to (batch, h*w, channels)
        Q_flat = tf.reshape(Q, [batch, -1, self.channels])
        K_flat = tf.reshape(K, [batch, -1, self.channels])
        V_flat = tf.reshape(V, [batch, -1, self.channels])
        
        # Calculate attention scores
        # scores = Q @ K^T / sqrt(channels)
        scores = tf.matmul(Q_flat, K_flat, transpose_b=True)
        scores = scores / tf.cast(tf.math.sqrt(tf.cast(self.channels, tf.float32)), 'float32')
        
        # Apply softmax to get attention weights
        A = tf.nn.softmax(tf.cast(scores, dtype=tf.float32), axis=-1)
        
        # Apply attention to values
        out = tf.matmul(tf.cast(A, V_flat.dtype), V_flat)
        
        # Reshape back to spatial dimensions
        out = tf.reshape(out, [batch, h, w, self.channels])
        
        # Residual connection with learned scaling
        return self.gamma * out + feat
    
    def get_config(self):
        """Get layer configuration for serialization"""
        config = super().get_config()
        config.update({"channels": self.channels})
        return config


class ResizeToMatch(Layer):
    """
    Resizes input tensor to match target spatial dimensions
    
    Useful for feature fusion when different backbones produce
    features of different spatial sizes.
    """
    
    def call(self, inputs):
        """
        Resize source to match target dimensions
        
        Args:
            inputs: Tuple of (source_features, target_features)
            
        Returns:
            Resized source features
        """
        source_features, target_features = inputs
        target_shape = tf.shape(target_features)[1:3]
        return tf.image.resize(source_features, target_shape)
    
    def get_config(self):
        """Get layer configuration for serialization"""
        return super().get_config()


class SpatialAttention(Layer):
    """
    Spatial Attention Module
    
    Focuses on 'where' is important in the feature map.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv = Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')
    
    def call(self, x):
        """
        Apply spatial attention
        
        Args:
            x: Input features (batch, height, width, channels)
            
        Returns:
            Attention-weighted features
        """
        # Average and max pooling across channel dimension
        avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
        
        # Concatenate
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        
        # Generate attention map
        attention = self.conv(concat)
        
        # Apply attention
        return x * attention


class ChannelAttention(Layer):
    """
    Channel Attention Module
    
    Focuses on 'what' is important in the feature map.
    """
    
    def __init__(self, reduction_ratio: int = 16, **kwargs):
        """
        Initialize Channel Attention
        
        Args:
            reduction_ratio: Reduction ratio for bottleneck layer
        """
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
    
    def build(self, input_shape):
        channels = input_shape[-1]
        self.fc1 = tf.keras.layers.Dense(channels // self.reduction_ratio, activation='relu')
        self.fc2 = tf.keras.layers.Dense(channels, activation='sigmoid')
    
    def call(self, x):
        """
        Apply channel attention
        
        Args:
            x: Input features
            
        Returns:
            Attention-weighted features
        """
        # Global average pooling
        gap = tf.reduce_mean(x, axis=[1, 2], keepdims=False)
        
        # FC layers
        attention = self.fc1(gap)
        attention = self.fc2(attention)
        
        # Reshape and apply
        attention = tf.reshape(attention, [-1, 1, 1, tf.shape(x)[-1]])
        return x * attention
    
    def get_config(self):
        config = super().get_config()
        config.update({"reduction_ratio": self.reduction_ratio})
        return config