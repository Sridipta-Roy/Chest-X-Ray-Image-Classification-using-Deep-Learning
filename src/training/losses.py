"""
Custom loss functions for training
"""
import tensorflow as tf
import tensorflow.keras.backend as K


def focal_loss(alpha: float = 0.25, gamma: float = 2.0):
    """
    Focal Loss for multi-label classification
    
    Focal Loss addresses class imbalance by down-weighting easy examples
    and focusing on hard examples.
    
    Args:
        alpha: Weighting factor for positive class (0-1)
        gamma: Focusing parameter for modulating loss (typically 2.0)
        
    Returns:
        Loss function that can be used with model.compile()
    """
    def loss(y_true, y_pred):
        """
        Calculate focal loss
        
        Args:
            y_true: Ground truth labels (batch_size, num_classes)
            y_pred: Predicted probabilities (batch_size, num_classes)
            
        Returns:
            Scalar loss value
        """
        # Cast to float32 for numerical stability
        y_true = tf.cast(y_true, tf.float32)
        
        # Clip predictions to prevent log(0)
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1. - K.epsilon())
        
        # Calculate binary cross-entropy
        bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        
        # Calculate probability for modulation
        # p_t is the model's estimated probability for the true class
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        
        # Calculate modulating factor
        # This reduces loss for well-classified examples
        modulating_factor = tf.pow(1.0 - p_t, gamma)
        
        # Calculate alpha factor
        # This balances positive/negative examples
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        
        # Combine all components
        focal_loss_val = alpha_factor * modulating_factor * bce
        
        # Return mean loss
        return K.mean(focal_loss_val)
    
    return loss


def weighted_binary_crossentropy(pos_weight: float = 1.0):
    """
    Weighted Binary Cross-Entropy Loss
    
    Useful for handling class imbalance by assigning different weights
    to positive and negative examples.
    
    Args:
        pos_weight: Weight for positive class
        
    Returns:
        Loss function
    """
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1. - K.epsilon())
        
        # Calculate weighted BCE
        bce = -(pos_weight * y_true * tf.math.log(y_pred) + 
                (1 - y_true) * tf.math.log(1 - y_pred))
        
        return K.mean(bce)
    
    return loss


def binary_crossentropy_loss(y_true, y_pred):
    """
    Standard Binary Cross-Entropy Loss for multi-label classification
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        
    Returns:
        Scalar loss value
    """
    return K.mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))