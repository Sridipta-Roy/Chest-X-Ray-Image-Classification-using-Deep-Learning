from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import (Input, GlobalAveragePooling2D, Dense, Dropout)
from tensorflow.keras.models import Model

def build_densenet_model(input_shape, num_classes, dropout_rate=0.5):
    """Build DenseNet121 model"""
    base_model = DenseNet121(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = False
    
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(num_classes, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='DenseNet121')
    return model, base_model