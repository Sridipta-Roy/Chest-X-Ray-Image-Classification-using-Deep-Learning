"""
TensorFlow data pipeline for efficient data loading
"""
import tensorflow as tf
from typing import Tuple


class DataPipeline:
    """Creates TensorFlow data pipelines with preprocessing and augmentation"""
    
    def __init__(self, img_size: Tuple[int, int] = (224, 224), 
                 batch_size: int = 32,
                 model_type: str = "DEAFNet"):
        """
        Initialize data pipeline
        
        Args:
            img_size: Target image size (height, width)
            batch_size: Batch size for training
            model_type: Type of model for appropriate preprocessing
        """
        self.img_size = img_size
        self.batch_size = batch_size
        self.model_type = model_type
        
        # Define data augmentation
        self.data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(height_factor=0.1, width_factor=0.1),
            tf.keras.layers.RandomBrightness(factor=0.1),
        ], name="data_augmentation")
    
    def parse_image(self, filepath: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Load and preprocess an image
        
        Args:
            filepath: Path to image file
            label: Multi-hot encoded label vector
            
        Returns:
            Tuple of (preprocessed_image, label)
        """
        # Read and decode image
        image = tf.io.read_file(filepath)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.resize(image, self.img_size)
        
        # Apply model-specific preprocessing
        if self.model_type == "EfficientNetB0":
            image = tf.keras.applications.efficientnet.preprocess_input(image)
        elif self.model_type == "DenseNet121":
            image = tf.keras.applications.densenet.preprocess_input(image)
        else:  # DEAFNet or SimpleCNN
            image = tf.image.convert_image_dtype(image, tf.float32)
        
        return image, label
    
    def create_dataset(self, filepaths, labels, augment: bool = False) -> tf.data.Dataset:
        """
        Create a complete tf.data.Dataset pipeline
        
        Args:
            filepaths: Array of image file paths
            labels: Array of multi-hot encoded labels
            augment: Whether to apply data augmentation
            
        Returns:
            Batched and prefetched tf.data.Dataset
        """
        # Create dataset from tensors
        dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))
        
        # Parse images
        dataset = dataset.map(
            self.parse_image, 
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Apply augmentation if needed
        if augment:
            dataset = dataset.map(
                lambda x, y: (self.data_augmentation(x, training=True), y),
                num_parallel_calls=tf.data.AUTOTUNE
            )
        
        # Batch and prefetch
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        
        return dataset
    
    def create_train_val_test_datasets(self, X_train, y_train, X_val, y_val, 
                                      X_test, y_test) -> Tuple[tf.data.Dataset, ...]:
        """
        Create training, validation, and test datasets
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            X_test, y_test: Test data
            
        Returns:
            Tuple of (train_ds, val_ds, test_ds)
        """
        print("\nCreating data pipelines...")
        
        train_ds = self.create_dataset(X_train, y_train, augment=True)
        val_ds = self.create_dataset(X_val, y_val, augment=False)
        test_ds = self.create_dataset(X_test, y_test, augment=False)
        
        print("Data pipelines created successfully.")
        return train_ds, val_ds, test_ds