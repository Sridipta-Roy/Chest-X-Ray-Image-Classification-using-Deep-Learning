"""
Configuration file for ChestXRay Multi-Label Classification
"""
import os
from pathlib import Path

class Config:
    """Base configuration class"""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    
    # Google Drive paths (update these based on your drive structure)
    GDRIVE_ROOT = "/content/drive/MyDrive/Chest XRay"
    DATA_DIR = os.path.join(GDRIVE_ROOT, "archive")
    METADATA_PATH = os.path.join(DATA_DIR, "Data_Entry_2017.csv")
    
    # Output directories
    OUTPUTS_DIR = PROJECT_ROOT / "outputs"
    MODELS_DIR = OUTPUTS_DIR / "models"
    LOGS_DIR = OUTPUTS_DIR / "logs"
    FIGURES_DIR = OUTPUTS_DIR / "figures"
    HISTORY_DIR = OUTPUTS_DIR / "history"
    
    # Create directories if they don't exist
    for dir_path in [OUTPUTS_DIR, MODELS_DIR, LOGS_DIR, FIGURES_DIR, HISTORY_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Model parameters
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    DROPOUT_RATE = 0.5
    SEED = 42
    
    # Training parameters
    EPOCHS_STAGE1 = 10
    EPOCHS_STAGE2 = 25
    LEARNING_RATE_STAGE1 = 1e-3
    LEARNING_RATE_STAGE2 = 1e-5
    LEARNING_RATE_SIMPLE_CNN = 1e-4
    
    # Data split ratios
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.1
    TEST_RATIO = 0.2
    
    # Development settings
    USE_SUBSET = False
    SUBSET_FRACTION = 0.5
    
    # Loss function parameters
    FOCAL_LOSS_ALPHA = 0.25
    FOCAL_LOSS_GAMMA = 2.0
    
    # Uncertainty estimation
    MC_DROPOUT_SAMPLES = 100
    
    # Available models
    AVAILABLE_MODELS = ["SimpleCNN", "DenseNet121", "EfficientNetB0", "DEAFNet"]
    
    # Callbacks
    EARLY_STOPPING_PATIENCE = 5
    
    # Fusion network parameters
    FUSION_CHANNELS = 256
    
    @classmethod
    def get_model_path(cls, model_name: str) -> Path:
        """Get the path for saving model weights"""
        return cls.MODELS_DIR / f"{model_name.lower()}_best.weights.h5"
    
    @classmethod
    def get_history_path(cls) -> Path:
        """Get the path for training history CSV"""
        return cls.HISTORY_DIR / "training_history_combined.csv"
    
    @classmethod
    def validate_model_name(cls, model_name: str) -> bool:
        """Validate if the model name is supported"""
        return model_name in cls.AVAILABLE_MODELS
    
    @classmethod
    def update_gdrive_path(cls, gdrive_root: str):
        """Update Google Drive paths (useful for different environments)"""
        cls.GDRIVE_ROOT = gdrive_root
        cls.DATA_DIR = os.path.join(gdrive_root, "archive")
        cls.METADATA_PATH = os.path.join(cls.DATA_DIR, "Data_Entry_2017.csv")


class DevelopmentConfig(Config):
    """Configuration for development/testing"""
    USE_SUBSET = True
    SUBSET_FRACTION = 0.1
    EPOCHS_STAGE1 = 2
    EPOCHS_STAGE2 = 3


class ProductionConfig(Config):
    """Configuration for production training"""
    USE_SUBSET = False
    EPOCHS_STAGE1 = 10
    EPOCHS_STAGE2 = 25