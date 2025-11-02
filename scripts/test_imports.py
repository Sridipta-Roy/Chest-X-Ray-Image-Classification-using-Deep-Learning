def test_imports():
    """Test that all modules can be imported"""
    from config import Config
    from src.data.data_loader import ChestXRayDataLoader
    from src.data.data_split import DataSplitter
    from src.data.data_pipeline import DataPipeline
    from src.training.losses import focal_loss
    from src.models.deafnet import build_deafnet_model
    print("All imports successful!")

if __name__ == "__main__":
    test_imports()