"""
Main training script for ChestXRay Multi-Label Classification
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import tensorflow as tf
from config import Config, DevelopmentConfig
from src.data.data_loader import ChestXRayDataLoader
from src.data.data_split import DataSplitter
from src.data.data_pipeline import DataPipeline
from src.training.trainer import ModelTrainer
from src.utils.logger import setup_logger


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train ChestXRay Multi-Label Classification Model'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='DEAFNet',
        choices=['SimpleCNN', 'DenseNet121', 'EfficientNetB0', 'DEAFNet'],
        help='Model architecture to train'
    )
    
    parser.add_argument(
        '--dev',
        action='store_true',
        help='Use development config (subset of data, fewer epochs)'
    )
    
    parser.add_argument(
        '--gdrive-root',
        type=str,
        default=None,
        help='Google Drive root path (if different from config)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size for training'
    )
    
    parser.add_argument(
        '--epochs-stage1',
        type=int,
        default=None,
        help='Number of epochs for stage 1 (head training)'
    )
    
    parser.add_argument(
        '--epochs-stage2',
        type=int,
        default=None,
        help='Number of epochs for stage 2 (fine-tuning)'
    )
    
    return parser.parse_args()


def main():
    """Main training pipeline"""
    # Parse arguments
    args = parse_args()
    
    # Setup configuration
    config = DevelopmentConfig if args.dev else Config
    
    # Update config based on arguments
    if args.gdrive_root:
        config.update_gdrive_path(args.gdrive_root)
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.epochs_stage1:
        config.EPOCHS_STAGE1 = args.epochs_stage1
    if args.epochs_stage2:
        config.EPOCHS_STAGE2 = args.epochs_stage2
    
    # Setup logger
    logger = setup_logger('training', config.LOGS_DIR / 'training.log')
    logger.info(f"Starting training for model: {args.model}")
    logger.info(f"Configuration: {'Development' if args.dev else 'Production'}")
    
    # Step 1: Load Data
    logger.info("Step 1: Loading data...")
    data_loader = ChestXRayDataLoader(
        metadata_path=config.METADATA_PATH,
        image_root=config.GDRIVE_ROOT
    )
    df, all_labels = data_loader.get_data()
    
    # Use subset if specified
    if config.USE_SUBSET:
        df = data_loader.get_subset(fraction=config.SUBSET_FRACTION, seed=config.SEED)
    
    logger.info(f"Total images: {len(df)}")
    logger.info(f"Number of labels: {len(all_labels)}")
    
    # Step 2: Split Data
    logger.info("Step 2: Splitting data...")
    splitter = DataSplitter(
        train_ratio=config.TRAIN_RATIO,
        val_ratio=config.VAL_RATIO,
        test_ratio=config.TEST_RATIO
    )
    train_df, val_df, test_df = splitter.split(df)
    X_train, y_train, X_val, y_val, X_test, y_test = splitter.get_arrays(
        train_df, val_df, test_df
    )
    
    # Step 3: Create Data Pipelines
    logger.info("Step 3: Creating data pipelines...")
    pipeline = DataPipeline(
        img_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        model_type=args.model
    )
    train_ds, val_ds, test_ds = pipeline.create_train_val_test_datasets(
        X_train, y_train, X_val, y_val, X_test, y_test
    )
    
    # Step 4: Build and Train Model
    logger.info(f"Step 4: Building {args.model} model...")
    trainer = ModelTrainer(
        model_name=args.model,
        num_classes=len(all_labels),
        config=config
    )
    
    # Build model
    model = trainer.build_model()
    logger.info("Model built successfully")
    
    # Train model
    logger.info("Starting training...")
    history = trainer.train(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds
    )
    
    logger.info("Training completed successfully!")
    
    # Step 5: Evaluate
    logger.info("Step 5: Final evaluation...")
    trainer.evaluate(test_ds, all_labels)
    
    logger.info("All done!")


if __name__ == "__main__":
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    
    # Run main
    main()