"""
Multi-label stratified data splitting utilities
"""
import numpy as np
import pandas as pd
from typing import Tuple
from skmultilearn.model_selection import IterativeStratification


class DataSplitter:
    """Handles stratified splitting for multi-label classification"""
    
    def __init__(self, train_ratio: float = 0.7, val_ratio: float = 0.1, test_ratio: float = 0.2):
        """
        Initialize data splitter
        
        Args:
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
        """
        assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Perform multi-label stratified split
        
        Args:
            df: Dataframe with 'full_path' and 'labels_vec' columns
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        print("\nSplitting data into training, validation, and test sets...")
        
        X = df['full_path'].values.reshape(-1, 1)
        y = np.array(df['labels_vec'].tolist())
        
        # Calculate n_splits for train/temp split
        # n_splits = 1 / (1 - train_ratio)
        train_split = int(1 / (1 - self.train_ratio))
        
        # Split into Train and Temp (val + test)
        stratifier_traintest = IterativeStratification(n_splits=train_split, order=1)
        train_indices, temp_indices = next(stratifier_traintest.split(X, y))
        
        train_df = df.iloc[train_indices]
        temp_df = df.iloc[temp_indices]
        
        # Split Temp into Val and Test
        X_temp = temp_df['full_path'].values.reshape(-1, 1)
        y_temp = np.array(temp_df['labels_vec'].tolist())
        
        # n_splits for val/test split
        val_test_split = int(1 / (self.test_ratio / (self.val_ratio + self.test_ratio)))
        
        stratifier_valtest = IterativeStratification(n_splits=val_test_split, order=1)
        val_indices, test_indices = next(stratifier_valtest.split(X_temp, y_temp))
        
        val_df = temp_df.iloc[val_indices]
        test_df = temp_df.iloc[test_indices]
        
        print("\nData split successfully:")
        print(f"Train samples: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
        print(f"Validation samples: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
        print(f"Test samples: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
        
        return train_df, val_df, test_df
    
    def get_arrays(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                   test_df: pd.DataFrame) -> Tuple:
        """
        Extract numpy arrays from dataframes
        
        Args:
            train_df: Training dataframe
            val_df: Validation dataframe
            test_df: Test dataframe
            
        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        X_train = train_df['full_path'].values
        y_train = np.array(train_df['labels_vec'].tolist())
        
        X_val = val_df['full_path'].values
        y_val = np.array(val_df['labels_vec'].tolist())
        
        X_test = test_df['full_path'].values
        y_test = np.array(test_df['labels_vec'].tolist())
        
        return X_train, y_train, X_val, y_val, X_test, y_test