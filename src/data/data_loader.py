"""
Data loading and preprocessing utilities
"""
import os
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from pathlib import Path


class ChestXRayDataLoader:
    """Handles loading and preprocessing of ChestXRay dataset"""
    
    def __init__(self, metadata_path: str, image_root: str):
        """
        Initialize data loader
        
        Args:
            metadata_path: Path to Data_Entry_2017.csv
            image_root: Root directory containing image folders
        """
        self.metadata_path = metadata_path
        self.image_root = image_root
        self.df = None
        self.all_labels = None
        
    def load_metadata(self) -> pd.DataFrame:
        """Load and preprocess metadata CSV"""
        print("Loading metadata...")
        self.df = pd.read_csv(self.metadata_path)
        
        # Build image path mapping
        image_path_map = self._build_image_path_map()
        self.df['full_path'] = self.df['Image Index'].map(image_path_map)
        
        # Clean up dataframe
        self.df = self.df.dropna(subset=['full_path'])
        if 'Unnamed: 11' in self.df.columns:
            self.df.drop(columns=['Unnamed: 11'], inplace=True)
            
        print(f"Total images found: {len(self.df)}")
        return self.df
    
    def _build_image_path_map(self) -> Dict[str, str]:
        """Build mapping from image filename to full path"""
        image_path_map = {}
        
        for folder in os.listdir(self.image_root):
            subfolder = os.path.join(self.image_root, folder, "images")
            if not os.path.isdir(subfolder):
                continue
                
            for img_file in os.listdir(subfolder):
                if img_file.endswith(".png"):
                    full_path = os.path.join(subfolder, img_file)
                    image_path_map[img_file] = full_path
                    
        return image_path_map
    
    def encode_labels(self) -> List[str]:
        """
        Multi-hot encode labels
        
        Returns:
            List of all unique labels
        """
        # Remove "No Finding" label
        self.df['Finding Labels'] = self.df['Finding Labels'].str.replace('No Finding', '')
        
        # Get all unique labels
        all_labels_raw = [
            label for labels in self.df['Finding Labels'] 
            for label in labels.split('|')
        ]
        self.all_labels = sorted([label for label in pd.Series(all_labels_raw).unique() if label])
        
        print(f'\nAll Labels ({len(self.all_labels)}): {self.all_labels}')
        
        # Create binary columns for each label
        for label in self.all_labels:
            self.df[label] = self.df['Finding Labels'].str.contains(label).astype(float)
        
        # Create label vector column
        self.df['labels_vec'] = self.df.apply(
            lambda row: [row[label] for label in self.all_labels], 
            axis=1
        )
        
        print("Labels have been multi-hot encoded.")
        return self.all_labels
    
    def get_label_distribution(self) -> pd.Series:
        """Get frequency distribution of all labels"""
        label_counts = Counter(
            label for labels in self.df['Finding Labels'] 
            for label in labels.split('|') if label
        )
        return pd.Series(label_counts).sort_values(ascending=False)
    
    def get_subset(self, fraction: float = 0.5, seed: int = 42) -> pd.DataFrame:
        """
        Get a random subset of data based on unique patients
        
        Args:
            fraction: Fraction of patients to include
            seed: Random seed for reproducibility
            
        Returns:
            Subset dataframe
        """
        np.random.seed(seed)
        unique_patients = self.df['Patient ID'].unique()
        subset_patients = np.random.choice(
            unique_patients,
            size=int(len(unique_patients) * fraction),
            replace=False
        )
        subset_df = self.df[self.df['Patient ID'].isin(subset_patients)]
        print(f"Created subset with {len(subset_df)} images from {len(subset_patients)} patients")
        return subset_df
    
    def get_data(self) -> Tuple[pd.DataFrame, List[str]]:
        """
        Complete data loading pipeline
        
        Returns:
            Tuple of (dataframe, list of labels)
        """
        self.load_metadata()
        self.encode_labels()
        return self.df, self.all_labels