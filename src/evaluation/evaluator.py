import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt

class ModelEvaluator:
    """Model evaluation and visualization"""
    
    def __init__(self, model, model_name, config):
        self.model = model
        self.model_name = model_name
        self.config = config
    
    def evaluate(self, test_ds, labels_list):
        """Evaluate model and plot results"""
        print("\n--- FINAL EVALUATION ON TEST SET ---")
        
        # Load best weights
        best_weights_path = self.config.get_model_path(self.model_name)
        print(f"Loading best weights from {best_weights_path}...")
        self.model.load_weights(str(best_weights_path))
        
        # Evaluate
        results = self.model.evaluate(test_ds, return_dict=True, verbose=0)
        print(f"\nTest Loss: {results['loss']:.4f}")
        print(f"Test AUC: {results['auc']:.4f}")
        
        # Get predictions
        test_true = np.concatenate([y for x, y in test_ds], axis=0)
        test_pred = self.model.predict(test_ds)
        
        # Calculate class-wise AUC
        final_auc = roc_auc_score(test_true, test_pred, average='macro')
        print(f"\nTest AUC (macro-average): {final_auc:.4f}")
        
        print("\n--- Class-wise Test AUC ---")
        class_auc_scores = roc_auc_score(test_true, test_pred, average=None)
        for i, pathology in enumerate(labels_list):
            print(f"{pathology:>20}: {class_auc_scores[i]:.4f}")
        
        # Plot ROC curves
        self._plot_roc_curves(test_true, test_pred, labels_list)
    
    def _plot_roc_curves(self, y_true, y_pred, labels_list):
        """Plot ROC curves for all classes"""
        plt.figure(figsize=(12, 10))
        
        for i, pathology in enumerate(labels_list):
            fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{pathology} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.50)')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves - {self.model_name}')
        plt.legend(loc="lower right", fontsize=8)
        plt.grid(True)
        
        # Save figure
        save_path = self.config.FIGURES_DIR / f'roc_curves_{self.model_name.lower()}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nROC curves saved to {save_path}")
        plt.show()