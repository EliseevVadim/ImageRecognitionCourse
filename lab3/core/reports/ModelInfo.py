import pandas as pd

from dataclasses import dataclass

from sklearn.ensemble import GradientBoostingClassifier


@dataclass
class ModelInfo:
    model: GradientBoostingClassifier
    model_name: str
    train_score: float
    test_score: float
    train_f1: float
    test_f1: float
    train_roc_auc: float
    test_roc_auc: float
    loss_history: list
    f1_score_history: list
    roc_auc_history: list

    def to_series(self) -> pd.Series:
        return pd.Series({
            'model_name': self.model_name,
            'train_score': self.train_score,
            'train_f1': self.train_f1,
            'train_roc_auc': self.train_roc_auc,
            'test_score': self.test_score,
            'test_f1': self.test_f1,
            'test_roc_auc': self.test_roc_auc
        })
