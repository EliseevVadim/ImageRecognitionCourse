import pandas as pd

from core.reports.ModelInfo import ModelInfo


class TrainingReport:
    def __init__(self):
        self.models_infos = []
        pass

    def add_model_info(self, model_info: ModelInfo):
        self.models_infos.append(model_info)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([model_info.to_series()
                             for model_info in self.models_infos]).sort_values(by='test_roc_auc', ascending=False)

    def __iter__(self):
        return iter(self.models_infos)

    def __getitem__(self, index):
        return self.models_infos[index]
