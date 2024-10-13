from core.evaluation import *
from core.reports.TrainingReport import TrainingReport


class GBModelTrainer:
    def __init__(self, train_x: pd.DataFrame, train_y: pd.Series, test_x: pd.DataFrame, test_y: pd.Series,
                 trees_numbers: list, learning_rates: list, subsample_sizes: list, losses: list,
                 random_state=42):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.trees_numbers = trees_numbers
        self.learning_rates = learning_rates
        self.subsample_sizes = subsample_sizes
        self.losses = losses
        self.random_state = random_state

    def train(self):
        best_model = None
        best_roc_auc = 0
        training_report = TrainingReport()
        for trees_number in self.trees_numbers:
            for learning_rate in self.learning_rates:
                for subsample_size in self.subsample_sizes:
                    for loss in self.losses:
                        model = GradientBoostingClassifier(
                            n_estimators=trees_number,
                            learning_rate=learning_rate,
                            subsample=subsample_size,
                            loss=loss,
                            max_depth=3,
                            random_state=self.random_state
                        )
                        model.fit(self.train_x, self.train_y)
                        model_summary = evaluate_model(model=model,
                                                       model_name=f"model_{loss}_{trees_number}_trees_{learning_rate}"
                                                                  f"_lr_"
                                                                  f"{subsample_size}_subsample",
                                                       train_data=self.train_x,
                                                       y_train=self.train_y, test_data=self.test_x, y_test=self.test_y)
                        training_report.add_model_info(model_summary)
                        if model_summary.test_roc_auc > best_roc_auc:
                            best_model = model
                            best_roc_auc = model_summary.test_roc_auc
        return training_report, best_model
