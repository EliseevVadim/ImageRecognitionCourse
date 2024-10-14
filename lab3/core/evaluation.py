import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score

from core.reports.ModelInfo import ModelInfo


def evaluate_model(model: GradientBoostingClassifier, model_name: str, train_data: pd.DataFrame, y_train: pd.Series,
                   test_data: pd.DataFrame, y_test: pd.Series) -> ModelInfo:
    (test_f1,
     test_roc_auc,
     test_score,
     train_f1,
     train_roc_auc,
     train_score) = compute_basic_model_metrics(model, test_data, train_data, y_test, y_train)
    f1_score_history = retrieve_f1_score_history(model, x=test_data, y=y_test)
    roc_auc_history = retrieve_roc_auc_score_history(model, x=test_data, y=y_test)
    loss_history = retrieve_loss_history(model)
    return ModelInfo(
        model=model,
        model_name=model_name,
        train_score=train_score,
        test_score=test_score,
        train_f1=train_f1,
        test_f1=test_f1,
        train_roc_auc=train_roc_auc,
        test_roc_auc=test_roc_auc,
        f1_score_history=f1_score_history,
        roc_auc_history=roc_auc_history,
        loss_history=loss_history
    )


def evaluate_calibrated_model(model: GradientBoostingClassifier, model_name: str, train_data: pd.DataFrame,
                              y_train: pd.Series, test_data: pd.DataFrame, y_test: pd.Series):
    (test_f1,
     test_roc_auc,
     test_score,
     train_f1,
     train_roc_auc,
     train_score) = compute_basic_model_metrics(model, test_data, train_data, y_test, y_train)
    return pd.Series({
        'model_name': model_name,
        'train_score': train_score,
        'train_f1': train_f1,
        'train_roc_auc': train_roc_auc,
        'test_score': test_score,
        'test_f1': test_f1,
        'test_roc_auc': test_roc_auc
    })


def compute_basic_model_metrics(model, test_data, train_data, y_test, y_train):
    train_score = model.score(train_data, y_train)
    test_score = model.score(test_data, y_test)
    train_predictions = model.predict(train_data)
    test_predictions = model.predict(test_data)
    train_predictions_proba = model.predict_proba(train_data)
    test_predictions_proba = model.predict_proba(test_data)
    train_f1 = f1_score(y_train, train_predictions)
    test_f1 = f1_score(y_test, test_predictions)
    train_roc_auc = roc_auc_score(y_train, train_predictions_proba[:, 1])
    test_roc_auc = roc_auc_score(y_test, test_predictions_proba[:, 1])
    return test_f1, test_roc_auc, test_score, train_f1, train_roc_auc, train_score


def retrieve_f1_score_history(model: GradientBoostingClassifier, x: pd.DataFrame, y: pd.Series) -> list:
    history = []
    for predictions in model.staged_predict(x):
        history.append(f1_score(y, predictions))
    return history


def retrieve_roc_auc_score_history(model: GradientBoostingClassifier, x: pd.DataFrame, y: pd.Series) -> list:
    history = []
    for predictions in model.staged_predict_proba(x):
        history.append(roc_auc_score(y, predictions[:, 1]))
    return history


def retrieve_loss_history(model: GradientBoostingClassifier) -> list:
    return model.train_score_.tolist()
