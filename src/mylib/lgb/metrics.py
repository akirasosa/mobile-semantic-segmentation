import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_squared_log_error


def lgb_rmsle_score(preds: np.ndarray, dval: lgb.Dataset):
    label = dval.get_label()
    y_true = np.exp(label)
    y_pred = np.exp(preds)

    return 'rmsle', np.sqrt(mean_squared_log_error(y_true, y_pred)), False