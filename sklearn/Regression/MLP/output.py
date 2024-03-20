from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from functools import partial
import numpy as np
import pandas as pd

from .utils import (add_total, 
                    multiclass_curve_wrapper, 
                    multiclass_score_wrapper, 
                    mcm_based_score, 
                    auc_curve_from_roc, 
                    correlation, 
                    gain_curve)

def get_triple(data, target_col, weight_col):
    """
    从提供的DataFrame中提取特征集X，目标变量y，以及样本权重weight。
    
    参数:
    - data: DataFrame，包含特征、目标变量和样本权重的完整数据集。
    - target_col: str，目标变量的列名。
    - weight_col: str，样本权重的列名。

    返回:
    - X: DataFrame，仅包含特征数据的部分。
    - y: Series，包含目标变量的数据。
    - weight: Series，包含样本权重的数据。
    """
    if not hasattr(data, "columns"):
        raise AttributeError("Data should be a pandas DataFrame with column info.")
    
    # 确定特征列：排除目标变量和权重列
    feature_cols = [col for col in data.columns if col not in [target_col, weight_col]]
    X = data[feature_cols]
    y = data[target_col] if target_col is not None else None
    weight = data[weight_col] if weight_col is not None else None
    
    return X, y, weight


class BaseOutputGenerator:
    def __init__(self):
        self.engine = None
        self.X_train = None
        self.y_train = None
        self.weight_train = None
        self.X_test = None
        self.y_test = None
        self.weight_test = None

    def set_engine(self, engine):
        self.engine = engine
        return self

    def _set_data(self):
        target_col = self.engine.info.get("target_col")
        weight_col = self.engine.info.get("weight_col")
        data_train = self.engine.data_train
        data_test = self.engine.data_test
        self.X_train, self.y_train, self.weight_train = self._get_triple(data_train, target_col, weight_col)
        self.X_test, self.y_test, self.weight_test = self._get_triple(data_test, target_col, weight_col)
        return self

    def _get_triple(self, data, target_col, weight_col):
        # 假设实现根据你的数据结构
        X = data.drop([target_col, weight_col], axis=1)
        y = data[target_col]
        weight = data[weight_col]
        return X, y, weight

    def output(self):
        raise NotImplementedError("Subclass must implement abstract method")



class ClassificationOutputGenerator(BaseOutputGenerator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def output(self):
        self._set_data()
        features = pd.concat([self.X_train, self.X_test], axis=0)
        y_pred_train = self.engine.pipeline.predict(self.X_train)
        y_score_train = self.engine.pipeline.predict_proba(self.X_train)
        y_pred_test = self.engine.pipeline.predict(self.X_test)
        y_score_test = self.engine.pipeline.predict_proba(self.X_test)
        
        y_score = np.concatenate([y_score_train, y_score_test], axis=0)
        y_pred = np.concatenate([y_pred_train, y_pred_test], axis=0)
        y = np.concatenate([self.y_train, self.y_test], axis=0)
        weight = np.concatenate([self.weight_train, self.weight_test], axis=0)
        
        labels = np.unique(y)
        
        confusion_mtx = {
            "train": add_total(confusion_matrix(self.y_train, y_pred_train, sample_weight=self.weight_train, labels=labels)),
            "test": add_total(confusion_matrix(self.y_test, y_pred_test, sample_weight=self.weight_test, labels=labels)),
            "classes": [f"class_{i}" for i in labels] + ["total"]
        }
        
        prediction_evaluation = {
            "gain": multiclass_curve_wrapper(gain_curve, y_true=y, y_score=y_score, sample_weight=weight),
            "lift": multiclass_curve_wrapper(partial(gain_curve, metric="lift"), y_true=y, y_score=y_score, sample_weight=weight),
            "table": mcm_based_score(y_true=y, y_pred=y_pred, sample_weight=weight)
        }
        
        table_train = mcm_based_score(y_true=self.y_train, y_pred=y_pred_train, sample_weight=self.weight_train)
        table_test = mcm_based_score(y_true=self.y_test, y_pred=y_pred_test, sample_weight=self.weight_test)
        
        performance_evaluation = {
            "roc_curve_train": multiclass_curve_wrapper(roc_curve, y_true=self.y_train, y_score=y_score_train, sample_weight=self.weight_train),
            "roc_curve_test": multiclass_curve_wrapper(roc_curve, y_true=self.y_test, y_score=y_score_test, sample_weight=self.weight_test),
            "roc_score_train": pd.merge(multiclass_score_wrapper(roc_auc_score, y_true=self.y_train, y_score=y_score_train, sample_weight=self.weight_train), table_train[["recall", "fpr"]], left_index=True, right_index=True),
            "roc_score_test": pd.merge(multiclass_score_wrapper(roc_auc_score, y_true=self.y_test, y_score=y_score_test, sample_weight=self.weight_test), table_test[["recall", "fpr"]], left_index=True, right_index=True)
        }
        
        performance_evaluation["auc_curve_train"] = {k: auc_curve_from_roc(*zip(*v)) for k, v in performance_evaluation["roc_curve_train"].items()}
        performance_evaluation["auc_curve_test"] = {k: auc_curve_from_roc(*zip(*v)) for k, v in performance_evaluation["roc_curve_test"].items()}
        
        pr_info = {
            "pr_curve_train": multiclass_curve_wrapper(precision_recall_curve, y_true=self.y_train, y_score=y_score_train, sample_weight=self.weight_train),
            "pr_curve_test": multiclass_curve_wrapper(precision_recall_curve, y_true=self.y_test, y_score=y_score_test, sample_weight=self.weight_test),
            "pr_score_train": pd.merge(multiclass_score_wrapper(average_precision_score, y_true=self.y_train, y_score=y_score_train, sample_weight=self.weight_train), table_train[["precision", "recall"]], left_index=True, right_index=True),
            "pr_score_test": pd.merge(multiclass_score_wrapper(average_precision_score, y_true=self.y_test, y_score=y_score_test, sample_weight=self.weight_test), table_test[["precision", "recall"]], left_index=True, right_index=True)
        }
        
        base_info = pd.concat([table_train.loc[["weighted"], ["accuracy", "precision", "recall", "f1_score"]], table_test.loc[["weighted"], ["accuracy", "precision", "recall", "f1_score"]]], axis=0, ignore_index=True)
        base_info["auc"] = [performance_evaluation["roc_score_train"].loc["weighted", "score"], performance_evaluation["roc_score_test"].loc["weighted", "score"]]
        base_info["factor_cnt"] = [self.weight_train.sum(), self.weight_test.sum()]
        base_info["set"] = ["train", "test"]
        
        correlation_df = correlation(features, y, weight)
        
        res = {
            "base_info": base_info,
            "confusion_matrix": confusion_mtx,
            "prediction_evaluation": prediction_evaluation,
            "performance_evaluation": performance_evaluation,
            "PR_info": pr_info,
            "correlation": correlation_df
        }
        
        return res

    
class RegressionOutputGenerator(BaseOutputGenerator):
    def __init__(self):
        super().__init__()

    def output(self):
        self._set_data()
        y_pred_train = self.engine.model.predict(self.X_train)
        y_pred_test = self.engine.model.predict(self.X_test)



        # 组装base_info DataFrame
        base_info = pd.DataFrame({
            "train": [self.y_train, y_pred_train],
            "test": [self.y_test, y_pred_test]
        })

        # 其他指标（如残差分析、变量重要性等）根据需要添加

        # 返回结果字典
        res = {
            "base_info": base_info
            # 添加其他需要的输出信息
        }
        return res
