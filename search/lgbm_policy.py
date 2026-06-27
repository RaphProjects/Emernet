import numpy as np

MUT_TYPES = ["add_node", "remove_node", "replace_node", "add_edge", "remove_edge"]
MOD_TYPES = ["EinsteinAggregator", "Activation", "LearnableParameter", "Constant", "Noise"]
NUM_CATEGORICAL = len(MUT_TYPES) + len(MOD_TYPES)


class LGBMPolicy:
    def __init__(self):
        self.model = None
        self.feature_names = None

    def _validate_imports(self):
        try:
            import lightgbm as lgb
            return lgb
        except ImportError:
            raise ImportError(
                "lightgbm is required for Phase B. Install it with: pip install lightgbm"
            )

    def extract_feature_vector(self, h_graph: np.ndarray, mutation_info: dict) -> np.ndarray:
        return h_graph.flatten().astype(np.float32)

    def train(self, X: np.ndarray, y: np.ndarray, h_graph_dim=384):
        lgb = self._validate_imports()
        from sklearn.multioutput import MultiOutputRegressor

        self.model = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            verbose=-1,
            random_state=42,
        )
        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 2 and y.shape[1] > 1:
            self.model = MultiOutputRegressor(self.model)
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("LGBM model not trained yet")
        return self.model.predict(X)

    def feature_importance(self, h_graph_dim=384) -> dict:
        if self.model is None:
            return {}
        names = self._feature_names(h_graph_dim)
        if hasattr(self.model, "estimators_"):
            importances = np.mean([est.feature_importances_ for est in self.model.estimators_], axis=0)
        else:
            importances = self.model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        return {
            "feature_names": [names[i] for i in sorted_idx],
            "importances": [float(importances[i]) for i in sorted_idx],
        }

    def _feature_names(self, h_graph_dim=384):
        names = [f"h_graph_{i}" for i in range(h_graph_dim)]
        return names

    def shap_explain(self, X: np.ndarray, h_graph_dim=384):
        try:
            import shap
        except ImportError:
            raise ImportError("shap is required for SHAP analysis. Install with: pip install shap")

        if self.model is None:
            raise RuntimeError("LGBM model not trained yet")

        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X)

        feature_names = self._feature_names(h_graph_dim)
        return {
            "shap_values": shap_values,
            "feature_names": feature_names,
        }
