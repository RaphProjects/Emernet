import hashlib

import numpy as np
import torch
import torch.nn as nn

from search.lgbm_policy import LGBMPolicy


class MutationNNPolicy(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=192, output_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x):
        if hasattr(self, "net"):
            return self.net(x)
        return self.linear(x)


class MetaScoreLayer(nn.Module):
    def __init__(self, input_dim=67, hidden_dim=48, output_dim=3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.03),
            nn.Linear(hidden_dim, output_dim),
        )
        self.output = self.net[-1]

    def forward(self, x):
        return self.net(x)


class EvolutionPolicy:
    def __init__(
        self,
        feature_dim=384,
        lgbm_policy=None,
        nn_policy=None,
        meta_layer=None,
        records_seen=0,
        replay_X=None,
        replay_y=None,
        replay_components=None,
        replay_keys=None,
    ):
        self.feature_dim = feature_dim
        self.lgbm_policy = lgbm_policy if lgbm_policy is not None else LGBMPolicy()
        self.nn_policy = nn_policy if nn_policy is not None else MutationNNPolicy(input_dim=feature_dim)
        self.meta_layer = meta_layer if meta_layer is not None else MetaScoreLayer()
        self.records_seen = records_seen
        self.latent_dim = getattr(self.nn_policy, "output_dim", 64)
        self.nn_trained = False
        self.meta_trained = False
        self.replay_X = list(replay_X) if replay_X is not None else []
        self.replay_y = list(replay_y) if replay_y is not None else []
        self.replay_components = list(replay_components) if replay_components is not None else []
        self.replay_keys = set(replay_keys) if replay_keys is not None else set()

    def _ensure_replay_fields(self):
        if not hasattr(self, "replay_X"):
            self.replay_X = []
        if not hasattr(self, "replay_y"):
            self.replay_y = []
        if not hasattr(self, "replay_components"):
            self.replay_components = []
        if len(self.replay_components) != len(self.replay_y):
            y = np.asarray(self.replay_y, dtype=np.float32)
            fallback = np.stack([y, y, np.zeros_like(y), np.zeros_like(y)], axis=1) if len(y) else []
            self.replay_components = [np.asarray(row, dtype=np.float32).copy() for row in fallback]
        if not hasattr(self, "replay_keys"):
            self.replay_keys = set()
        if len(self.replay_keys) != len(self.replay_y):
            self.replay_keys = {
                self._record_key(x, y, c)
                for x, y, c in zip(self.replay_X, self.replay_y, self.replay_components)
            }

    def _reset_replay(self):
        self.replay_X = []
        self.replay_y = []
        self.replay_components = []
        self.replay_keys = set()
        self.records_seen = 0

    def _record_key(self, x, y, components):
        h = hashlib.sha1()
        h.update(np.round(np.asarray(x, dtype=np.float32), 6).tobytes())
        h.update(np.round(np.asarray([y], dtype=np.float32), 6).tobytes())
        h.update(np.round(np.asarray(components, dtype=np.float32), 6).tobytes())
        return h.hexdigest()

    def _append_replay(self, X, y, y_components):
        self._ensure_replay_fields()
        added = 0
        for x_row, y_value, component_row in zip(X, y, y_components):
            key = self._record_key(x_row, y_value, component_row)
            if key in self.replay_keys:
                continue
            self.replay_keys.add(key)
            self.replay_X.append(np.asarray(x_row, dtype=np.float32).copy())
            self.replay_y.append(float(y_value))
            self.replay_components.append(np.asarray(component_row, dtype=np.float32).copy())
            added += 1
        return added

    def _replay_arrays(self):
        self._ensure_replay_fields()
        if not self.replay_y:
            return None, None, None
        X = np.asarray(self.replay_X, dtype=np.float32)
        y = np.asarray(self.replay_y, dtype=np.float32)
        y_components = np.asarray(self.replay_components, dtype=np.float32)
        if y_components.ndim != 2 or y_components.shape[1] != 4:
            y_components = np.stack([y, y, np.zeros_like(y), np.zeros_like(y)], axis=1)
            self.replay_components = [row.astype(np.float32).copy() for row in y_components]
            self.replay_keys = {
                self._record_key(x, target, components)
                for x, target, components in zip(self.replay_X, self.replay_y, self.replay_components)
            }
        return X, y, y_components

    def ensure_feature_dim(self, feature_dim):
        feature_dim = int(feature_dim)
        meta_out = getattr(
            self.meta_layer,
            "output_dim",
            getattr(getattr(self.meta_layer, "linear", getattr(self.meta_layer, "output", None)), "out_features", 1),
        )
        if feature_dim == self.feature_dim and getattr(self.nn_policy, "output_dim", 0) == self.latent_dim and meta_out == 3:
            return
        self.feature_dim = feature_dim
        self.lgbm_policy = LGBMPolicy()
        self.latent_dim = 64
        self.nn_policy = MutationNNPolicy(input_dim=feature_dim, output_dim=self.latent_dim)
        self.meta_layer = MetaScoreLayer(input_dim=3 + self.latent_dim, output_dim=3)
        self.nn_trained = False
        self.meta_trained = False
        self._ensure_replay_fields()
        migrated = []
        for row in self.replay_X:
            arr = np.asarray(row, dtype=np.float32).reshape(-1)
            if arr.shape[0] >= feature_dim:
                migrated.append(arr[:feature_dim].copy())
        if len(migrated) == len(self.replay_y):
            self.replay_X = migrated
            self.replay_keys = {
                self._record_key(x, y, c)
                for x, y, c in zip(self.replay_X, self.replay_y, self.replay_components)
            }
            self.records_seen = len(self.replay_y)
        else:
            self._reset_replay()

    def _lgbm_preds(self, X):
        if self.lgbm_policy is None or self.lgbm_policy.model is None:
            return np.full((len(X), 3), np.nan, dtype=np.float32)
        try:
            preds = np.asarray(self.lgbm_policy.predict(X), dtype=np.float32)
            if preds.ndim == 1:
                preds = np.stack([preds, np.zeros_like(preds), np.zeros_like(preds)], axis=1)
            if preds.shape[1] < 3:
                pad = np.zeros((len(preds), 3 - preds.shape[1]), dtype=np.float32)
                preds = np.concatenate([preds, pad], axis=1)
            return preds[:, :3]
        except Exception:
            return np.full((len(X), 3), np.nan, dtype=np.float32)

    def _nn_latent(self, X):
        if not self.nn_trained:
            return np.full((len(X), self.latent_dim), np.nan, dtype=np.float32)
        self.nn_policy.eval()
        with torch.no_grad():
            device = next(self.nn_policy.parameters()).device
            xt = torch.tensor(X, dtype=torch.float32, device=device)
            return self.nn_policy(xt).detach().cpu().numpy().astype(np.float32)

    def predict_components(self, X, current_score=None, best_score=None, speed_bal=0.3, opp_simp_bal=0.0):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        self.ensure_feature_dim(X.shape[1])
        fallback_final = np.full((len(X),), float(best_score if best_score is not None else current_score or 0.0), dtype=np.float32)
        lgbm_metrics = self._lgbm_preds(X)
        metrics = self._predict_metrics(X)
        learn = metrics[:, 0]
        speed = metrics[:, 1]
        opp = metrics[:, 2]
        final = self.score_from_components(learn, speed, opp, speed_bal=speed_bal, opp_simp_bal=opp_simp_bal)
        return {
            "final": final,
            "learnability": learn.astype(np.float32),
            "speed": speed.astype(np.float32),
            "opp_simp_raw": opp.astype(np.float32),
            "opp_simp_bonus": opp.astype(np.float32),
            "lgbm_metrics": lgbm_metrics,
            "fallback_final": fallback_final,
        }

    def predict(self, X, current_score=None, best_score=None, speed_bal=0.3, opp_simp_bal=0.0):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        self.ensure_feature_dim(X.shape[1])
        metrics = self._predict_metrics(X)
        return self.score_from_components(metrics[:, 0], metrics[:, 1], metrics[:, 2], speed_bal=speed_bal, opp_simp_bal=opp_simp_bal)

    def _meta_features(self, lgbm_metrics, latent):
        lgbm_f = np.where(np.isfinite(lgbm_metrics), lgbm_metrics, 0.0)
        latent_f = np.where(np.isfinite(latent), latent, 0.0)
        return np.concatenate([lgbm_f, latent_f], axis=1).astype(np.float32)

    def _predict_metrics(self, X):
        lgbm_metrics = self._lgbm_preds(X)
        latent = self._nn_latent(X)
        if self.meta_trained and np.isfinite(latent).all():
            meta = self._meta_features(lgbm_metrics, latent)
            self.meta_layer.eval()
            with torch.no_grad():
                device = next(self.meta_layer.parameters()).device
                return self.meta_layer(torch.tensor(meta, dtype=torch.float32, device=device)).detach().cpu().numpy().astype(np.float32)
        fallback = np.where(np.isfinite(lgbm_metrics), lgbm_metrics, 0.0)
        return fallback.astype(np.float32)

    def score_from_components(self, learnability, speed, opp_raw, speed_bal=0.3, opp_simp_bal=0.0):
        weighted_learnability = np.asarray(learnability, dtype=np.float32) + float(opp_simp_bal) * np.asarray(opp_raw, dtype=np.float32)
        return (weighted_learnability * (1.0 - float(speed_bal)) + np.asarray(speed, dtype=np.float32) * float(speed_bal)).astype(np.float32)

    def train(self, X, y, y_components=None, current_score=None, best_score=None, nn_epochs=80, meta_epochs=120, lr=1e-3):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        self.ensure_feature_dim(X.shape[1])
        if y_components is None:
            y_components = np.stack([y, y, np.zeros_like(y), np.zeros_like(y)], axis=1)
        y_components = np.asarray(y_components, dtype=np.float32)
        if y_components.ndim == 1:
            y_components = y_components.reshape(1, -1)

        self._append_replay(X, y, y_components)
        X, y, y_components = self._replay_arrays()
        if X is None or y is None or y_components is None:
            self.records_seen = 0
            return self

        self.records_seen = int(len(y))
        y_metrics = y_components[:, 1:4]
        if len(y) >= 2:
            try:
                self.lgbm_policy.train(X, y_metrics)
            except Exception:
                pass

        if len(y) >= 2:
            self.nn_policy.train()
            self.meta_layer.train()
            device = next(self.nn_policy.parameters()).device
            self.meta_layer.to(device)
            opt = torch.optim.Adam(self.nn_policy.parameters(), lr=lr, weight_decay=1e-4)
            opt_meta = torch.optim.Adam(self.meta_layer.parameters(), lr=lr, weight_decay=1e-4)
            xt = torch.tensor(X, dtype=torch.float32, device=device)
            yt = torch.tensor(y_metrics, dtype=torch.float32, device=device)
            lgbm_pred_values = self._lgbm_preds(X)
            lgbm_metrics = torch.tensor(np.where(np.isfinite(lgbm_pred_values), lgbm_pred_values, 0.0), dtype=torch.float32, device=device)
            for _ in range(nn_epochs):
                opt.zero_grad()
                opt_meta.zero_grad()
                latent = self.nn_policy(xt)
                meta_x = torch.cat([lgbm_metrics, latent], dim=1)
                loss = nn.functional.mse_loss(self.meta_layer(meta_x), yt)
                loss.backward()
                opt.step()
                opt_meta.step()
            self.nn_trained = True
            self.meta_trained = True

        if len(y) >= 4:
            lgbm_metrics = self._lgbm_preds(X)
            latent = self._nn_latent(X)
            device = next(self.meta_layer.parameters()).device
            meta_x = torch.tensor(self._meta_features(lgbm_metrics, latent), dtype=torch.float32, device=device)
            meta_y = torch.tensor(y_metrics, dtype=torch.float32, device=device)
            self.meta_layer.train()
            opt = torch.optim.Adam(self.meta_layer.parameters(), lr=lr)
            for _ in range(meta_epochs):
                opt.zero_grad()
                loss = nn.functional.mse_loss(self.meta_layer(meta_x), meta_y)
                loss.backward()
                opt.step()
            self.meta_trained = True
        return self
