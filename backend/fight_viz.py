import copy

from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import numpy as np
import math

def get_valid_target(arch, input_tensor, n_samples, device, max_attempts=5):
    from graph.executor import Executor
    import copy

    for attempt in range(max_attempts):
        arch_copy = copy.deepcopy(arch)
        arch_copy.reset_state()
        ex = Executor(arch_copy).to(device)

        with torch.no_grad():
            ex.forward(input_tensor)  # trigger lazy init

        ex.randomize_weights()

        with torch.no_grad():
            out = ex.forward(input_tensor)[0]
            std, mean = torch.std_mean(out, dim=0, keepdim=True)
            out = (out - mean) / (std + 1e-6)

        flat = out.reshape(n_samples, -1).detach().cpu().numpy()

        if not np.isfinite(flat).all():
            print(f"  Target attempt {attempt+1}: NaN, retrying...")
            continue

        variance = flat.var(axis=0).mean()
        if variance < 1e-6:
            print(f"  Target attempt {attempt+1}: flat (var={variance:.2e}), retrying...")
            continue

        # Looks good
        return ex, out, flat

    return None, None, None  # Failed all attempts

def run_fight_visualization(
    arch_a, arch_b,
    n_samples=200, n_snapshots=20,
    max_iter=300, lr=1e-3, max_retries=3
    ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from graph.executor import Executor
    import copy

    x = torch.linspace(-3, 3, n_samples, device=device)
    input_tensor = x.reshape(-1, 1, 1).to(device)

    # Generate valid targets ──
    _, target_a, flat_a = get_valid_target(arch_a, input_tensor, n_samples, device)
    _, target_b, flat_b = get_valid_target(arch_b, input_tensor, n_samples, device)

    if target_a is None or target_b is None:
        raise RuntimeError("Could not generate valid (non-flat, finite) target functions.")

    # PCA
    def make_pca(flat):
        if flat.shape[1] == 1:
            result = flat.flatten()
            result = (result - result.mean()) / (result.std() + 1e-8)
            return None, result
        pca = PCA(n_components=1)
        pca.fit(flat)
        result = pca.transform(flat).flatten()
        result = (result - result.mean()) / (result.std() + 1e-8)
        return pca, result

    pca_a, target_a_1d = make_pca(flat_a)
    pca_b, target_b_1d = make_pca(flat_b)

    # Setup learners
    arch_a_learner = copy.deepcopy(arch_a)
    arch_b_learner = copy.deepcopy(arch_b)
    arch_a_learner.reset_state()
    arch_b_learner.reset_state()

    exec_a_learner = Executor(arch_a_learner).to(device)
    exec_b_learner = Executor(arch_b_learner).to(device)
    exec_a_learner.set_Output_Adapter(input_tensor, target_b.shape, force=True)
    exec_b_learner.set_Output_Adapter(input_tensor, target_a.shape, force=True)
    exec_a_learner.randomize_weights()
    exec_b_learner.randomize_weights()

    # Helpers 
    def project(executor, pca):
        with torch.no_grad():
            pred = executor.forward(input_tensor)
            flat = pred[0].reshape(n_samples, -1).cpu().numpy()
        if not np.isfinite(flat).all():
            return np.zeros(n_samples)
        if pca is not None:
            result = pca.transform(flat).flatten()
        else:
            result = flat.flatten()
        std = result.std()
        if std < 1e-8:
            return np.zeros(n_samples)
        return (result - result.mean()) / std

    snapshot_epochs = sorted(set(
        [0]
        + [int(i * (max_iter - 1) / max(n_snapshots - 1, 1)) for i in range(n_snapshots)]
        + [max_iter - 1]
    ))

    # Training 
    def train_one(executor, target, pca_target):
        loss_fn = torch.nn.MSELoss()

        for attempt in range(max_retries):
            if attempt > 0:
                executor.randomize_weights()

            opt = torch.optim.Adam(executor.parameters(), lr=lr)
            snapshots = []
            loss_history = []
            nan_detected = False

            for epoch in range(max_iter):
                if epoch in snapshot_epochs:
                    executor.eval()
                    pred_1d = project(executor, pca_target)
                    with torch.no_grad():
                        loss_val = loss_fn(executor.forward(input_tensor)[0], target).item()
                    snapshots.append({
                        "epoch": int(epoch),
                        "loss": float(loss_val) if math.isfinite(loss_val) else None,
                        "y": pred_1d.tolist(),
                    })

                executor.train()
                opt.zero_grad()
                pred = executor.forward(input_tensor)

                if not torch.isfinite(pred[0]).all():
                    nan_detected = True
                    break

                loss = loss_fn(pred[0], target)

                if not torch.isfinite(loss):
                    nan_detected = True
                    break

                loss.backward()
                torch.nn.utils.clip_grad_norm_(executor.parameters(), 1.0)
                opt.step()
                loss_history.append(loss.item())

            if not nan_detected:
                return snapshots, loss_history, False

        return snapshots, loss_history, True

    snapshots_a, loss_history_a, broken_a = train_one(exec_a_learner, target_b, pca_b)
    snapshots_b, loss_history_b, broken_b = train_one(exec_b_learner, target_a, pca_a)

    return {
        "x": x.tolist(),
        "fight_a": {
            "label": "A learning B",
            "target": target_b_1d.tolist(),
            "snapshots": snapshots_a,
            "loss_history": loss_history_a,
            "broken": broken_a,
        },
        "fight_b": {
            "label": "B learning A",
            "target": target_a_1d.tolist(),
            "snapshots": snapshots_b,
            "loss_history": loss_history_b,
            "broken": broken_b,
        },
    }