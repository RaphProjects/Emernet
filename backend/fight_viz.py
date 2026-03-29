import copy

from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import numpy as np
import math
import time

avg_learn = 0.2401 # estimated over 340 runs, error +- 0.0043
std_learn = 0.6251 # estimated over 340 runs, error +- 0.0067
avg_speed = -1.8010 # estimated over 340 runs, error +- 0.0093
std_speed = 1.2892 # estimated over 340 runs, error +- 0.0124
speed_bal = 0.3

def get_valid_target(arch, input_tensor, n_samples, device, max_attempts=5, generating=True):
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
        if variance < 1e-6 and generating:
            print(f"  Target attempt {attempt+1}: flat (var={variance:.2e}), retrying...")
            continue

        # Looks good
        return ex, out, flat

    return None, None, None  # Failed all attempts

def run_fight_visualization(
    arch_a, arch_b,
    n_samples=200, n_snapshots=50,
    max_iter=500, lr=1e-2, max_retries=3, generating_A=True, generating_B=True
    ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from graph.executor import Executor
    import copy

    x = torch.linspace(-3, 3, n_samples, device=device)
    input_tensor = x.reshape(-1, 1, 1).to(device)


    # Generate valid targets ──

    _, target_a, flat_a = get_valid_target(arch_a, input_tensor, n_samples, device, generating=generating_A)
    _, target_b, flat_b = get_valid_target(arch_b, input_tensor, n_samples, device, generating=generating_B)

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
        return result

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
            start_time = time.time()
            for epoch in range(max_iter):
                if epoch in snapshot_epochs:
                    executor.eval()
                    pred_1d = project(executor, pca_target)

                    if pred_1d is None: # Catch the NaN
                        nan_detected = True
                        break
                    with torch.no_grad():
                        loss_val = loss_fn(executor.forward(input_tensor)[0], target).item()

                    # DEBUG PRINT: Print the first 3 values of pred_1d
                    print(f"Snapshot epoch {epoch}, first 3 preds: {pred_1d[:3]}")
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

            fit_time = time.time() - start_time
            if not nan_detected:
                final_loss = loss_history[-1] if loss_history else 1e10

                learnability = math.log(math.sqrt(1.0 / max(final_loss, 1e-10)))
                speed = math.log(math.sqrt(1.0 / max(fit_time, 1e-10)))

                learnability = (learnability-avg_learn)/std_learn
                speed = (speed-avg_speed)/std_speed

                score = learnability*(1-speed_bal) + speed*speed_bal
                return snapshots, loss_history, False, fit_time, score
            
        return snapshots, loss_history, True

    snapshots_a, loss_history_a, broken_a, time_a, score_a = train_one(exec_a_learner, target_b, pca_b)
    snapshots_b, loss_history_b, broken_b, time_b, score_b = train_one(exec_b_learner, target_a, pca_a)

    return {
        "x": x.tolist(),
        "fight_a": {
            "label": "A learning B",
            "target": target_b_1d.tolist(),
            "snapshots": snapshots_a,
            "loss_history": loss_history_a,
            "broken": broken_a,
            "fit_time": time_a,   
            "score": score_a    
        },
        "fight_b": {
            "label": "B learning A",
            "target": target_a_1d.tolist(),
            "snapshots": snapshots_b,
            "loss_history": loss_history_b,
            "broken": broken_b,
            "fit_time": time_b, 
            "score": score_b  
        },
    }

def run_tournament_fight(arch_a, arch_b, max_iter=500, lr=5e-3, max_retries=2):
    """
    Simplified fight for tournaments — no snapshots, no PCA.
    Returns (details_a, details_b) dicts, or None on total failure.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from graph.executor import Executor

    n_samples = 200
    x = torch.linspace(-3, 3, n_samples, device=device)
    input_tensor = x.reshape(-1, 1, 1).to(device)

    _, target_a, _ = get_valid_target(arch_a, input_tensor, n_samples, device)
    _, target_b, _ = get_valid_target(arch_b, input_tensor, n_samples, device)

    if target_a is None or target_b is None:
        return None

    def train_and_score(arch, target):
        for attempt in range(max_retries):
            try:
                arch_copy = copy.deepcopy(arch)
                arch_copy.reset_state()
                ex = Executor(arch_copy).to(device)
                ex.set_Output_Adapter(input_tensor, target.shape, force=True)
                ex.randomize_weights()

                loss_fn = nn.MSELoss()
                opt = torch.optim.Adam(ex.parameters(), lr=lr)

                start_time = time.time()
                last_loss = 1e10
                nan_hit = False

                for _ in range(max_iter):
                    ex.train()
                    opt.zero_grad()
                    pred = ex.forward(input_tensor)
                    if not torch.isfinite(pred[0]).all():
                        nan_hit = True; break
                    loss = loss_fn(pred[0], target)
                    if not torch.isfinite(loss):
                        nan_hit = True; break
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(ex.parameters(), 1.0)
                    opt.step()
                    last_loss = loss.item()

                if nan_hit:
                    continue

                fit_time = time.time() - start_time
                learn = math.log(math.sqrt(1.0 / max(last_loss, 1e-10)))
                spd   = math.log(math.sqrt(1.0 / max(fit_time, 1e-10)))
                learn_z = (learn - avg_learn) / std_learn
                spd_z   = (spd   - avg_speed) / std_speed
                score   = learn_z * (1 - speed_bal) + spd_z * speed_bal

                return {
                    "score":        score,
                    "learnability": learn_z,
                    "speed":        spd_z,
                    "fit_time":     fit_time,
                    "final_loss":   last_loss,
                }

            except Exception as e:
                print(f"  Tournament train attempt {attempt+1} failed: {e}")
                continue
        return None

    ra = train_and_score(arch_a, target_b)
    rb = train_and_score(arch_b, target_a)

    if ra is None or rb is None:
        return None
    return (ra, rb)