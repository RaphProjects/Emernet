import copy
import math
import random
import time
import torch
import networkx
import numpy as np

from graph.architecture import Architecture
from graph.generator import Generator
from graph.mutator import Mutator
from modules import UNIFIED_PRESET

from search.encoder import (
    ArchEncoder, RewardPredictor, build_node_features, NODE_FEATURE_DIM,
    topological_order, predecessors_in_order, encode_architecture, batch_encode, train_gnn
)
from search.data_collector import propose_random_mutations
from search.lgbm_policy import LGBMPolicy
from search.evolution_policy import EvolutionPolicy


GNN_EMBEDDING_DIM = 384


class RLSearch:
    def __init__(self, arena, module_set="Unified", verbose=True, device=None):
        self.arena = arena
        self.module_set = module_set
        self.verbose = verbose
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.generator = Generator(
            generation_type=arena.generation_type,
            module_types=module_set
        )

    def log(self, msg):
        if self.verbose:
            print(f"[RLSearch] {msg}")

    def _compute_tournament_scores(self, n_archs, fight_counts, log_scores,
                                   raw_learn_sum, raw_speed_sum, raw_time_sum,
                                   simp_opp_bal):
        simps = []
        for col in range(n_archs):
            vals = [
                log_scores[row][col]
                for row in range(n_archs)
                if row != col and log_scores[row][col] != 0.0
            ]
            col_mean = sum(vals) / len(vals) if vals else self.arena.avg_simp
            simps.append((col_mean - self.arena.avg_simp) / self.arena.std_simp)

        scores, learns, speeds, times, raw_learns, opp_bonuses, opp_raw_values = [], [], [], [], [], [], []
        for k in range(n_archs):
            fc = max(fight_counts[k], 1)
            avg_speed = raw_speed_sum[k] / fc
            norm_speed = (avg_speed - self.arena.avg_speed) / self.arena.std_speed
            avg_learn = raw_learn_sum[k] / fc
            raw_norm_learn = (avg_learn - self.arena.avg_learn) / self.arena.std_learn
            total_w = 0.0
            w_count = 0
            for col in range(n_archs):
                if col != k and log_scores[k][col] != 0.0:
                    if simp_opp_bal > 0:
                        total_w += log_scores[k][col] * math.exp(simps[col] * simp_opp_bal)
                    w_count += 1
            if simp_opp_bal > 0:
                weighted_norm_learn = ((total_w / max(w_count, 1)) - self.arena.avg_learn) / self.arena.std_learn
            else:
                weighted_norm_learn = raw_norm_learn
            opp_raw = simps[k]
            opp_bonus = weighted_norm_learn - raw_norm_learn
            combined_learn = raw_norm_learn + float(simp_opp_bal) * opp_raw
            combined = combined_learn * (1.0 - self.arena.speed_bal) + norm_speed * self.arena.speed_bal
            scores.append(combined)
            learns.append(raw_norm_learn)
            speeds.append(norm_speed)
            times.append(raw_time_sum[k] / fc)
            raw_learns.append(raw_norm_learn)
            opp_bonuses.append(opp_bonus)
            opp_raw_values.append(opp_raw)
        return scores, learns, speeds, times, raw_learns, opp_bonuses, opp_raw_values

    def run_full_tournament(self, architectures, labels=None, progress_callback=None,
                            context=None, simp_opp_bal=0.2, max_duration=600):
        context = context or {}
        labels = labels or [f"Arch {i}" for i in range(len(architectures))]
        n_archs = len(architectures)
        total = n_archs * (n_archs - 1) // 2
        arch_payload = []
        for idx, arch in enumerate(architectures):
            arch_payload.append({
                "id": idx,
                "name": labels[idx],
                "n_nodes": len(arch.nodes),
                "n_params": arch.parameter_count(),
                "graph": self._arch_to_brief_json(arch),
            })

        if progress_callback is not None:
            progress_callback({
                "type": "rl_tournament_init",
                "architectures": arch_payload,
                "n_archs": n_archs,
                "total_fights": total,
                "simp_opp_bal": simp_opp_bal,
                **context,
            })

        if n_archs < 2:
            return {
                "scores": [0.0] * n_archs,
                "learnabilities": [0.0] * n_archs,
                "speeds": [0.0] * n_archs,
                "raw_learnabilities": [0.0] * n_archs,
                "opp_simp_bonuses": [0.0] * n_archs,
                "opp_simp_raw_bonuses": [0.0] * n_archs,
                "fit_times": [0.0] * n_archs,
                "fight_counts": [0] * n_archs,
            }

        log_scores = [[0.0] * n_archs for _ in range(n_archs)]
        raw_learn_sum = [0.0] * n_archs
        raw_speed_sum = [0.0] * n_archs
        raw_time_sum = [0.0] * n_archs
        fight_counts = [0] * n_archs
        fight = 0

        for i in range(n_archs):
            for j in range(i + 1, n_archs):
                fight += 1
                failed = False
                try:
                    started = time.time()
                    score_i, score_j, delay_i, delay_j = self.arena.get_scores(
                        architectures[i], architectures[j],
                        randomizeHP=True, pcp=0, get_delays=True,
                        max_duration=max_duration,
                    )
                    if time.time() - started > max_duration * 0.9:
                        failed = True
                except Exception as exc:
                    score_i, score_j = 1e-5, 1e-5
                    delay_i, delay_j = 10.0, 10.0
                    failed = True
                    error = str(exc)
                else:
                    error = None

                log_learn_i = math.log(max(score_i, 1e-10))
                log_learn_j = math.log(max(score_j, 1e-10))
                log_speed_i = math.log(max(1.0 / max(delay_i, 1e-6), 1e-10))
                log_speed_j = math.log(max(1.0 / max(delay_j, 1e-6), 1e-10))

                log_scores[i][j] = log_learn_i
                log_scores[j][i] = log_learn_j
                raw_learn_sum[i] += log_learn_i
                raw_learn_sum[j] += log_learn_j
                raw_speed_sum[i] += log_speed_i
                raw_speed_sum[j] += log_speed_j
                raw_time_sum[i] += delay_i
                raw_time_sum[j] += delay_j
                fight_counts[i] += 1
                fight_counts[j] += 1

                scores, learns, speeds, times, raw_learns, opp_bonuses, opp_raw_bonuses = self._compute_tournament_scores(
                    n_archs, fight_counts, log_scores, raw_learn_sum,
                    raw_speed_sum, raw_time_sum, simp_opp_bal
                )

                fight_learn_i = (log_learn_i - self.arena.avg_learn) / self.arena.std_learn
                fight_learn_j = (log_learn_j - self.arena.avg_learn) / self.arena.std_learn
                fight_speed_i = (log_speed_i - self.arena.avg_speed) / self.arena.std_speed
                fight_speed_j = (log_speed_j - self.arena.avg_speed) / self.arena.std_speed
                fight_score_i = fight_learn_i * (1.0 - self.arena.speed_bal) + fight_speed_i * self.arena.speed_bal
                fight_score_j = fight_learn_j * (1.0 - self.arena.speed_bal) + fight_speed_j * self.arena.speed_bal

                if progress_callback is not None:
                    progress_callback({
                        "type": "rl_tournament_fight",
                        "fight": fight,
                        "total": total,
                        "i": i,
                        "j": j,
                        "failed": failed,
                        "error": error,
                        "score_i": float(fight_score_i),
                        "score_j": float(fight_score_j),
                        "learn_i": float(fight_learn_i),
                        "learn_j": float(fight_learn_j),
                        "speed_i": float(fight_speed_i),
                        "speed_j": float(fight_speed_j),
                        "time_i": float(delay_i),
                        "time_j": float(delay_j),
                        "scores": [float(s) for s in scores],
                        "learnabilities": [float(s) for s in learns],
                        "speeds": [float(s) for s in speeds],
                        "raw_learnabilities": [float(s) for s in raw_learns],
                        "opp_simp_bonuses": [float(s) for s in opp_bonuses],
                        "opp_simp_raw_bonuses": [float(s) for s in opp_raw_bonuses],
                        "fit_times": [float(t) for t in times],
                        "fight_counts": fight_counts[:],
                        **context,
                    })

        final_scores, final_learns, final_speeds, final_times, final_raw_learns, final_opp_bonuses, final_opp_raw_bonuses = self._compute_tournament_scores(
            n_archs, fight_counts, log_scores, raw_learn_sum,
            raw_speed_sum, raw_time_sum, simp_opp_bal
        )
        result = {
            "scores": [float(s) for s in final_scores],
            "learnabilities": [float(s) for s in final_learns],
            "speeds": [float(s) for s in final_speeds],
            "raw_learnabilities": [float(s) for s in final_raw_learns],
            "opp_simp_bonuses": [float(s) for s in final_opp_bonuses],
            "opp_simp_raw_bonuses": [float(s) for s in final_opp_raw_bonuses],
            "fit_times": [float(t) for t in final_times],
            "fight_counts": fight_counts[:],
        }
        if progress_callback is not None:
            progress_callback({
                "type": "rl_tournament_done",
                "final_scores": result["scores"],
                "learnabilities": result["learnabilities"],
                "speeds": result["speeds"],
                "raw_learnabilities": result["raw_learnabilities"],
                "opp_simp_bonuses": result["opp_simp_bonuses"],
                "opp_simp_raw_bonuses": result["opp_simp_raw_bonuses"],
                "fit_times": result["fit_times"],
                "fight_counts": result["fight_counts"],
                **context,
            })
        return result

    def evaluate_architecture(self, arch: Architecture, n_samples=1, progress_callback=None, context=None) -> float:
        context = context or {}
        total = 0.0
        if progress_callback is not None:
            try:
                progress_callback({
                    "type": "arena_eval_start",
                    "n_samples": n_samples,
                    "n_nodes": len(arch.nodes),
                    "n_params": arch.parameter_count(),
                    **context,
                })
            except Exception:
                pass
        for sample_idx in range(1, n_samples + 1):
            opponent = self.generator.generate(self.arena.architecture_size)
            try:
                score_arch, score_opp = self.arena.get_scores(
                    arch, opponent,
                    randomizeHP=True,
                    pcp=0,
                )
                lr = math.log(max(score_arch, 1e-10))
                sr = math.log(max(score_opp, 1e-10))
                contribution = lr + sr
                total += contribution
                if progress_callback is not None:
                    try:
                        progress_callback({
                            "type": "arena_fight_result",
                            "sample": sample_idx,
                            "n_samples": n_samples,
                            "score_arch": float(score_arch),
                            "score_opponent": float(score_opp),
                            "log_score_arch": float(lr),
                            "log_score_opponent": float(sr),
                            "reward_contribution": float(contribution),
                            "opponent_nodes": len(opponent.nodes),
                            "opponent_params": opponent.parameter_count(),
                            **context,
                        })
                    except Exception:
                        pass
            except Exception as exc:
                total += -40.0
                if progress_callback is not None:
                    try:
                        progress_callback({
                            "type": "arena_fight_result",
                            "sample": sample_idx,
                            "n_samples": n_samples,
                            "failed": True,
                            "error": str(exc),
                            "reward_contribution": -40.0,
                            **context,
                        })
                    except Exception:
                        pass
        avg = total / n_samples
        if progress_callback is not None:
            try:
                progress_callback({
                    "type": "arena_eval_done",
                    "n_samples": n_samples,
                    "average_reward": float(avg),
                    **context,
                })
            except Exception:
                pass
        return avg

    def _arch_to_brief_json(self, arch: Architecture) -> dict:
        import networkx as nx
        topo = list(nx.topological_sort(arch))
        depth = {}
        for node in topo:
            preds = list(arch.predecessors(node))
            if not preds:
                depth[node] = 0
            else:
                depth[node] = max(depth[p] for p in preds) + 1
        layers: dict[int, list] = {}
        for node, d in depth.items():
            layers.setdefault(d, []).append(node)
        x_spacing, y_spacing = 160, 90
        nodes = []
        for d, layer_nodes in layers.items():
            n = len(layer_nodes)
            total_width = (n - 1) * x_spacing
            start_x = -total_width / 2
            for i, node in enumerate(layer_nodes):
                mod = arch.nodes[node]['module']
                nodes.append({
                    "id": str(node),
                    "type": mod.__class__.__name__,
                    "module_type": mod.module_type.name,
                    "x": start_x + i * x_spacing + 500,
                    "y": d * y_spacing + 60,
                })
        edges = [{"source": str(u), "target": str(v)} for u, v in arch.edges]
        return {"nodes": nodes, "edges": edges}

    def phase_a(self, n_episodes=500, train_frequency=50, n_gnn_epochs=100,
                lr=0.001, progress_callback=None) -> tuple[Architecture, ArchEncoder, RewardPredictor, list]:
        self.log("Phase A: GNN + small MLP supervised pre-training")

        encoder = ArchEncoder(NODE_FEATURE_DIM, hidden_dim=128, n_layers=4).to(self.device)
        predictor = RewardPredictor(input_dim=GNN_EMBEDDING_DIM).to(self.device)

        buffer_archs = []
        buffer_rewards = []
        best_arch = None
        best_reward = -float('inf')
        history = []
        invalid_skips = 0
        evaluated = 0

        for episode in range(1, n_episodes + 1):
            if random.random() < 0.2 and best_arch is not None:
                mutator = Mutator(copy.deepcopy(best_arch))
                try:
                    mt = random.choice(UNIFIED_PRESET)
                    op = random.choice(["add_node", "replace_node", "add_edge"])
                    if op == "add_node":
                        mutator.add_node(mt)
                    elif op == "replace_node":
                        valid = [n for n in mutator.arch.nodes if n != 0]
                        mutator.replace_node(random.choice(valid), mt)
                    elif op == "add_edge":
                        sources = list(mutator.arch.nodes)
                        targets = [n for n in sources if n != 0]
                        random.shuffle(sources)
                        random.shuffle(targets)
                        for s in sources:
                            for t in targets:
                                if mutator.validator.can_add_edge(s, t):
                                    mutator.add_edge(s, t)
                                    break
                            else:
                                continue
                            break
                    arch = mutator.arch
                except Exception:
                    arch = self.generator.generate(self.arena.architecture_size)
            else:
                arch = self.generator.generate(self.arena.architecture_size)

            if not arch.isValid(verbose=False):
                invalid_skips += 1
                arch = self.generator.generate(self.arena.architecture_size)
                if not arch.isValid(verbose=False):
                    invalid_skips += 1
                    if progress_callback is not None:
                        try:
                            progress_callback({
                                "type": "rl_diagnostic",
                                "phase": "A",
                                "episode": episode,
                                "level": "warning",
                                "message": "Skipped invalid architecture after generation fallback.",
                                "invalid_skips": invalid_skips,
                            })
                        except Exception:
                            pass
                    continue

            reward = self.evaluate_architecture(
                arch,
                n_samples=2,
                progress_callback=progress_callback,
                context={"phase": "A", "episode": episode, "eval_role": "candidate"},
            )
            evaluated += 1
            buffer_archs.append(arch)
            buffer_rewards.append(reward)

            if reward > best_reward:
                best_reward = reward
                best_arch = copy.deepcopy(arch)

            gnn_loss = None
            if episode % train_frequency == 0 and len(buffer_archs) >= 10:
                gnn_loss = train_gnn(
                    encoder, predictor, buffer_archs, buffer_rewards,
                    n_epochs=n_gnn_epochs, lr=lr, device=self.device
                )
                history.append({
                    "episode": episode,
                    "phase": "A",
                    "best_reward": best_reward,
                    "buffer_size": len(buffer_archs),
                    "gnn_loss": gnn_loss,
                })
                self.log(f"ep {episode}: best_reward={best_reward:.4f}, "
                         f"buffer={len(buffer_archs)}, gnn_loss={gnn_loss:.4f}")

            if progress_callback is not None:
                prog = {
                    "type": "phase_a_episode",
                    "episode": episode,
                    "n_total": n_episodes,
                    "best_reward": best_reward,
                    "gnn_loss": gnn_loss,
                    "reward": reward,
                    "buffer_size": len(buffer_archs),
                    "evaluated": evaluated,
                    "invalid_skips": invalid_skips,
                }
                if (episode == 1 or episode % 10 == 0) and best_arch is not None:
                    prog["best_arch_json"] = self._arch_to_brief_json(best_arch)
                try:
                    progress_callback(prog)
                except Exception:
                    pass

        if progress_callback is not None:
            try:
                progress_callback({
                    "type": "phase_a_done",
                    "n_episodes": n_episodes,
                    "best_reward": best_reward,
                })
            except Exception:
                pass

        return best_arch, encoder, predictor, history

    def train_arch_encoder(self, n_epochs=10, tournament_size=8, arch_size=None,
                           simp_opp_bal=0.2, n_gnn_epochs=100, lr=0.001,
                           progress_callback=None, should_stop=None):
        self.log("Train Arch Encoder: full-tournament supervised training")
        encoder = ArchEncoder(NODE_FEATURE_DIM, hidden_dim=128, n_layers=4).to(self.device)
        predictor = RewardPredictor(input_dim=GNN_EMBEDDING_DIM).to(self.device)
        history = []
        all_archs = []
        all_scores = []
        best_arch = None
        best_score = -float("inf")

        for epoch in range(1, n_epochs + 1):
            if should_stop is not None and should_stop():
                break
            archs = [self.generator.generate(arch_size or self.arena.architecture_size) for _ in range(tournament_size)]
            valid_archs = []
            labels = []
            for idx, arch in enumerate(archs):
                if arch.isValid(verbose=False):
                    valid_archs.append(arch)
                    labels.append(f"Epoch {epoch} Arch {idx + 1}")
            archs = valid_archs
            if len(archs) < 2:
                continue

            predictions = [self._predict_gnn_reward(encoder, predictor, arch) for arch in archs]
            tournament = self.run_full_tournament(
                archs,
                labels=labels,
                progress_callback=progress_callback,
                context={"phase": "encoder", "epoch": epoch},
                simp_opp_bal=simp_opp_bal,
            )
            if should_stop is not None and should_stop():
                break
            true_scores = tournament["scores"]
            errors = [
                abs(float(pred) - float(score))
                for pred, score in zip(predictions, true_scores)
                if pred is not None
            ]
            avg_delta = sum(errors) / len(errors) if errors else None

            all_archs.extend(copy.deepcopy(arch) for arch in archs)
            all_scores.extend(float(score) for score in true_scores)
            gnn_loss = train_gnn(
                encoder, predictor, all_archs, all_scores,
                n_epochs=n_gnn_epochs, lr=lr, device=self.device
            )

            epoch_best_idx = int(np.argmax(true_scores))
            if true_scores[epoch_best_idx] > best_score:
                best_score = float(true_scores[epoch_best_idx])
                best_arch = copy.deepcopy(archs[epoch_best_idx])

            row_details = []
            for idx, (arch, pred, score) in enumerate(zip(archs, predictions, true_scores)):
                row_details.append({
                    "id": idx,
                    "name": labels[idx],
                    "predicted_score": pred,
                    "true_score": float(score),
                    "abs_error": abs(float(pred) - float(score)) if pred is not None else None,
                    "n_nodes": len(arch.nodes),
                    "n_params": arch.parameter_count(),
                })

            history_row = {
                "epoch": epoch,
                "phase": "encoder",
                "avg_delta": avg_delta,
                "gnn_loss": float(gnn_loss),
                "best_score": best_score,
                "tournament_size": len(archs),
            }
            history.append(history_row)
            if progress_callback is not None:
                progress_callback({
                    "type": "encoder_epoch",
                    **history_row,
                    "architectures": row_details,
                    "best_arch_json": self._arch_to_brief_json(best_arch) if best_arch is not None else None,
                })

        if progress_callback is not None:
            progress_callback({
                "type": "encoder_done",
                "n_epochs": len(history),
                "best_score": best_score,
                "interrupted": should_stop() if should_stop is not None else False,
            })
        return {
            "best_architecture": best_arch,
            "best_reward": best_score,
            "training_history": history,
            "gnn_encoder": encoder,
            "gnn_predictor": predictor,
            "evolution_policy": EvolutionPolicy(),
            "interrupted": should_stop() if should_stop is not None else False,
        }

    def phase_b(self, encoder: ArchEncoder, predictor: RewardPredictor | None = None,
                n_episodes=500, n_candidates_per_step=10, retrain_frequency=50,
                initial_arch: Architecture | None = None, initial_lgbm_policy: LGBMPolicy | None = None,
                progress_callback=None) -> tuple[Architecture, LGBMPolicy, list]:
        self.log("Phase B: LGBM-guided search")

        encoder.eval()
        if predictor is not None:
            predictor.eval()
        lgbm_policy = initial_lgbm_policy if initial_lgbm_policy is not None else LGBMPolicy()

        arch = copy.deepcopy(initial_arch) if initial_arch is not None else self.generator.generate(self.arena.architecture_size)
        best_arch = copy.deepcopy(arch)
        best_reward = self.evaluate_architecture(
            best_arch,
            n_samples=1,
            progress_callback=progress_callback,
            context={"phase": "B", "episode": 0, "eval_role": "initial_base"},
        )

        records = []
        history = []
        top_architectures_by_signature = {}

        def architecture_signature(arch):
            return (
                tuple(
                    (
                        int(n),
                        type(arch.nodes[n]["module"]).__name__,
                        getattr(arch.nodes[n]["module"].module_type, "name", ""),
                        getattr(arch.nodes[n]["module"].mapping_type, "name", ""),
                    )
                    for n in sorted(arch.nodes)
                ),
                tuple(sorted((int(s), int(t)) for s, t in arch.edges)),
            )

        def ranked_top_architectures():
            ranked = sorted(
                top_architectures_by_signature.values(),
                key=lambda row: row["score"],
                reverse=True,
            )[:20]
            top_architectures_by_signature.clear()
            for row in ranked:
                top_architectures_by_signature[row["signature"]] = row
            return [
                {
                    **row,
                    "rank": rank,
                }
                for rank, row in enumerate(ranked, start=1)
            ]

        def add_tournament_top_architectures(tournament, architectures, epoch, focus_role):
            for idx, arch in enumerate(architectures):
                role = "evolved" if idx == 0 else "met"
                source = (
                    "initial" if epoch == 0 and idx == 0 else
                    f"initial opponent {idx}" if epoch == 0 else
                    f"epoch {epoch}" if idx == 0 else
                    f"epoch {epoch} opponent {idx}"
                )
                signature = architecture_signature(arch)
                score = float(tournament["scores"][idx])
                existing = top_architectures_by_signature.get(signature)
                if existing is not None and existing["score"] >= score:
                    continue
                top_architectures_by_signature[signature] = {
                    "signature": signature,
                    "role": role,
                    "source": source,
                    "epoch": int(epoch),
                    "eval_role": focus_role,
                    "score": score,
                    "learnability": float(tournament["learnabilities"][idx]),
                    "speed": float(tournament["speeds"][idx]),
                    "opp_simp_raw": float(tournament.get("opp_simp_raw_bonuses", tournament["opp_simp_bonuses"])[idx]),
                    "n_nodes": len(arch.nodes),
                    "n_params": arch.parameter_count(),
                    "architecture": copy.deepcopy(arch),
                    "arch_json": self._arch_to_brief_json(arch),
                }
            return ranked_top_architectures()

        def emit_top_architectures():
            if progress_callback is not None:
                progress_callback({
                    "type": "evolve_top_architectures",
                    "top_architectures": ranked_top_architectures(),
                })
        invalid_skips = 0
        empty_candidate_steps = 0

        for episode in range(1, n_episodes + 1):
            candidates = propose_random_mutations(arch, n_candidates=n_candidates_per_step)
            if progress_callback is not None:
                try:
                    progress_callback({
                        "type": "phase_b_candidates",
                        "episode": episode,
                        "n_total": n_episodes,
                        "n_candidates": len(candidates),
                    })
                except Exception:
                    pass

            predictions = None
            predicted_best_idx = None
            used_model = False
            if lgbm_policy.model is not None and candidates:
                try:
                    h_graph = self._get_embedding(encoder, arch)
                    X_candidates = np.array([
                        lgbm_policy.extract_feature_vector(h_graph, c)
                        for c in candidates
                    ])
                    predictions = lgbm_policy.predict(X_candidates)
                    predicted_best_idx = int(np.argmax(predictions))
                except Exception:
                    pass

            if predictions is not None and random.random() >= 0.15:
                best_idx = predicted_best_idx
                used_model = True
            else:
                best_idx = random.randrange(len(candidates)) if candidates else -1

            if best_idx < 0 or best_idx >= len(candidates):
                arch = self.generator.generate(self.arena.architecture_size)
                empty_candidate_steps += 1
                if progress_callback is not None:
                    try:
                        progress_callback({
                            "type": "phase_b_episode",
                            "episode": episode,
                            "n_total": n_episodes,
                            "skipped": True,
                            "skip_reason": "No valid mutation candidates; regenerated the working architecture.",
                            "delta_reward": 0.0,
                            "old_reward": None,
                            "new_reward": None,
                            "mutation_type": "regenerate",
                            "target_module_type": "",
                            "predicted_delta": None,
                            "used_model": False,
                            "n_nodes_before": len(best_arch.nodes) if best_arch is not None else 0,
                            "n_nodes_after": len(arch.nodes),
                            "n_records": len(records),
                            "lgbm_trained": False,
                            "top_predictions": [],
                            "invalid_skips": invalid_skips,
                            "empty_candidate_steps": empty_candidate_steps,
                        })
                    except Exception:
                        pass
                continue

            chosen = candidates[best_idx]
            mutated_arch = chosen["mutated_arch"]
            n_nodes_before = len(arch.nodes)

            if not mutated_arch.isValid(verbose=False):
                invalid_skips += 1
                if progress_callback is not None:
                    try:
                        progress_callback({
                            "type": "phase_b_episode",
                            "episode": episode,
                            "n_total": n_episodes,
                            "skipped": True,
                            "skip_reason": "Selected mutation produced an invalid architecture.",
                            "delta_reward": 0.0,
                            "old_reward": None,
                            "new_reward": None,
                            "mutation_type": chosen.get("mutation_type", "unknown"),
                            "target_module_type": chosen.get("target_module_type", ""),
                            "predicted_delta": None,
                            "used_model": used_model,
                            "n_nodes_before": n_nodes_before,
                            "n_nodes_after": len(mutated_arch.nodes),
                            "n_records": len(records),
                            "lgbm_trained": False,
                            "top_predictions": [],
                            "invalid_skips": invalid_skips,
                            "empty_candidate_steps": empty_candidate_steps,
                        })
                    except Exception:
                        pass
                continue

            old_reward = self.evaluate_architecture(
                arch,
                n_samples=2,
                progress_callback=progress_callback,
                context={"phase": "B", "episode": episode, "eval_role": "base"},
            )
            new_reward = self.evaluate_architecture(
                mutated_arch,
                n_samples=2,
                progress_callback=progress_callback,
                context={"phase": "B", "episode": episode, "eval_role": "mutated"},
            )
            delta = new_reward - old_reward

            chosen["delta_reward"] = delta
            records.append({
                "base_arch": copy.deepcopy(arch),
                "mutation": chosen,
            })

            if new_reward > best_reward:
                best_arch = copy.deepcopy(mutated_arch)
                best_reward = new_reward

            arch = mutated_arch

            lgbm_trained = False
            if episode % retrain_frequency == 0 and len(records) >= 10:
                lgbm_policy = self._train_lgbm(encoder, lgbm_policy, records)
                lgbm_trained = True

            history.append({
                "episode": episode,
                "phase": "B",
                "delta_reward": delta,
                "mutation_type": chosen["mutation_type"],
                "lgbm_trained": lgbm_trained,
            })

            if episode % 50 == 0:
                self.log(f"ep {episode}: delta={delta:.4f}, "
                         f"records={len(records)}")

            if progress_callback is not None:
                top_preds = []
                if predictions is not None and len(predictions) > 0:
                    sorted_indices = np.argsort(predictions)[::-1][:5]
                    for si in sorted_indices:
                        c = candidates[si]
                        top_preds.append({
                            "mutation_type": c["mutation_type"],
                            "target_module_type": c.get("target_module_type", ""),
                            "predicted_delta": float(predictions[si]),
                        })

                prog = {
                    "type": "phase_b_episode",
                    "episode": episode,
                    "n_total": n_episodes,
                    "delta_reward": delta,
                    "old_reward": float(old_reward),
                    "new_reward": float(new_reward),
                    "mutation_type": chosen["mutation_type"],
                    "target_module_type": chosen.get("target_module_type", ""),
                    "predicted_delta": float(predictions[best_idx]) if predictions is not None else None,
                    "used_model": used_model,
                    "n_nodes_before": n_nodes_before,
                    "n_nodes_after": len(mutated_arch.nodes),
                    "n_records": len(records),
                    "lgbm_trained": lgbm_trained,
                    "top_predictions": top_preds,
                    "invalid_skips": invalid_skips,
                    "empty_candidate_steps": empty_candidate_steps,
                    "version_score": self._build_version_score(
                        episode=episode,
                        mutation=chosen,
                        old_reward=old_reward,
                        new_reward=new_reward,
                        predicted_delta=float(predictions[best_idx]) if predictions is not None else None,
                        used_model=used_model,
                        base_arch=records[-1]["base_arch"],
                        mutated_arch=mutated_arch,
                        encoder=encoder,
                        predictor=predictor,
                    ),
                }
                if (episode == 1 or episode % 10 == 0) and best_arch is not None:
                    prog["best_arch_json"] = self._arch_to_brief_json(best_arch)
                try:
                    progress_callback(prog)
                except Exception:
                    pass

        if len(records) >= 10:
            lgbm_policy = self._train_lgbm(encoder, lgbm_policy, records)

        if progress_callback is not None:
            try:
                progress_callback({
                    "type": "phase_b_done",
                    "n_episodes": n_episodes,
                })
            except Exception:
                pass

        return best_arch, lgbm_policy, history

    def _get_embedding(self, encoder, arch):
        h = encode_architecture(arch, encoder, self.device)
        return h.detach().cpu().numpy()

    def _predict_gnn_reward(self, encoder, predictor, arch):
        if encoder is None or predictor is None:
            return None
        try:
            with torch.no_grad():
                h = encode_architecture(arch, encoder, self.device)
                return float(predictor(h).detach().cpu().item())
        except Exception:
            return None

    def _build_version_score(self, episode, mutation, old_reward, new_reward,
                             predicted_delta, used_model, base_arch, mutated_arch,
                             encoder=None, predictor=None):
        base_gnn = self._predict_gnn_reward(encoder, predictor, base_arch)
        mutated_gnn = self._predict_gnn_reward(encoder, predictor, mutated_arch)
        return {
            "episode": episode,
            "version": episode,
            "mutation_type": mutation.get("mutation_type", ""),
            "target_module_type": mutation.get("target_module_type", ""),
            "arena_old_reward": float(old_reward),
            "arena_new_reward": float(new_reward),
            "arena_delta": float(new_reward - old_reward),
            "gnn_old_reward": base_gnn,
            "gnn_new_reward": mutated_gnn,
            "gnn_delta": (mutated_gnn - base_gnn) if base_gnn is not None and mutated_gnn is not None else None,
            "lgbm_predicted_delta": predicted_delta,
            "lgbm_used": used_model,
            "n_nodes": len(mutated_arch.nodes),
            "n_params": mutated_arch.parameter_count(),
        }

    def _train_lgbm(self, encoder, lgbm_policy, records):
        X_list, y_list = [], []

        for rec in records:
            base_arch = rec["base_arch"]
            mutation = rec["mutation"]
            if "delta_reward" not in mutation:
                continue
            h = self._get_embedding(encoder, base_arch)
            fv = lgbm_policy.extract_feature_vector(h, mutation)
            X_list.append(fv)
            y_list.append(mutation["delta_reward"])

        if len(X_list) < 10:
            self.log(f"  Not enough data for LGBM ({len(X_list)} rows), skipping")
            return lgbm_policy

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)
        lgbm_policy.train(X, y)
        self.log(f"  Trained LGBM on {len(X)} rows")
        return lgbm_policy

    def _train_lgbm_scores(self, encoder, lgbm_policy, records):
        X_list, y_list = [], []
        for rec in records:
            h = self._get_embedding(encoder, rec["base_arch"])
            X_list.append(lgbm_policy.extract_feature_vector(h, rec["mutation"]))
            y_list.append(rec["score"])
        if len(X_list) < 2:
            return lgbm_policy
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)
        lgbm_policy.train(X, y)
        self.log(f"  Trained score LGBM on {len(X)} rows")
        return lgbm_policy

    def _train_evolution_policy(self, encoder, evolution_policy, records, current_score=None, best_score=None):
        X_list, y_list, component_list = [], [], []
        for rec in records:
            arch_for_embedding = rec.get("evaluated_arch", rec["base_arch"])
            h = self._get_embedding(encoder, arch_for_embedding)
            X_list.append(evolution_policy.lgbm_policy.extract_feature_vector(h, rec["mutation"]))
            y_list.append(rec["score"])
            component_list.append([
                rec["score"],
                rec.get("learnability", rec["score"]),
                rec.get("speed", 0.0),
                rec.get("opp_simp_raw", rec.get("opp_simp_raw_bonus", rec.get("opp_simp_bonus", 0.0))),
            ])
        if len(X_list) < 2:
            return evolution_policy
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32)
        y_components = np.array(component_list, dtype=np.float32)
        evolution_policy.train(X, y, y_components=y_components, current_score=current_score, best_score=best_score)
        return evolution_policy

    def _direct_score_mutation(self, arch, source="tournament_arch"):
        return {
            "mutation_type": source,
            "target_module_type": "",
            "mutation_sequence": [],
            "target_module_sequence": [],
            "n_mutation_steps": 0,
            "n_nodes_after": len(arch.nodes),
            "n_params_after": arch.parameter_count(),
            "is_valid": not bool(arch.validation_errors()),
            "mutated_arch": arch,
        }

    def _add_tournament_policy_records(self, records, tournament, architectures, source):
        start = len(records)
        for idx, arch in enumerate(architectures):
            records.append({
                "base_arch": copy.deepcopy(arch),
                "evaluated_arch": copy.deepcopy(arch),
                "mutation": self._direct_score_mutation(arch, source=source),
                "score": float(tournament["scores"][idx]),
                "learnability": float(tournament["learnabilities"][idx]),
                "speed": float(tournament["speeds"][idx]),
                "opp_simp_bonus": float(tournament["opp_simp_bonuses"][idx]),
                "opp_simp_raw": float(tournament.get("opp_simp_raw_bonuses", tournament["opp_simp_bonuses"])[idx]),
                "opp_simp_raw_bonus": float(tournament.get("opp_simp_raw_bonuses", tournament["opp_simp_bonuses"])[idx]),
                "record_source": source,
            })
        return len(records) - start

    def evolve_architecture(self, encoder, predictor=None, n_epochs=20,
                            tournament_size=8, n_candidates_per_step=10,
                            arch_size=None, simp_opp_bal=0.2,
                            initial_arch=None, initial_lgbm_policy=None,
                            initial_evolution_policy=None,
                            acceptance_temperature=0.05,
                            train_on_tournament_archs=False,
                            retrain_frequency=1, progress_callback=None,
                            should_stop=None):
        self.log("Evolve Architecture: tournament-scored mutation search")
        encoder = encoder.to(self.device)
        encoder.eval()
        if predictor is not None:
            predictor = predictor.to(self.device)
            predictor.eval()
        evolution_policy = initial_evolution_policy if initial_evolution_policy is not None else EvolutionPolicy(lgbm_policy=initial_lgbm_policy)
        lgbm_policy = evolution_policy.lgbm_policy
        current = copy.deepcopy(initial_arch) if initial_arch is not None else self.generator.generate(arch_size or self.arena.architecture_size)
        current_errors = current.validation_errors()
        if current_errors:
            raise ValueError("Initial architecture is invalid: " + "; ".join(current_errors[:8]))
        records = []
        history = []

        def tournament_for_focus(focus_arch, epoch, role):
            opponents = [
                self.generator.generate(arch_size or self.arena.architecture_size)
                for _ in range(max(tournament_size - 1, 1))
            ]
            archs = [focus_arch] + opponents
            labels = [role] + [f"Opponent {i}" for i in range(1, len(archs))]
            tournament = self.run_full_tournament(
                archs,
                labels=labels,
                progress_callback=progress_callback,
                context={"phase": "evolve", "epoch": epoch, "eval_role": role},
                simp_opp_bal=simp_opp_bal,
            )
            return tournament, archs

        if progress_callback is not None:
            progress_callback({
                "type": "evolve_initial",
                "score": None,
                "gnn_score": self._predict_gnn_reward(encoder, predictor, current),
                "current_arch_json": self._arch_to_brief_json(current),
                "pending_tournament": True,
            })

        initial_tournament, initial_archs = tournament_for_focus(current, 0, "initial")
        add_tournament_top_architectures(initial_tournament, initial_archs, 0, "initial")
        emit_top_architectures()
        current_score = float(initial_tournament["scores"][0])
        if train_on_tournament_archs:
            self._add_tournament_policy_records(records, initial_tournament, initial_archs, "initial_tournament_arch")
            if len(records) >= 2:
                evolution_policy = self._train_evolution_policy(
                    encoder,
                    evolution_policy,
                    records,
                    current_score=current_score,
                    best_score=current_score,
                )
                lgbm_policy = evolution_policy.lgbm_policy
        best_arch = copy.deepcopy(current)
        best_score = current_score
        versions = [{
            "version": 0,
            "label": "initial",
            "score": current_score,
            "arena_delta": 0.0,
            "true_learnability": float(initial_tournament["learnabilities"][0]),
            "true_speed": float(initial_tournament["speeds"][0]),
            "true_opp_simp_bonus": float(initial_tournament["opp_simp_bonuses"][0]),
            "true_opp_simp_raw_bonus": float(initial_tournament.get("opp_simp_raw_bonuses", initial_tournament["opp_simp_bonuses"])[0]),
            "architecture": copy.deepcopy(current),
            "arch_json": self._arch_to_brief_json(current),
        }]
        if progress_callback is not None:
            progress_callback({
                "type": "evolve_initial",
                "score": current_score,
                "gnn_score": self._predict_gnn_reward(encoder, predictor, current),
                "current_arch_json": self._arch_to_brief_json(current),
                "n_records": len(records),
            })

        for epoch in range(1, n_epochs + 1):
            if should_stop is not None and should_stop():
                break
            used_model = (
                evolution_policy.lgbm_policy.model is not None
                or evolution_policy.nn_trained
                or evolution_policy.meta_trained
            )

            valid_candidates = []
            predictions = []
            seen_candidate_signatures = set()
            max_candidate_pool = max(n_candidates_per_step * 8, n_candidates_per_step)

            def candidate_signature(candidate):
                arch = candidate["mutated_arch"]
                return (
                    tuple((n, type(arch.nodes[n]["module"]).__name__) for n in sorted(arch.nodes)),
                    tuple(sorted((int(s), int(t)) for s, t in arch.edges)),
                )

            while len(valid_candidates) < max_candidate_pool:
                batch = propose_random_mutations(current, n_candidates=n_candidates_per_step)
                made_progress = False
                for cand in batch:
                    if cand["mutated_arch"].validation_errors():
                        continue
                    sig = candidate_signature(cand)
                    if sig in seen_candidate_signatures:
                        continue
                    seen_candidate_signatures.add(sig)
                    made_progress = True
                    valid_candidates.append(cand)
                    if used_model:
                        try:
                            h_graph = self._get_embedding(encoder, cand["mutated_arch"])
                            fv = evolution_policy.lgbm_policy.extract_feature_vector(h_graph, cand)
                            comps = evolution_policy.predict_components(
                                np.array([fv], dtype=np.float32),
                                current_score=current_score,
                                best_score=best_score,
                                speed_bal=self.arena.speed_bal,
                                opp_simp_bal=simp_opp_bal,
                            )
                            predictions.append(float(comps["final"][0]))
                            cand["predicted_components"] = {
                                "learnability": float(comps["learnability"][0]),
                                "speed": float(comps["speed"][0]),
                                "opp_simp_bonus": float(comps["opp_simp_bonus"][0]),
                                "opp_simp_raw": float(comps["opp_simp_raw"][0]),
                            }
                        except Exception:
                            predictions.append(None)
                    else:
                        predictions.append(self._predict_gnn_reward(encoder, predictor, cand["mutated_arch"]))

                has_better_prediction = any(
                    pred is not None and pred > best_score
                    for pred in predictions
                )
                if has_better_prediction or not made_progress:
                    break

            if progress_callback is not None:
                sorted_candidate_rows = sorted(
                    [
                        {
                            "rank": idx + 1,
                            "mutation_type": cand.get("mutation_type", ""),
                            "target_module_type": cand.get("target_module_type", ""),
                            "mutation_sequence": cand.get("mutation_sequence", [cand.get("mutation_type", "")]),
                            "n_mutation_steps": cand.get("n_mutation_steps", 1),
                            "predicted_score": predictions[idx],
                            "predicted_components": cand.get("predicted_components"),
                            "n_nodes": len(cand["mutated_arch"].nodes),
                            "n_params": cand["mutated_arch"].parameter_count(),
                        }
                        for idx, cand in enumerate(valid_candidates)
                    ],
                    key=lambda row: row["predicted_score"] if row["predicted_score"] is not None else -float("inf"),
                    reverse=True,
                )
                progress_callback({
                    "type": "evolve_candidates",
                    "epoch": epoch,
                    "n_total": n_epochs,
                    "used_model": used_model,
                    "current_score": current_score,
                    "best_score": best_score,
                    "acceptance_temperature": acceptance_temperature,
                    "searched_candidates": len(valid_candidates),
                    "candidates": sorted_candidate_rows,
                })

            if valid_candidates:
                scored = [
                    (idx, pred if pred is not None else -float("inf"))
                    for idx, pred in enumerate(predictions)
                ]
                best_idx = max(scored, key=lambda item: item[1])[0]
            else:
                best_idx = -1

            if best_idx < 0 or predictions[best_idx] is None:
                row = {
                    "epoch": epoch,
                    "phase": "evolve",
                    "mutation_type": "none",
                    "target_module_type": "",
                    "predicted_score": predictions[best_idx] if best_idx >= 0 and predictions else None,
                    "true_score": current_score,
                    "prediction_error": None,
                    "previous_score": current_score,
                    "best_score": best_score,
                    "accepted": False,
                    "skipped": True,
                    "skip_reason": "No candidate predicted above the current best score.",
                    "lgbm_trained": False,
                    "n_records": len(records),
                }
                history.append(row)
                if progress_callback is not None:
                    progress_callback({
                        "type": "evolve_epoch",
                        **row,
                        "current_arch_json": self._arch_to_brief_json(current),
                        "best_arch_json": self._arch_to_brief_json(best_arch),
                    })
                continue

            chosen = valid_candidates[best_idx]
            predicted_score = predictions[best_idx]
            exploratory = predicted_score <= best_score
            new_arch = chosen["mutated_arch"]
            previous_score = current_score
            if progress_callback is not None:
                progress_callback({
                    "type": "evolve_testing",
                    "epoch": epoch,
                    "n_total": n_epochs,
                    "mutation_type": chosen.get("mutation_type", ""),
                    "target_module_type": chosen.get("target_module_type", ""),
                    "mutation_sequence": chosen.get("mutation_sequence", [chosen.get("mutation_type", "")]),
                    "n_mutation_steps": chosen.get("n_mutation_steps", 1),
                    "predicted_score": predicted_score,
                    "tested_arch_json": self._arch_to_brief_json(new_arch),
                })
            tournament, tournament_archs = tournament_for_focus(new_arch, epoch, "mutated")
            add_tournament_top_architectures(tournament, tournament_archs, epoch, "mutated")
            emit_top_architectures()
            true_score = float(tournament["scores"][0])
            true_learnability = float(tournament["learnabilities"][0])
            true_speed = float(tournament["speeds"][0])
            true_opp_bonus = float(tournament["opp_simp_bonuses"][0])
            true_opp_raw_bonus = float(tournament.get("opp_simp_raw_bonuses", tournament["opp_simp_bonuses"])[0])
            error = abs(float(predicted_score) - true_score) if predicted_score is not None else None
            delta = true_score - current_score
            if delta >= 0:
                acceptance_probability = 1.0
            elif acceptance_temperature > 0:
                acceptance_probability = math.exp(delta / acceptance_temperature)
            else:
                acceptance_probability = 0.0
            accepted = delta >= 0 or random.random() < acceptance_probability
            chosen["predicted_score"] = predicted_score
            chosen["true_score"] = true_score
            records.append({
                "base_arch": copy.deepcopy(current),
                "evaluated_arch": copy.deepcopy(new_arch),
                "mutation": chosen,
                "score": true_score,
                "learnability": true_learnability,
                "speed": true_speed,
                "opp_simp_bonus": true_opp_bonus,
                "opp_simp_raw_bonus": true_opp_raw_bonus,
                "record_source": "chosen_mutation",
            })
            tournament_records_added = 0
            if train_on_tournament_archs:
                tournament_records_added = self._add_tournament_policy_records(
                    records, tournament, tournament_archs, "evolution_tournament_arch"
                )

            tested_version = {
                "version": len(versions),
                "label": f"epoch {epoch} {'accepted' if accepted else 'rejected'}",
                "score": true_score,
                "arena_delta": delta,
                "accepted": accepted,
                "exploratory": exploratory,
                "acceptance_probability": acceptance_probability,
                "true_learnability": true_learnability,
                "true_speed": true_speed,
                "true_opp_simp_bonus": true_opp_bonus,
                "true_opp_simp_raw_bonus": true_opp_raw_bonus,
                "prediction_error": error,
                "architecture": copy.deepcopy(new_arch),
                "arch_json": self._arch_to_brief_json(new_arch),
            }
            versions.append(tested_version)

            if accepted:
                current = copy.deepcopy(new_arch)
                current_score = true_score
                if current_score > best_score:
                    best_score = current_score
                    best_arch = copy.deepcopy(current)

            lgbm_trained = False
            if len(records) >= 2 and (epoch % max(retrain_frequency, 1) == 0):
                evolution_policy = self._train_evolution_policy(encoder, evolution_policy, records, current_score=current_score, best_score=best_score)
                lgbm_policy = evolution_policy.lgbm_policy
                lgbm_trained = (
                    evolution_policy.lgbm_policy.model is not None
                    or evolution_policy.nn_trained
                    or evolution_policy.meta_trained
                )

            row = {
                "epoch": epoch,
                "phase": "evolve",
                "mutation_type": chosen.get("mutation_type", ""),
                "target_module_type": chosen.get("target_module_type", ""),
                "mutation_sequence": chosen.get("mutation_sequence", [chosen.get("mutation_type", "")]),
                "n_mutation_steps": chosen.get("n_mutation_steps", 1),
                "predicted_score": predicted_score,
                "true_score": true_score,
                "prediction_error": error,
                "true_learnability": true_learnability,
                "true_speed": true_speed,
                "true_opp_simp_bonus": true_opp_bonus,
                "true_opp_simp_raw_bonus": true_opp_raw_bonus,
                "predicted_components": chosen.get("predicted_components"),
                "previous_score": previous_score,
                "best_score": best_score,
                "accepted": accepted,
                "exploratory": exploratory,
                "acceptance_probability": acceptance_probability,
                "acceptance_temperature": acceptance_temperature,
                "lgbm_trained": lgbm_trained,
                "n_records": len(records),
                "tournament_records_added": tournament_records_added,
            }
            history.append(row)
            if progress_callback is not None:
                progress_callback({
                    "type": "evolve_epoch",
                    **row,
                    "current_arch_json": self._arch_to_brief_json(current),
                    "best_arch_json": self._arch_to_brief_json(best_arch),
                    "version_score": {
                        "version": tested_version["version"],
                        "epoch": epoch,
                        "accepted": accepted,
                        "exploratory": exploratory,
                        "mutation_type": row["mutation_type"],
                        "target_module_type": row["target_module_type"],
                        "mutation_sequence": row["mutation_sequence"],
                        "n_mutation_steps": row["n_mutation_steps"],
                        "arena_score": true_score,
                        "previous_arena_score": previous_score,
                        "arena_delta": delta,
                        "acceptance_probability": acceptance_probability,
                        "true_learnability": true_learnability,
                        "true_speed": true_speed,
                        "true_opp_simp_bonus": true_opp_bonus,
                        "true_opp_simp_raw_bonus": true_opp_raw_bonus,
                        "predicted_components": chosen.get("predicted_components"),
                        "gnn_score": self._predict_gnn_reward(encoder, predictor, current),
                        "lgbm_predicted_score": predicted_score if used_model else None,
                        "prediction_error": error,
                        "n_nodes": len(current.nodes),
                        "n_params": current.parameter_count(),
                    },
                })

        if len(records) >= 2:
            evolution_policy = self._train_evolution_policy(encoder, evolution_policy, records, current_score=current_score, best_score=best_score)
            lgbm_policy = evolution_policy.lgbm_policy

        if progress_callback is not None:
            progress_callback({
                "type": "evolve_done",
                "n_epochs": len(history),
                "best_score": best_score,
                "interrupted": should_stop() if should_stop is not None else False,
            })
        return {
            "best_architecture": best_arch,
            "best_reward": best_score,
            "training_history": history,
            "gnn_encoder": encoder,
            "gnn_predictor": predictor,
            "lgbm_model": lgbm_policy,
            "evolution_policy": evolution_policy,
            "versions": versions,
            "top_architectures": ranked_top_architectures(),
            "interrupted": should_stop() if should_stop is not None else False,
        }

    def run(self, n_phase_a_episodes=500, n_phase_b_episodes=500,
            n_candidates_per_step=10, train_frequency=50,
            retrain_frequency=50, n_gnn_epochs=100,
            initial_encoder=None, initial_predictor=None,
            initial_lgbm_policy=None, initial_arch=None,
            skip_phase_b=False,
            progress_callback=None) -> dict:
        self.log("Starting RL search")

        def wrap_cb(msg):
            if progress_callback is not None:
                try:
                    progress_callback(msg)
                except Exception:
                    pass

        if initial_encoder is not None:
            encoder = initial_encoder.to(self.device)
            predictor = initial_predictor.to(self.device) if initial_predictor is not None else None
            best_arch_a = None
            history_a = []
            wrap_cb({
                "type": "phase_a_done",
                "n_episodes": 0,
                "best_reward": None,
                "skipped": True,
                "message": "Phase A skipped because a saved GNN was loaded.",
            })
            self.log("Phase A skipped; using loaded GNN")
        else:
            best_arch_a, encoder, predictor, history_a = self.phase_a(
                n_episodes=n_phase_a_episodes,
                train_frequency=train_frequency,
                n_gnn_epochs=n_gnn_epochs,
                progress_callback=wrap_cb,
            )
            best_a_reward = history_a[-1]["best_reward"] if history_a else float('nan')
            self.log(f"Phase A complete. Best reward: {best_a_reward:.4f}")

        if skip_phase_b:
            best_arch_b = None
            lgbm_policy = initial_lgbm_policy if initial_lgbm_policy is not None else LGBMPolicy()
            history_b = []
            wrap_cb({
                "type": "phase_b_done",
                "n_episodes": 0,
                "skipped": True,
                "message": "Phase B skipped; GNN training only.",
            })
            self.log("Phase B skipped; GNN training only")
        else:
            best_arch_b, lgbm_policy, history_b = self.phase_b(
                encoder=encoder,
                predictor=predictor,
                n_episodes=n_phase_b_episodes,
                n_candidates_per_step=n_candidates_per_step,
                retrain_frequency=retrain_frequency,
                initial_arch=initial_arch,
                initial_lgbm_policy=initial_lgbm_policy,
                progress_callback=wrap_cb,
            )

        full_history = history_a + history_b

        if lgbm_policy.model is not None:
            feature_imp = lgbm_policy.feature_importance(
                h_graph_dim=GNN_EMBEDDING_DIM
            )
        else:
            feature_imp = {}

        final_arch = best_arch_b if best_arch_b is not None else best_arch_a
        final_reward = self.evaluate_architecture(
            final_arch,
            n_samples=3,
            progress_callback=wrap_cb,
            context={"phase": "done", "episode": 0, "eval_role": "final_best"},
        )

        result = {
            "best_architecture": final_arch,
            "best_reward": final_reward,
            "training_history": full_history,
            "gnn_encoder": encoder,
            "gnn_predictor": predictor,
            "lgbm_model": lgbm_policy,
            "feature_importance": feature_imp,
            "phase_a_history": history_a,
            "phase_b_history": history_b,
        }

        self.log(f"RL search complete. Final reward: {final_reward:.4f}")
        return result
