import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

from modules.base import ModuleType
from modules.operations import *
from modules.learnable import *
from modules.input import *
from graph.architecture import *
from graph.executor import *
from graph.generator import *
from tournament import arena
from tournament.arena import *
from scipy.stats import pearsonr
from backend.fight_viz import run_fight_visualization

def linReg():
    architecture = Architecture()

    inputTens = torch.randn(2,3,4)
    outputTargetTens = torch.randn(2,1,1)
    inputModule = Input()
    inputModule.set_data(inputTens)
    weights = LearnableParameter((4,8))
    bias = LearnableParameter((1,1,8))
    matmul = MatMul()
    addition = Add()

    architecture.add_node(0,inputModule)
    architecture.add_node(1,weights)
    architecture.add_node(2,bias)
    architecture.add_node(3,matmul)
    architecture.add_node(4,addition)

    architecture.add_edge(0,3)
    architecture.add_edge(1,3)
    architecture.add_edge(3,4)
    architecture.add_edge(2,4)

    print(architecture.isValid())
    print(list(networkx.topological_sort(architecture)))
    print(list(architecture.nodes))

    executor = Executor(architecture)
    executor.set_Output_Adapter(inputTens, outputTargetTens.shape)
    output = executor.forward(inputTens)
    optimizer = torch.optim.Adam(executor.parameters(), lr=0.01)
    for i in range(101):
        output = executor.forward(inputTens)
        loss = torch.nn.functional.mse_loss(output[0], outputTargetTens)
        if i % 10 == 0:
            print(f"Loss: {loss.item()}")
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def twolayersMLP():
    architecture = Architecture()

    inputTens = torch.randn(16,3,4)
    outputTargetTens = torch.randn(16,1,4)
    inputModule = Input()
    inputModule.set_data(inputTens)
    architecture.add_node(0,inputModule)

    architecture.append_node(LearnableParameter((1,4,13)))
    architecture.append_node(LearnableParameter((1,1,7)))
    architecture.append_node(MatMul())
    architecture.append_node(Add())
    architecture.append_node(Activation())

    architecture.append_node(LearnableParameter((1,13,4)))
    architecture.append_node(LearnableParameter((1,1,4)))
    architecture.append_node(MatMul())
    architecture.append_node(Add())

    #First layer connections
    architecture.add_edge(0,3)
    architecture.add_edge(1,3)
    architecture.add_edge(3,4)
    architecture.add_edge(2,4)
    architecture.add_edge(4,5)
    #Second layer connections
    architecture.add_edge(5,8)
    architecture.add_edge(6,8)
    architecture.add_edge(7,9)
    architecture.add_edge(8,9)

    print(architecture.isValid())
    print(list(networkx.topological_sort(architecture)))

    executor = Executor(architecture)
    executor.fit(inputTens, outputTargetTens, verbose=True, lr=0.002, max_iter=1000, batch_size=16, patience = 10, min_delta = 1e-7)

def patch_module_types(arch):
    for nid in arch.nodes:
        mod = arch.nodes[nid]["module"]

        if isinstance(mod, Input):
            mod.module_type = ModuleType.INPUT
        elif isinstance(mod, LearnableParameter):
            mod.module_type = ModuleType.LEARNABLE
        elif isinstance(mod, Activation):
            mod.module_type = ModuleType.ACTIVATION
        elif isinstance(mod, (Add, MatMul, Mult)):
            mod.module_type = ModuleType.BASIC
        elif isinstance(mod, (Pooling, Concat, Transpose, Shift, Split)):
            mod.module_type = ModuleType.STRUCTURAL
        elif isinstance(mod, Normalizer):
            mod.module_type = ModuleType.NORM

    return arch

def test_fight_viz():
    arena = Arena(n_fights=48, architecture_size=12, arena_contestants=3, dataset_size=512, train_test_split=0.7, generation_type="agnostic", verbose=False, report=False)
    generator = Generator(generation_type="agnostic")
    arch_a = generator.generate(12)
    arch_b = generator.generate(12)
    #arch_b = arena.make_mlp([32,16,8])

    print("Running fight visualization...")
    result = run_fight_visualization(arch_a, arch_b, max_iter=500, lr=5e-3, n_snapshots=20)

    x = result["x"]
    fight_a = result["fight_a"]
    fight_b = result["fight_b"]

    print(f"A broken: {fight_a['broken']}")
    print(f"B broken: {fight_b['broken']}")
    print(f"Snapshots A: {len(fight_a['snapshots'])}")
    print(f"Snapshots B: {len(fight_b['snapshots'])}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Fight Visualization Test", fontsize=14)

    # Pick a few snapshots to overlay
    def plot_fight(ax_curve, ax_loss, fight, title):
        target = fight["target"]
        snapshots = fight["snapshots"]
        loss_history = fight["loss_history"]
        broken = fight["broken"]

        # Plot target
        ax_curve.plot(x, target, color="blue", linewidth=2, label="Target", zorder=10)


        # Calculate bounds based strictly on the target
        t_min, t_max = min(target), max(target)
        margin = (t_max - t_min) * 0.1  # Add 10% padding
        
        # Handle the "Degeneracy Trap" (constant targets) to avoid [0, 0] limits
        if margin == 0: 
            margin = 1.0 
            
        ax_curve.set_ylim(t_min - margin, t_max + margin)

        # Overlay snapshots with increasing opacity
        n = len(snapshots)
        for i, snap in enumerate(snapshots):
            alpha = 0.15 + 0.85 * (i / max(n - 1, 1))
            color = "red" if i < n - 1 else "orange"
            ax_curve.plot(x, snap["y"], color=color, alpha=alpha, linewidth=1)

        # Mark last snapshot clearly
        if snapshots:
            last = snapshots[-1]
            ax_curve.plot(x, last["y"], color="orange", linewidth=2,
                         label=f"Final (epoch {last['epoch']})", zorder=9)

        if broken:
            ax_curve.text(0.5, 0.5, "ARCH BROKEN",
                         transform=ax_curve.transAxes,
                         fontsize=16, color="red", fontweight="bold",
                         ha="center", va="center", alpha=0.6)

        ax_curve.set_title(title)
        ax_curve.legend(fontsize=8)
        ax_curve.set_xlabel("x")
        ax_curve.set_ylabel("y")

        # Loss curve
        if loss_history:
            ax_loss.plot(loss_history, color="green", linewidth=1)
            ax_loss.set_title(f"{title} — Loss")
            ax_loss.set_xlabel("Epoch")
            ax_loss.set_ylabel("MSE")
            ax_loss.set_yscale("log")

        # Mark snapshot epochs on loss curve
        for snap in snapshots:
            epoch = snap["epoch"]
            if epoch < len(loss_history):
                ax_loss.axvline(epoch, color="gray", alpha=0.3, linewidth=0.8)

    plot_fight(axes[0][0], axes[0][1], fight_a, "A learning B")
    plot_fight(axes[1][0], axes[1][1], fight_b, "B learning A")

    plt.tight_layout()
    plt.savefig("fight_viz_test.png", dpi=120)
    plt.show()
    print("Saved to fight_viz_test.png")
    

arena = Arena(n_fights=48, architecture_size=12, arena_contestants=3, dataset_size=512, train_test_split=0.7, generation_type="agnostic", verbose=False, report=False)
print(arena.tune_simp_opp_bal(n_archs=12, n_rounds=10, verbose=True, randomizeHP=True))