import copy

import torch

from graph.executor import Executor


DEFAULT_SMOKE_SHAPES = ((2, 1, 8), (2, 3, 5), (2, 15, 18))


def architecture_execution_errors(arch, shapes=DEFAULT_SMOKE_SHAPES, device=None):
    errors = []
    if arch.validation_errors():
        return arch.validation_errors()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for shape in shapes:
        try:
            arch_copy = copy.deepcopy(arch)
            arch_copy.reset_state()
            executor = Executor(arch_copy).to(device)
            x = torch.randn(*shape, device=device)
            with torch.no_grad():
                out = executor.forward(x)
            if not out or not torch.isfinite(out[0]).all():
                errors.append(f"Forward smoke test produced non-finite output for input shape {shape}")
        except Exception as exc:
            errors.append(f"Forward smoke test failed for input shape {shape}: {exc}")
    return errors


def is_executable_architecture(arch, shapes=DEFAULT_SMOKE_SHAPES, device=None):
    return not architecture_execution_errors(arch, shapes=shapes, device=device)
