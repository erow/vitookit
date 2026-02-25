# AGENTS.md

## Cursor Cloud specific instructions

**Vitookit** is a Python CLI toolkit for evaluating vision models (ViT). There is no web UI, database, or Docker infrastructure — it is a single `pip install -e .` Python package.

### Running the application

- Install: `pip install -e .` (editable mode) from the repo root.
- The `vitrun` and `submitit` CLI entry points are installed to `~/.local/bin/`. Ensure `$HOME/.local/bin` is on `PATH`.
- `vitrun` wraps `torchrun` and resolves evaluation scripts from `vitookit/evaluation/`.
- No GPU is available in the Cloud VM; CPU-only inference/testing works for development. Training/evaluation scripts require `--standalone` or explicit `torchrun` args.

### Testing

- The only test file is `vitookit/datasets/test_build_dataset.py`, which is a skeleton placeholder with an undefined `Args` class — it will fail by design.
- To verify the environment, run Python import checks and inference:
  ```python
  from vitookit.models.build_model import build_model
  model = build_model('vit_tiny_patch16_224', num_classes=10)
  ```
- CIFAR10 auto-downloads to any path and is the easiest dataset for end-to-end verification.

### Linting

- No linter is configured in the repo. `pyflakes vitookit/` can be used for basic static analysis. Pre-existing warnings (unused imports, star imports) are expected.

### Gotchas

- There is no `vitookit/__init__.py` at the package root — Python 3 implicit namespace packages handle this, but `find_packages` in `setup.py` with `include=['vitookit']` relies on this. The editable install (`pip install -e .`) works correctly despite this.
- `timm` deprecation warnings about `timm.models.layers` and `timm.models.registry` are expected and harmless.
- `wandb` is a dependency but not required at runtime for local dev — it gracefully handles missing auth.
