# Version 0.3.1 - 2026-10-02

- Fix the Conda build: map pip dependency names to real Conda packages, add the `romi-eu` channel for `plantdb.commons`, restrict the Python build matrix to supported versions, and build with `--no-deps` (the CPU torch stack is pip-installed separately since conda-build blocks pip network access and `torchvision` 0.28 has no Conda build).

# Version 0.3.0 - 2026-10-02

- Switch dataset backend from `romidata` to `plantdb` and unify the FSDB API across the codebase.
- Upgrade to PyTorch 2.13 and raise the minimum supported Python version to 3.10.
- Rework model loading API (`model_from_file`, `label_names`) and add checkpoint conversion handling.
- Refactor the `finetune` CLI and `convert_models.py` to use a unified logger and log-level option.
- Free GPU memory after segmentation inference.
- Reorganize the test suite into fast, self-contained `tests/unit`, weight-dependent `tests/integration`, and full-model `tests/model` groups.
- Add CI/CD workflows for unit tests, Conda/PyPI publishing, and GitHub Pages documentation.
- Add a developer release guide (`docs/developers/releases.md`) and link it from the MkDocs navigation.
