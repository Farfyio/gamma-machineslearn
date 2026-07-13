# Project Structure

This repository is organized to keep data-independent machine learning logic modular and reusable.

## Directories

- `src/gamma_machineslearn/config`: configuration dataclasses and YAML loading
- `src/gamma_machineslearn/logging_utils`: central logger configuration helpers
- `src/gamma_machineslearn/preprocessing`: reusable preprocessing pipeline and feature validation hooks
- `src/gamma_machineslearn/evaluation`: shared metric calculation and report writing helpers
- `tests`: placeholder location for unit/integration tests
- `docs`: project documentation

## Notes

- Dataset-specific logic should be added outside these shared modules.
- Keep preprocessing and evaluation utilities generic for reuse across experiments.
