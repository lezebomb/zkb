# Local Verification Checklist

- `PYTHONPATH=src pytest -q`
- `python scripts/verify_training_setup.py`
- `/health` returns `status: ok` and `bridge_mode: mock-local`
- `/infer` returns JSON on success and failure
- `/debug/status` shows expected local model paths
- `ALLOW_MODEL_AUTO_DOWNLOAD=false` before competition
- `ALLOW_DATASET_DOWNLOAD=false` before competition
- detect outputs original pixel coordinates
- package zip is under 2GB and includes requirements/configs/scripts/docs/tests/models

