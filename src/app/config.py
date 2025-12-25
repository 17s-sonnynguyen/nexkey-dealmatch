from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # NexKey-DealMatch/
DATA_PATH = PROJECT_ROOT / "data" / "nexkey_synthetic_dataset_v1"
CKPT_PATH = PROJECT_ROOT / "models" / "checkpoints"

MAX_LEN_DUAL = 48
MAX_LEN_CROSS = 96
DEFAULT_TOP_N = 50
DEFAULT_TOP_K = 5
