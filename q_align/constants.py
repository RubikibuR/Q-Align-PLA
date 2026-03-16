CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "./demo_logs"

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<|image|>"

# PLA Model Constants
PROTEIN_TOKEN_INDEX = -201
LIGAND_TOKEN_INDEX = -202
DEFAULT_PROTEIN_TOKEN = "<|protein|>"
DEFAULT_LIGAND_TOKEN = "<|ligand|>"

# 亲和力等级名称，按 pKd 升序排列（低 → 高亲和力），全部为单 token 词
PLA_LEVEL_NAMES = ["negligible", "weak", "moderate", "strong", "potent"]
