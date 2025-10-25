from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
TRANSACTIONS_DIR = DATA_DIR / "transactions"
ARTICLES_PATH = DATA_DIR / "articles.parquet"
DB_PATH = DATA_DIR / "shopsight.db"

# Data processing
TOP_N_ARTICLES = 60

# UI settings
SEARCH_PLACEHOLDER = "e.g. Jade denim, 706016001"
CHAT_HISTORY_LIMIT = 12
HELP_URL = "https://github.com/<your-handle>/shopsight-prototype"
FEEDBACK_EMAIL = "demo-feedback@example.com"

