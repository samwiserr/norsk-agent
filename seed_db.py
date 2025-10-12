# seed_db.py
import os
from src.utils.db import log_interaction, DB_PATH

print("Database path:", os.path.abspath(DB_PATH))
log_interaction("seed", "dev-session", "hello", "world")
print("✅ Seeded a test log entry.")
