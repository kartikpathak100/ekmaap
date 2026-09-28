"""Settings from environment variables (see .env.example)."""
import os
from pathlib import Path


class Settings:
    def __init__(self):
        self.database_url = os.environ.get("EKMAAP_DATABASE_URL", "postgresql+psycopg://ekmaap:ekmaap@localhost:5432/ekmaap")
        self.data_dir = Path(os.environ.get("EKMAAP_DATA_DIR", "./data/store")).resolve()
        self.secret_key = os.environ.get("EKMAAP_SECRET_KEY", "change-me-in-production")
        self.token_hours = float(os.environ.get("EKMAAP_TOKEN_HOURS", "12"))
        self.admin_login = os.environ.get("EKMAAP_ADMIN_LOGIN")
        self.admin_password = os.environ.get("EKMAAP_ADMIN_PASSWORD")
        self.px_per_mm = float(os.environ.get("EKMAAP_PX_PER_MM", "3.0"))
        self.max_upload_mb = float(os.environ.get("EKMAAP_MAX_UPLOAD_MB", "25"))
        self.cors_origins = [o for o in os.environ.get("EKMAAP_CORS_ORIGINS", "").split(",") if o]


settings = Settings()
