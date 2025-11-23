from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    MODEL_PATH: str = "models/melanoma_resnet.pt"

    # Local CAM output (used if R2 isn't configured)
    CAM_OUTPUT_DIR: str = "cams"

    # Base URL of this ML service (only needed if you serve CAMs locally)
    BASE_URL: str = "https://ml.pigmemento.app"

    # R2 (S3-compatible). If these are all set, we'll upload CAMs to R2.
    R2_ACCOUNT_ID: str | None = None
    R2_ACCESS_KEY_ID: str | None = None
    R2_SECRET_ACCESS_KEY: str | None = None
    R2_BUCKET: str | None = None
    R2_PUBLIC_BASE_URL: str | None = None  # e.g. https://cdn.pigmemento.app


settings = Settings()