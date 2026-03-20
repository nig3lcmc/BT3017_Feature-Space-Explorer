from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    app_title: str = "Feature Space Explorer"
    app_icon: str = "📊"
    layout: str = "wide"
    default_dataset: str = "iris"


CONFIG = AppConfig()