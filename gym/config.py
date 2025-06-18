"""
Configuration management with validation bugs for testing the SWE agent.
"""

import json
import os
from typing import Any

import yaml


class ConfigManager:
    """Configuration manager with various bugs."""

    def __init__(self, config_file: str = "config.yaml"):
        self.config_file = config_file
        self.config = {}
        self.defaults = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "myapp",
                "user": "admin",
            },
            "api": {"timeout": 30, "retries": 3, "base_url": "https://api.example.com"},
            "logging": {"level": "INFO", "file": "app.log"},
        }

    def load_config(self) -> dict[str, Any]:
        """Load configuration from file."""
        # Bug: No error handling for file not found
        with open(self.config_file) as file:
            if self.config_file.endswith(".yaml") or self.config_file.endswith(".yml"):
                # Bug: No error handling for invalid YAML
                self.config = yaml.safe_load(file)
            elif self.config_file.endswith(".json"):
                # Bug: No error handling for invalid JSON
                self.config = json.load(file)
            else:
                # Bug: Doesn't handle other file types gracefully
                raise ValueError(f"Unsupported config file format: {self.config_file}")

        return self.config

    def save_config(self, config: dict[str, Any] | None = None) -> bool:
        """Save configuration to file."""
        if config is not None:
            self.config = config

        # Bug: No error handling for write permissions
        with open(self.config_file, "w") as file:
            if self.config_file.endswith(".yaml") or self.config_file.endswith(".yml"):
                yaml.dump(self.config, file)
            elif self.config_file.endswith(".json"):
                json.dump(self.config, file, indent=2)

        return True

    def get_value(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        # Bug: No validation of key_path format
        keys = key_path.split(".")
        value = self.config

        for key in keys:
            # Bug: No handling of missing keys
            value = value[key]

        return value

    def set_value(self, key_path: str, value: Any) -> None:
        """Set configuration value using dot notation."""
        keys = key_path.split(".")
        config = self.config

        # Bug: Doesn't create nested dictionaries if they don't exist
        for key in keys[:-1]:
            config = config[key]

        config[keys[-1]] = value

    def validate_config(self) -> list[str]:
        """Validate configuration against schema."""
        errors = []

        # Database validation
        if "database" in self.config:
            db_config = self.config["database"]

            # Bug: No validation of port type
            if "port" in db_config and db_config["port"] < 1:
                errors.append("Database port must be a positive integer")

            # Bug: No validation of required fields
            required_db_fields = ["host", "name", "user"]
            for field in required_db_fields:
                if field not in db_config:
                    errors.append(f"Missing required database field: {field}")

        # API validation
        if "api" in self.config:
            api_config = self.config["api"]

            # Bug: No URL format validation
            if "base_url" in api_config:
                url = api_config["base_url"]
                if not url.startswith("http"):
                    errors.append("API base_url must start with http:// or https://")

            # Bug: No validation of numeric ranges
            if "timeout" in api_config:
                timeout = api_config["timeout"]
                if timeout <= 0:
                    errors.append("API timeout must be positive")

        return errors

    def merge_with_defaults(self) -> dict[str, Any]:
        """Merge configuration with defaults."""
        # Bug: Simple dictionary update doesn't handle nested dictionaries
        merged = self.defaults.copy()
        merged.update(self.config)
        return merged

    def get_environment_overrides(self) -> dict[str, Any]:
        """Get configuration overrides from environment variables."""
        overrides = {}

        # Bug: Hard-coded environment variable names
        env_mappings = {
            "DB_HOST": "database.host",
            "DB_PORT": "database.port",
            "DB_NAME": "database.name",
            "API_TIMEOUT": "api.timeout",
        }

        for env_var, config_path in env_mappings.items():
            if env_var in os.environ:
                # Bug: No type conversion for environment variables
                value = os.environ[env_var]
                self.set_value(config_path, value)

        return overrides

    def backup_config(self) -> str:
        """Create a backup of the current configuration."""
        backup_file = f"{self.config_file}.backup"

        # Bug: No error handling for backup operation
        with open(backup_file, "w") as file:
            if self.config_file.endswith(".yaml") or self.config_file.endswith(".yml"):
                yaml.dump(self.config, file)
            elif self.config_file.endswith(".json"):
                json.dump(self.config, file, indent=2)

        return backup_file

    def restore_from_backup(self, backup_file: str) -> bool:
        """Restore configuration from backup."""
        # Bug: No validation that backup file exists
        with open(backup_file) as file:
            if backup_file.endswith(".yaml") or backup_file.endswith(".yml"):
                self.config = yaml.safe_load(file)
            elif backup_file.endswith(".json"):
                self.config = json.load(file)

        # Bug: Doesn't save the restored config to the main file
        return True

    def get_database_url(self) -> str:
        """Generate database URL from configuration."""
        db_config = self.config.get("database", {})

        # Bug: No validation that required fields exist
        host = db_config["host"]
        port = db_config["port"]
        name = db_config["name"]
        user = db_config["user"]

        # Bug: No handling of password or other connection parameters
        return f"postgresql://{user}@{host}:{port}/{name}"

    def is_production(self) -> bool:
        """Check if running in production environment."""
        # Bug: Case-sensitive comparison
        env = os.environ.get("ENVIRONMENT", "development")
        return env == "production"

    def get_log_level(self) -> str:
        """Get logging level with environment override."""
        # Bug: No validation of log level values
        log_config = self.config.get("logging", {})
        env_level = os.environ.get("LOG_LEVEL")

        if env_level:
            return env_level

        return log_config.get("level", "INFO")


def main():
    """Demo function with configuration bugs."""
    config_manager = ConfigManager("sample_config.yaml")

    try:
        # Bug: Will fail if file doesn't exist
        config = config_manager.load_config()
        print(f"Loaded config: {config}")

        # Bug: Will fail if key path doesn't exist
        db_host = config_manager.get_value("database.host")
        print(f"Database host: {db_host}")

        # Bug: Will fail if database config is incomplete
        db_url = config_manager.get_database_url()
        print(f"Database URL: {db_url}")

        # Bug: Will fail if there are validation errors
        errors = config_manager.validate_config()
        if errors:
            print(f"Configuration errors: {errors}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
