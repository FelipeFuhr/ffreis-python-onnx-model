"""Runtime entrypoint for HTTP serving."""

from __future__ import annotations

import logging
import os

from application import create_application
from config import Settings

settings = Settings()

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)

application = create_application(settings)


def main() -> None:
    """Replace current process with Gunicorn server process."""
    os.execvp(
        "gunicorn",
        [
            "gunicorn",
            "-c",
            "python:gunicorn_configuration",
            "serving:application",
        ],
    )


if __name__ == "__main__":
    main()
