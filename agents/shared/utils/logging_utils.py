import logging
from logging.handlers import RotatingFileHandler
from rich.logging import RichHandler
import os
from datetime import datetime

def setup_logging():
    """Set up logging with a file handler and a rich console handler."""
    if not os.path.exists("logs"):
        os.makedirs("logs")

    log_filename = datetime.now().strftime("logs/research_agent_%Y%m%d_%H%M%S.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            RotatingFileHandler(
                log_filename,
                maxBytes=10*1024*1024, # 10 MB
                backupCount=5
            ),
            RichHandler(
                level=logging.WARNING,
                show_path=False,
                show_level=True,
                show_time=True,
                markup=True,
                rich_tracebacks=True
            )
        ]
    )

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.info("Logging setup complete.")

def get_logger(name: str):
    """Get a logger with the specified name."""
    return logging.getLogger(name)
