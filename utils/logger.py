import logging
import os
from datetime import datetime

class Logger:
    def setup_logger():

        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)

        log_filename = f"log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
        log_path = os.path.join(log_dir, log_filename)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ]
        )

        logger = logging.getLogger()

        return logger
