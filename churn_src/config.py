import logging
import os
import sys
from pathlib import Path

# We create this configuration file to create a file
# That will track the outputs
# from stdout(standard ouput)(channnel a program writes executed files)


# 1. Find the project root, wherever this file happens to live
Base_path = Path(__file__).parent.absolute()
# 2. Define (and create) a logs directory relative to that root
log_path = os.path.join(Base_path, "logs")
# 3. Creates the file if exists
log_path.parent.mkdir(exist_ok=True, parents=True)

# 4. Configure logging ONCE, here, and nowhere else in the project
logging.basicConfig(
    stream=sys.stdout,
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=logging.FileHandler(log_path, "info_log"),
)  # durable file output

root_logger = logging.getLogger()
