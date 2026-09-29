import logging
import sys
from pathlib import Path

# We create this configuration file to create a file
# That will track the outputs
# from stdout(standard ouput)(channnel a program writes executed files)


# 1. Find the project root, wherever this file happens to live
Base_path = Path(__file__).parent.parent.absolute()
# 2. Define (and create) a logs directory relative to that root
log_path = Base_path / "logs"
# 3. Creates the file if exists
log_path.mkdir(exist_ok=True, parents=True)

# 4. Configure logging ONCE, here, and nowhere else in the project
logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path / "info.log"),
    ],
)

root_logger = logging.getLogger()
