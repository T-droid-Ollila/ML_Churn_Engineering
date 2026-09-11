# 8. dotenv: Load environment variables from ./mlrun file

from dotenv import load_dotenv

# 14. load_dotenv(): Reads the .env file in your project root

# - Looks for MLFLOW_TRACKING_URI=./mlruns or anyother private information

load_dotenv()
