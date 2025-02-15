import os
from pathlib import Path
from dotenv import load_dotenv

# Get the root directory of your project
ROOT_DIR = Path(__file__).parent.parent

# Load environment variables from .env file
def load_environment():
    """
    Load environment variables from .env file.
    This function ensures all required environment variables are present
    and properly loaded before the application starts.
    
    Raises:
        ValueError: If any required environment variables are missing
    """
    # Load .env file from project root
    env_path = ROOT_DIR / '.env'
    load_dotenv(env_path)
    
    # Check for required environment variables
    required_vars = ['OPENAI_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing_vars)}\n"
            f"Please create a .env file in {ROOT_DIR} with these variables."
        )

# Function to get specific environment variables
def get_openai_api_key():
    """
    Retrieve the OpenAI API key from environment variables.
    This function provides a centralized way to access the API key
    and ensures it's available before being used.
    
    Returns:
        str: The OpenAI API key
    """
    return os.getenv('OPENAI_API_KEY')