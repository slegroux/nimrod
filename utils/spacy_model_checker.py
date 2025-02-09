import importlib.util
import subprocess
import sys

def ensure_spacy_model(model_name='en_core_web_sm'):
    """
    Check if a spaCy language model is installed, and download it if not.
    
    Args:
        model_name (str, optional): Name of the spaCy model to check/download. 
                                    Defaults to 'en_core_web_sm'.
    
    Raises:
        RuntimeError: If model download fails
    
    Returns:
        bool: True if model is successfully installed or already exists
    """
    try:
        # First, check if spaCy is installed
        import spacy
    except ImportError:
        print(f"spaCy is not installed. Please install spaCy first using 'pip install spacy'.")
        return False
    
    try:
        # Try to load the model
        spacy.load(model_name)
        print(f"Model {model_name} is already installed.")
        return True
    except OSError:
        # Model not found, attempt to download
        print(f"Downloading {model_name} model...")
        try:
            subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])
            print(f"Successfully downloaded {model_name}.")
            return True
        except subprocess.CalledProcessError:
            print(f"Failed to download {model_name}. Please check your internet connection.")
            return False

# Example usage
if __name__ == "__main__":
    ensure_spacy_model()
