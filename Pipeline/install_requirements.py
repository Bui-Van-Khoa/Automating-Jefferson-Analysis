import subprocess
import sys

def install_requirements():
    try:
        # Check if requirements.txt exists
        with open('requirements.txt') as f:
            requirements = f.read().splitlines()
        
        print("Installing requirements from requirements.txt...")
        
        # Install each requirement one by one with progress feedback
        for req in requirements:
            if req.strip() and not req.strip().startswith('#'):
                print(f"Installing {req}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install"] + req.split())
        
        print("\nAll requirements installed successfully!")
    
    except FileNotFoundError:
        print("Error: requirements.txt not found in current directory")
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")

if __name__ == "__main__":
    install_requirements()