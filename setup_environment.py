#!/usr/bin/env python
import sys
import subprocess
import importlib

# List of required packages
dependencies = [
    'torch',
    'torchvision',
    'numpy',
    'matplotlib',
    'opencv-python'
]

def install_and_import(package):
    """
    Try to import the package; if not found, install it using pip.
    """
    try:
        importlib.import_module(package)
        print(f"{package} is already installed.")
    except ImportError:
        print(f"{package} not found. Installing {package}...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"{package} installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {package}. Error: {e}")
            sys.exit(1)

if __name__ == '__main__':
    # Check and install dependencies
    for package in dependencies:
        install_and_import(package)
    
    # Verify GPU availability using PyTorch
    try:
        import torch
        print("\nVerifying PyTorch installation and GPU availability...")
        if torch.cuda.is_available():
            print("CUDA is available! GPU is accessible.")
            print("Number of CUDA devices:", torch.cuda.device_count())
            print("Current CUDA device:", torch.cuda.current_device())
        else:
            print("CUDA is not available. The project will run on CPU.")
    except Exception as e:
        print("An error occurred while verifying PyTorch and GPU availability:", e)
