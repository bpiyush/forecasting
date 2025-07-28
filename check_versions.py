#!/usr/bin/env python3

import sys
import subprocess

def check_package_version(package_name):
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', 'show', package_name], 
                              capture_output=True, text=True, check=True)
        for line in result.stdout.split('\n'):
            if line.startswith('Version:'):
                return line.split(': ')[1]
    except:
        return "Not installed"

def check_torch_cuda():
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda if cuda_available else "N/A"
        device_count = torch.cuda.device_count() if cuda_available else 0
        return f"CUDA Available: {cuda_available}, CUDA Version: {cuda_version}, GPU Count: {device_count}"
    except:
        return "PyTorch not installed"

# List of important packages to check
packages = [
    'torch',
    'torchvision', 
    'torchaudio',
    'pytorch-lightning',
    'pytorchvideo',
    'detectron2',
    'opencv-python',
    'scikit-learn',
    'pandas',
    'numpy',
    'einops',
    'decord',
    'editdistance'
]

print("=== Package Version Check ===")
print(f"Python: {sys.version.split()[0]}")
print(f"Pip: {check_package_version('pip')}")
print()

for package in packages:
    version = check_package_version(package)
    print(f"{package:<20}: {version}")

print()
print("=== CUDA Info ===")
print(check_torch_cuda())

# Check GPU info if available
try:
    import torch
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"GPU Capability: sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}")
except:
    pass