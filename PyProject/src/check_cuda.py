# -*- coding: utf-8 -*-
import torch
import sys
import subprocess

def check_cuda():
    print("=== PyTorch & CUDA Diagnostic ===")
    print(f"Python Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Device Count: {torch.cuda.device_count()}")
        print(f"Current Device Name: {torch.cuda.get_device_name(0)}")
        print(f"cuDNN Enabled: {torch.backends.cudnn.enabled}")
    else:
        print("\n--- Why is CUDA not available? ---")
        # Check if NVIDIA drivers are installed
        try:
            nvidia_smi = subprocess.check_output("nvidia-smi", shell=True).decode()
            print("NVIDIA-SMI: OK (Drivers are installed)")
            print("Possible Reason: You might have installed the CPU-only version of PyTorch.")
            print("To fix, try: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        except:
            print("NVIDIA-SMI: FAILED (NVIDIA Drivers or GPU not found)")
            print("Possible Reason: Your NVIDIA drivers are missing or outdated.")

if __name__ == "__main__":
    check_cuda()
