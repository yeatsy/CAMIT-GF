import torch
import sys

def check_mps():
    print(f"PyTorch version: {torch.__version__}")
    
    # Check if MPS is built into PyTorch
    if not hasattr(torch.backends, 'mps'):
        print("MPS is not available in this PyTorch build.")
        print("Please install PyTorch 1.12 or newer with MPS support.")
        return False
    
    print(f"MPS built: {torch.backends.mps.is_built()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    
    # If MPS is available, run a simple test
    if torch.backends.mps.is_available():
        print("\nMPS is available! Running a simple test...")
        
        # Create a device
        device = torch.device("mps")
        print(f"Created device: {device}")
        
        # Create tensors on MPS
        try:
            x = torch.rand(5, 5, device=device)
            y = torch.rand(5, 5, device=device)
            z = x @ y  # Matrix multiplication
            print("\nTest successful! MPS is working correctly.")
            print(f"Sample calculation result:\n{z}")
            return True
        except Exception as e:
            print(f"\nError during MPS test: {e}")
            return False
    else:
        print("\nMPS is not available on this system.")
        
        # Check common issues
        if torch.backends.mps.is_built():
            print("MPS is built into PyTorch, but not available. Possible reasons:")
            print("- You're not running on Apple Silicon (M1/M2/M3)")
            print("- macOS version is too old (need macOS 12.3+)")
        else:
            print("MPS is not built into this PyTorch version.")
            print("Please install PyTorch 1.12 or newer with MPS support:")
            print("pip install torch torchvision torchaudio")
        
        return False

if __name__ == "__main__":
    print("Testing MPS (Metal Performance Shaders) support...\n")
    
    # Print system info
    import platform
    print(f"Python version: {platform.python_version()}")
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Machine: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    print("")
    
    # Check MPS
    success = check_mps()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1) 