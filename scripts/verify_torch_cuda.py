"""Verify PyTorch CUDA passthrough works in WSL2."""
import sys
import torch

print(f"Python:           {sys.version.split()[0]}")
print(f"PyTorch:          {torch.__version__}")
print(f"CUDA available:   {torch.cuda.is_available()}")
print(f"CUDA version:     {torch.version.cuda}")

if torch.cuda.is_available():
    n = torch.cuda.device_count()
    print(f"Device count:     {n}")
    for i in range(n):
        print(f"  Device {i}:    {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        mem_gb = props.total_memory / (1024 ** 3)
        print(f"    VRAM:        {mem_gb:.1f} GB")
        print(f"    Capability:  {props.major}.{props.minor}")

    # Quick smoke test: matmul on GPU
    a = torch.randn(1024, 1024, device="cuda:0")
    b = torch.randn(1024, 1024, device="cuda:0")
    c = a @ b
    torch.cuda.synchronize()
    print(f"  Matmul test:   OK (output shape {tuple(c.shape)})")
    print(f"  VRAM used:     {torch.cuda.memory_allocated() / (1024 ** 2):.1f} MB")
else:
    print("CUDA NOT available — bitsandbytes 4-bit quantization sẽ KHÔNG hoạt động.")
    sys.exit(1)
