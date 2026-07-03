import torch

print("===== H200 Test Run Start =====")
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU count:", torch.cuda.device_count())
    print("GPU name:", torch.cuda.get_device_name(0))

    x = torch.randn(1000, 1000).cuda()
    y = torch.matmul(x, x)
    print("Tensor device:", y.device)
    print("Result mean:", y.mean().item())
else:
    print("CUDA is not available. GPU was not detected.")

print("===== H200 Test Run Finished =====")
