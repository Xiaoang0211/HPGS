import torch
import lpips

def main():
    # 1) build LPIPS and move it (and its internal buffers) to GPU
    model = lpips.LPIPS(net='vgg').eval().cuda()
    
    # 2) two dummy inputs on GPU, same H×W you’ll use in C++
    H, W = 256, 256
    dummy1 = torch.randn(1, 3, H, W, device='cuda')
    dummy2 = torch.randn(1, 3, H, W, device='cuda')

    # 3) trace on CUDA — this will bake all constants (scale/shift) as CUDA tensors
    traced = torch.jit.trace(model, (dummy1, dummy2), strict=False)

    # 4) save out
    traced.save("lpips_vgg.pt")
    print("Traced LPIPS → lpips_vgg.pt (with CUDA constants)")

if __name__ == "__main__":
    main()
