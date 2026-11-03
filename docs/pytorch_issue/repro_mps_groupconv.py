# Repro for pytorch/pytorch#202452: tall-filter depthwise conv2d on MPS is wrong at some batch sizes,
# and the BACKWARD pass (weight grad and input grad) is wrong as well, not only the forward pass.
# Layer = EEGNet spatial conv (braindecode): Conv2d(8, 16, (64, 1), groups=8), input (batch, 8, 64, T).
import torch, torch.nn.functional as F

def check(batch, T, kh=64, cin=8, cout=16, groups=8):
    torch.manual_seed(0)
    w = torch.randn(cout, cin // groups, kh, 1)
    x = torch.randn(batch, cin, kh, T, requires_grad=True)
    wr = w.clone().requires_grad_(True)
    ref = F.conv2d(x, wr, groups=groups); ref.square().mean().backward()
    xm = x.detach().to("mps").requires_grad_(True); wm = w.to("mps").requires_grad_(True)
    out = F.conv2d(xm, wm, groups=groups); out.square().mean().backward()
    fwd = (ref.detach() - out.cpu()).abs().max().item()
    gw = ((wr.grad - wm.grad.cpu()).abs().max() / wr.grad.abs().max()).item()
    gx = ((x.grad - xm.grad.cpu()).abs().max() / x.grad.abs().max()).item()
    return fwd, ref.abs().max().item(), gw, gx

print(torch.__version__)
for T in (1001, 64):
    bad = {b: tuple(round(v, 3) for v in check(b, T)) for b in range(1, 65) if max(check(b, T)[0], check(b, T)[2], check(b, T)[3]) > 1e-3}
    print(f"Conv2d(8,16,(64,1),groups=8), width {T}: failing batch sizes in 1..64")
    print("   batch: (fwd max|diff|, fwd max|ref|, weight-grad rel diff, input-grad rel diff)")
    for b, v in bad.items(): print(f"   {b}: {v}")
print("Reporter's layer Conv2d(8,8,(200,1),groups=8), width 64:")
for b in (3, 16, 32):
    print(f"   batch {b}: {tuple(round(v, 3) for v in check(b, 64, kh=200, cout=8))}")
# workaround: run the grouped conv in two batch halves
torch.manual_seed(0); w = torch.randn(16, 1, 64, 1); x = torch.randn(32, 8, 64, 1001)
a = F.conv2d(x, w, groups=8)
b = torch.cat([F.conv2d(c, w.to("mps"), groups=8) for c in x.to("mps").chunk(2)]).cpu()
print("batch 32 split into two halves of 16, max|diff| vs CPU:", (a - b).abs().max().item())
