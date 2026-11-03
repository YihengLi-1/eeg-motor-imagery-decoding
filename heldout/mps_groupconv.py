"""Work around a PyTorch-MPS bug (seen on torch 2.14.1): grouped/depthwise convolution gives wrong
forward AND backward results when the batch size is a multiple of 32 (16, 31, 33 are fine; 32, 64, 128, 256 wrong).
EEGNet's conv_spatial (groups=8, tall (64,1) kernel) hits this with our batch size 32 (the (1,16) separable depthwise conv is fine; the defect needs a tall filter, see pytorch/pytorch#202452), which left
every from-scratch EEGNet run at chance (train acc 0.50) while LaBraM/CBraMod (no grouped conv) were unaffected.
Fix: run grouped convs on MPS in two batch halves. Importing this module patches torch.nn.functional.conv1d/conv2d
and runs a self-check against the CPU result; it raises if the patched op still disagrees with CPU.
Found 2026-11-01 (diagnosis: docs/mps_groupconv_bug.md). Upstream: pytorch/pytorch#202452 (open 2026-10-28, same defect,
filter height 64-255 via MPSGraph; #188335 fixed only height>=256). ponytail: remove once a torch release closes #202452."""
import torch, torch.nn.functional as F

_orig = {"conv1d": F.conv1d, "conv2d": F.conv2d}

def _wrap(name):
    orig = _orig[name]
    def conv(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        if groups > 1 and input.device.type == "mps" and input.shape[0] % 32 == 0 and input.shape[0] > 0:
            # split recursively: halves of 64/128/256 are still multiples of 32 (fixed 2026-11-03; the one-level
            # split left batch 64 and 128 wrong, see docs/mps_groupconv_bug.md "Second defect")
            return torch.cat([conv(c, weight, bias, stride, padding, dilation, groups) for c in input.chunk(2)])
        return orig(input, weight, bias, stride, padding, dilation, groups)
    conv.__name__ = name
    return conv

F.conv1d = _wrap("conv1d")
F.conv2d = _wrap("conv2d")

def self_check():
    """Depthwise conv at batch 32, 64 and 128: patched MPS result must match CPU (forward and weight-grad). Raises on failure."""
    if not torch.backends.mps.is_available():
        return "mps unavailable; patch inert"
    for n in (64, 128, 32):                        # 32 last: its tensors feed the "bug present" check below
        torch.manual_seed(0)
        conv = torch.nn.Conv2d(8, 16, (64, 1), groups=8, bias=False)
        x = torch.randn(n, 8, 64, 101, requires_grad=True)
        ref = conv(x); ref.square().mean().backward(); gref = conv.weight.grad.clone(); conv.weight.grad = None
        cm = conv.to("mps"); xm = x.detach().to("mps").requires_grad_(True)
        out = cm(xm); out.square().mean().backward()
        fd = float((ref - out.cpu()).abs().max()); gd = float((gref - cm.weight.grad.cpu()).abs().max()) / float(gref.abs().max())
        if fd > 1e-3 or gd > 1e-3:
            raise RuntimeError(f"mps_groupconv: patched grouped conv still disagrees with CPU at batch {n} (fwd {fd:.2e}, grad rel {gd:.2e})")
    # confirm the bug is real on this torch (unpatched) so we know the patch is still needed
    raw = _orig["conv2d"](xm.detach(), cm.weight.detach(), None, 1, 0, 1, 8)
    bug = float((ref.detach() - raw.cpu()).abs().max()) > 1e-3
    return f"mps_groupconv OK (fwd diff {fd:.1e}, grad rel {gd:.1e}); unpatched bug present={bug} torch {torch.__version__}"

_STATUS = self_check()

if __name__ == "__main__":
    print(_STATUS)
