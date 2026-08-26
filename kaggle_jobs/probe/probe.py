"""Probe: make the P100 usable and measure real training throughput."""
import subprocess, sys, time, textwrap

t0 = time.time()
import torch
print("preinstalled torch:", torch.__version__)
props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
if props:
    sm = "sm_%d%d" % (props.major, props.minor)
    print("device:", props.name, "|", sm, "| %.1f GB" % (props.total_memory/1e9))
    print("torch supports:", torch.cuda.get_arch_list())
    compat = sm in torch.cuda.get_arch_list()
else:
    sm, compat = None, False
print("compatible:", compat)

if not compat:
    print("\n=== installing a Pascal-compatible PyTorch (sm_60) ===", flush=True)
    t1 = time.time()
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "torch==2.5.1", "torchvision==0.20.1",
                        "--index-url", "https://download.pytorch.org/whl/cu121"],
                       capture_output=True, text=True)
    print("pip rc=%d in %.1f min" % (r.returncode, (time.time()-t1)/60))
    print((r.stdout or "")[-1500:])
    print((r.stderr or "")[-1500:])

bench = textwrap.dedent('''
    import time, torch, torchvision, torch.nn as nn
    print("torch now:", torch.__version__, "| arch list:", torch.cuda.get_arch_list())
    p = torch.cuda.get_device_properties(0)
    print("device:", p.name, "sm_%d%d" % (p.major, p.minor))
    x = torch.randn(2048, 2048, device="cuda")
    t0=time.time()
    for _ in range(20): x = torch.mm(x, x) * 1e-4
    torch.cuda.synchronize()
    print("GPU COMPUTE OK: 20x 2048^2 matmul in %.3fs" % (time.time()-t0))

    net = torchvision.models.resnet50(weights=None).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    for B in (24, 48):
        img = torch.randn(B,3,224,224, device="cuda")
        tgt = torch.randint(0,2,(B,),device="cuda").float()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                loss = nn.functional.binary_cross_entropy_with_logits(net(img)[:,0], tgt)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        torch.cuda.synchronize(); t0=time.time()
        for _ in range(6):
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                loss = nn.functional.binary_cross_entropy_with_logits(net(img)[:,0], tgt)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        torch.cuda.synchronize(); dt=(time.time()-t0)/6
        print("RESNET50 TRAIN batch %3d: %.3fs/step = %5.0f img/s | PEAK VRAM %.2f GB"
              % (B, dt, B/dt, torch.cuda.max_memory_allocated()/1e9))
''')
r = subprocess.run([sys.executable, "-c", bench], capture_output=True, text=True)
print(r.stdout)
print(r.stderr[-2000:] if r.returncode else "")
print("total probe time: %.1f min" % ((time.time()-t0)/60))
