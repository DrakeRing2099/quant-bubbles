import numpy as np
import torch, signatory
import roughpy as rp

path_np = np.random.randn(50, 3).astype(np.float64)

# signatory (no level-0)
path_torch = torch.from_numpy(path_np[None]).float()
sig_s = signatory.signature(path_torch, depth=3).numpy().flatten()

# roughpy - skip the first element (level-0 = 1.0)
ctx = rp.get_context(width=3, depth=3, coeffs=rp.DPReal)
rp_path = rp.LieIncrementStream.from_increments(np.diff(path_np, axis=0), ctx=ctx)
sig_r = np.array(rp_path.signature(rp.RealInterval(0, 1)))[1:]  # skip level-0

print("signatory:", sig_s[:8])
print("roughpy:  ", sig_r[:8])
print("shapes:", sig_s.shape, sig_r.shape)
print("max diff: ", np.max(np.abs(sig_s - sig_r)))