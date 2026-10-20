"""Model size and inference cost of the compared methods.

Parameters are counted on the trained fold-0 networks. Floating-point operations (FLOPs, two per multiply-add) of
convolution and matrix-multiplication layers are counted with torch.utils.flop_counter for one forward pass at
1024 x 256 pixels. The number of forward passes per scan is counted with forward hooks during the inference
procedure used in the evaluation (test-time flips, nnU-Net mirroring and tiling). Inference time is the GPU
wall-clock time of that procedure for one scan (anterior and posterior view), batch size 1, after warm-up; peak
memory includes the weights of the method being measured.
Usage (from the repository root): CUDA_VISIBLE_DEVICES=1 python code/final_eval/model_complexity.py <out_json>
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_final as ev  # noqa: E402

H, W = 1024, 256
N_WARM, N_RUN = 10, 50
FOLD = 0


def n_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def count_flops(fn) -> int:
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        fn()
    return int(counter.get_total_flops())


def count_calls(modules: list[torch.nn.Module], fn) -> int:
    calls = []
    hooks = [m.register_forward_hook(lambda *_: calls.append(1)) for m in modules]
    fn()
    for h in hooks:
        h.remove()
    return len(calls)


def timed(fn) -> dict:
    for _ in range(N_WARM):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(N_RUN):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return {"ms_mean": float(np.mean(times)), "ms_sd": float(np.std(times, ddof=1)),
            "peak_mem_mb": float(torch.cuda.max_memory_allocated() / 2 ** 20)}


def ours_entry(method: str, x: torch.Tensor) -> dict:
    m = ev.TwoViewPredictor(method, FOLD).model
    parts = {"encoder": n_params(m.encoder), "lesion_decoder": n_params(m.lesion_decoder),
             "anatomy_decoder": n_params(m.anat_decoder)}
    entry = {"params_total": n_params(m), "params_parts": parts,
             "params_lesion_path": parts["encoder"] + parts["lesion_decoder"],
             "flops_forward_both_views": count_flops(lambda: m(x, x)),
             "flops_forward_both_views_lesion_path": 2 * count_flops(lambda: m.lesion_decoder(m.encoder(x)))}
    del m
    torch.cuda.empty_cache()
    return entry


def main() -> None:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    image = torch.rand(2, H, W)  # one scan in the dataset frame: anterior and mirrored posterior view
    x = torch.rand(1, 1, H, W, device=ev.DEVICE)
    views = [rng.random((H, W), dtype=np.float32) for _ in range(2)]
    out = {"input": [H, W], "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "n_warmup": N_WARM, "n_runs": N_RUN, "fold": FOLD, "methods": {}}
    res = out["methods"]

    # size and FLOPs per forward pass ---------------------------------------------------------------------
    for method in ("ours_anat", "ours_noanat"):
        res[method] = ours_entry(method, x)
    m = ev.TwoViewPredictor("bsciseg", FOLD).model
    res["bsciseg"] = {"params_total": n_params(m), "flops_forward_both_views": count_flops(lambda: m(x, x))}
    del m
    nn_pred = ev.NnUNetViewPredictor(FOLD)
    net = nn_pred.predictor.network.to(ev.DEVICE).eval()
    res["nnunet"] = {"params_total": n_params(net), "flops_forward_one_view": count_flops(lambda: net(x)),
                     "network_class": type(net).__name__}

    # forward passes per scan and timing of the evaluation procedure ----------------------------------------
    nn_run = lambda: [nn_pred.predictor.predict_single_npy_array(v[None, None], {"spacing": [999.0, 1.0, 1.0]},
                                                                  None, None, True) for v in views]
    res["nnunet"]["forward_calls_per_scan"] = count_calls([net], nn_run)
    res["nnunet"]["time"] = timed(nn_run)
    del nn_pred, net
    torch.cuda.empty_cache()

    for method in ("ours_anat", "ours_noanat", "bsciseg"):
        pred = ev.TwoViewPredictor(method, FOLD)
        res[method]["forward_calls_per_scan"] = count_calls([pred.model], lambda: pred(image))
        res[method]["time"] = timed(lambda: pred(image))
        del pred
        torch.cuda.empty_cache()

    ens = ev.EnsemblePredictor(FOLD)
    res["ours_ens"] = {"forward_calls_per_scan": count_calls([mm.model for mm in ens.members], lambda: ens(image)),
                       "time": timed(lambda: ens(image))}
    del ens
    torch.cuda.empty_cache()

    # derived totals per scan -------------------------------------------------------------------------------
    for k in ("ours_anat", "ours_noanat", "bsciseg"):
        r = res[k]
        r["flops_per_scan"] = r["forward_calls_per_scan"] * r["flops_forward_both_views"]
        if "flops_forward_both_views_lesion_path" in r:
            r["flops_per_scan_lesion_path"] = r["forward_calls_per_scan"] * r["flops_forward_both_views_lesion_path"]
    res["nnunet"]["flops_per_scan"] = res["nnunet"]["forward_calls_per_scan"] * res["nnunet"]["flops_forward_one_view"]
    a, n = res["ours_anat"], res["ours_noanat"]
    res["ours_ens"].update({
        "params_total": a["params_total"] + n["params_total"],
        "params_lesion_path": a["params_lesion_path"] + n["params_lesion_path"],
        "flops_per_scan": a["flops_per_scan"] + n["flops_per_scan"],
        "flops_per_scan_lesion_path": a["flops_per_scan_lesion_path"] + n["flops_per_scan_lesion_path"]})

    Path(sys.argv[1]).write_text(json.dumps(out, indent=2))
    for k, r in res.items():
        print(f"{k:12s} params {r['params_total'] / 1e6:6.2f} M"
              + (f" (lesion path {r['params_lesion_path'] / 1e6:.2f} M)" if "params_lesion_path" in r else "")
              + f"  calls/scan {r['forward_calls_per_scan']}  GFLOPs/scan {r['flops_per_scan'] / 1e9:8.1f}"
              + (f" (lesion path {r['flops_per_scan_lesion_path'] / 1e9:.1f})" if "flops_per_scan_lesion_path" in r else "")
              + f"  time {r['time']['ms_mean']:.1f} +/- {r['time']['ms_sd']:.1f} ms  peak {r['time']['peak_mem_mb']:.0f} MB")


if __name__ == "__main__":
    main()
