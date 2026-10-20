"""Model size and inference cost of the compared methods for lesion prediction.

Parameters are counted on the trained fold-0 networks. Floating-point operations (FLOPs, two per multiply-add) of
convolution and matrix-multiplication layers are counted with torch.utils.flop_counter for one forward pass at
1024 x 256 pixels. The number of forward passes per scan is counted with forward hooks during the inference
procedure used in the evaluation (test-time flips, nnU-Net mirroring and tiling). Inference time is the GPU
wall-clock time of that procedure for one scan (anterior and posterior view), batch size 1, after warm-up; peak
memory includes the weights of the method being measured.

Our networks predict lesions with the encoder and the lesion decoder only: the anatomy decoder is a training-time
head (HBS and CVFM are disabled, so the lesion output does not depend on it). The lesion-only predictors below run
exactly the lesion path of the evaluation predictors, and the script checks that their lesion probability maps match.
Usage (from the repository root): CUDA_VISIBLE_DEVICES=1 python code/final_eval/model_complexity.py <out_json>
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_final as ev  # noqa: E402

H, W = 1024, 256
N_WARM, N_RUN = 10, 50
N_CHECK = 5  # random scans used to compare lesion-only and full predictions
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


class LesionOnlyPredictor:
    """TwoViewPredictor restricted to the lesion path: each view and its horizontal flip pass through the encoder and
    the lesion decoder; the anatomy decoder is not run (the encoder already processes the two views separately)."""

    def __init__(self, method: str, fold: int):
        self.model = ev.TwoViewPredictor(method, fold).model

    def _prob(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.model.lesion_decoder(self.model.encoder(x))[-1].float(), dim=1)[0, 1]

    @torch.no_grad()
    def __call__(self, image: torch.Tensor) -> np.ndarray:
        x = image.unsqueeze(0).to(ev.DEVICE)
        probs = []
        for v in range(2):
            xv = x[:, v:v + 1]
            p = self._prob(xv)
            pf = torch.flip(self._prob(torch.flip(xv, dims=[-1])), dims=[-1])
            probs.append(((p + pf) / 2).cpu().numpy())
        return np.stack(probs, axis=0)


class LesionOnlyEnsemble:
    def __init__(self, fold: int):
        self.members = [LesionOnlyPredictor("ours_anat", fold), LesionOnlyPredictor("ours_noanat", fold)]

    def __call__(self, image: torch.Tensor) -> np.ndarray:
        return np.mean([m(image) for m in self.members], axis=0)


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

    # lesion-only prediction: same lesion maps as the evaluation predictors, without the anatomy decoder -------
    scans = [torch.rand(2, H, W, generator=torch.Generator().manual_seed(s)) for s in range(N_CHECK)]
    full = {"ours_anat": ens.members[0], "ours_noanat": ens.members[1], "ours_ens": ens}
    lesion_only = {"ours_anat": LesionOnlyPredictor("ours_anat", FOLD),
                   "ours_noanat": LesionOnlyPredictor("ours_noanat", FOLD)}
    lesion_only["ours_ens"] = LesionOnlyEnsemble(FOLD)
    for key, pred in lesion_only.items():
        res[key]["max_abs_diff_lesion_only"] = float(max(np.abs(pred(s) - full[key](s)).max() for s in scans))
    del ens, full, lesion_only
    torch.cuda.empty_cache()
    for key in ("ours_anat", "ours_noanat", "ours_ens"):  # each measured alone, so peak memory holds only its weights
        pred = LesionOnlyEnsemble(FOLD) if key == "ours_ens" else LesionOnlyPredictor(key, FOLD)
        res[key]["time_lesion_path"] = timed(lambda: pred(image))
        del pred
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
        lesion = (f"  lesion path: {r['params_lesion_path'] / 1e6:.2f} M, {r['flops_per_scan_lesion_path'] / 1e9:.1f} "
                  f"GFLOPs, {r['time_lesion_path']['ms_mean']:.1f} ms, {r['time_lesion_path']['peak_mem_mb']:.0f} MB, "
                  f"max |diff| {r['max_abs_diff_lesion_only']:.2e}") if "time_lesion_path" in r else ""
        print(f"{k:12s} params {r['params_total'] / 1e6:6.2f} M  calls/scan {r['forward_calls_per_scan']}  "
              f"GFLOPs/scan {r['flops_per_scan'] / 1e9:8.1f}  time {r['time']['ms_mean']:.1f} ms  "
              f"peak {r['time']['peak_mem_mb']:.0f} MB" + lesion)


if __name__ == "__main__":
    main()
