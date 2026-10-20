"""Figure 2 (architecture) as a draw.io diagram with real model outputs as thumbnails.

Two columns: (a) masked-image pre-training and (c) inference on the left, (b) dual-view multi-task training with
partial labels on the right. The figure carries component names only; the losses are equations in the Methods
and the details are in the caption. Thumbnails come from results/arch_thumbs.npz (make_arch_thumbnails.py).
Writes figures/fig2_model_architecture.drawio and exports figures/Figure2_architecture.png (scale 2).
Usage: python make_figure2_drawio.py <project_dir>
"""
import base64
import io
import os
import subprocess
import sys
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
from PIL import Image

PROJECT = Path(sys.argv[1])
INK, TEAL, ORANGE, GRAY = "#12333D", "#168B8A", "#C16636", "#61747B"
TEAL_FILL, ORANGE_FILL, INK_FILL = "#E8F4F2", "#F6E5DB", "#E4EAEC"
FONT = "fontFamily=Times New Roman;fontStyle=1;"
DRAWIO = os.environ.get("DRAWIO_EXE", r"C:\Program Files\draw.io\draw.io.exe")  # draw.io desktop executable
ANATOMY_COLORS = ["#168B8A", "#12333D", "#5FB3B2", "#61747B", "#2E6B78", "#C16636", "#9CC9C7", "#3E5A63", "#7FA7A6",
                  "#A9B8BC"]
ENC = [200, 170, 145, 122, 102, 86, 72]   # encoder slab heights (relative), 7 residual stages
DEC = [72, 86, 102, 122, 145, 170]        # decoder slab heights, 6 levels


# ----------------------------------------------------------------------------- thumbnails
def to_png(rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def hex_rgb(color: str) -> np.ndarray:
    return np.array([int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)])


def thumbnails() -> dict:
    z = np.load(PROJECT / "results" / "arch_thumbs.npz")
    ant, post = z["ant"], z["post"]
    hi = np.percentile(ant[ant > 0], 99)
    gray = lambda img: np.repeat((1.0 - np.clip(img / hi, 0, 1))[..., None], 3, axis=2)
    rows = np.where(gray(ant)[..., 0].min(axis=1) < 0.97)[0]
    top, bottom = max(rows.min() - 8, 0), min(rows.max() + 8, ant.shape[0])
    crop = lambda a: a[top:bottom]

    def overlay(base, mask, color, alpha=0.75):
        a = (alpha * np.clip(mask, 0, 1))[..., None]
        return base * (1 - a) + hex_rgb(color) * a

    masked = gray(ant)
    masked[z["mask"] < 0.5] = 0.62  # masked patches drawn as flat gray
    labels = z["labels"][0]
    anat = gray(ant) * 0.35 + 0.65
    for k, col in enumerate(ANATOMY_COLORS, start=1):
        anat = np.where((labels == k)[..., None], hex_rgb(col), anat)
    return {
        "ant": to_png(crop(gray(ant))), "post": to_png(crop(gray(post))), "masked": to_png(crop(masked)),
        "recon": to_png(crop(gray(z["recon"]))), "target": to_png(crop(gray(ant))),
        "lesion": to_png(crop(overlay(gray(ant), z["p_anat"][0], ORANGE))),
        "anatomy": to_png(crop(anat)),
        "final": to_png(crop(overlay(gray(ant), (z["ensemble"][0] >= 0.5).astype(float), ORANGE, 0.85))),
    }


# ----------------------------------------------------------------------------- draw.io helpers
class Diagram:
    def __init__(self):
        self.cells, self.n = [], 2

    def _id(self):
        self.n += 1
        return f"c{self.n}"

    def box(self, x, y, w, h, text="", fill="none", stroke=INK, size=22, color=INK, rounded=1, align="center"):
        cid = self._id()
        style = (f"rounded={rounded};whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};strokeWidth=2;"
                 f"{FONT}fontSize={size};fontColor={color};align={align};verticalAlign=middle;arcSize=8;")
        self.cells.append(f'<mxCell id="{cid}" value="{escape(text, {chr(34): "&quot;"})}" style="{style}" '
                          f'vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')

    def text(self, x, y, w, h, text, size=22, color=INK, align="center"):
        self.box(x, y, w, h, text, fill="none", stroke="none", size=size, color=color, rounded=0, align=align)

    def image(self, x, y, w, h, b64):
        cid = self._id()
        style = f"shape=image;html=1;imageAspect=0;aspect=fixed;image=data:image/png,{b64};imageBorder={GRAY};"
        self.cells.append(f'<mxCell id="{cid}" value="" style="{style}" vertex="1" parent="1">'
                          f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')

    def ellipse(self, x, y, d, color):
        cid = self._id()
        style = f"ellipse;whiteSpace=wrap;html=1;fillColor={color};strokeColor={color};strokeWidth=1.5;"
        self.cells.append(f'<mxCell id="{cid}" value="" style="{style}" vertex="1" parent="1">'
                          f'<mxGeometry x="{x}" y="{y}" width="{d}" height="{d}" as="geometry"/></mxCell>')

    def arrow(self, x1, y1, x2, y2, color=GRAY, dashed=0, width=2.5, end="block", points=()):
        cid = self._id()
        style = (f"endArrow={end};endFill=1;html=1;strokeColor={color};strokeWidth={width};dashed={dashed};"
                 f"edgeStyle=none;rounded=0;")
        pts = "".join(f'<mxPoint x="{px}" y="{py}"/>' for px, py in points)
        arr = f'<Array as="points">{pts}</Array>' if points else ""
        self.cells.append(f'<mxCell id="{cid}" value="" style="{style}" edge="1" parent="1"><mxGeometry relative="1" '
                          f'as="geometry"><mxPoint x="{x1}" y="{y1}" as="sourcePoint"/><mxPoint x="{x2}" y="{y2}" '
                          f'as="targetPoint"/>{arr}</mxGeometry></mxCell>')

    def stack(self, x, yc, heights, w, gap, fill, stroke):
        """Feature-map slabs centred on yc; returns the x of the right edge."""
        for h in heights:
            self.box(x, yc - h / 2, w, h, fill=fill, stroke=stroke, rounded=0)
            x += w + gap
        return x - gap

    def xml(self, width, height):
        return (f'<mxfile host="python"><diagram name="Figure 2" id="fig2"><mxGraphModel dx="{width}" dy="{height}" '
                f'grid="0" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" '
                f'pageWidth="{width}" pageHeight="{height}" math="0" shadow="0"><root><mxCell id="0"/>'
                f'<mxCell id="1" parent="0"/>{"".join(self.cells)}</root></mxGraphModel></diagram></mxfile>')


def main():
    t = thumbnails()
    d = Diagram()
    W, H = 1240, 730
    IW, IH = 54, 214  # thumbnail size

    # (a) left top: masked-image pre-training -----------------------------------------------------------
    d.text(10, 4, 600, 36, "(a) Stage 1: masked-image pre-training", size=24, align="left")
    ya, yc = 50, 50 + IH / 2
    d.image(20, ya, IW, IH, t["masked"])
    d.arrow(80, yc, 100, yc)
    xe = d.stack(104, yc, [h * 0.95 for h in ENC], 17, 6, TEAL_FILL, TEAL)
    d.arrow(xe + 6, yc, xe + 26, yc)
    xr = d.stack(xe + 30, yc, [h * 0.95 for h in DEC], 17, 6, INK_FILL, GRAY)
    d.arrow(xr + 6, yc, xr + 26, yc)
    d.image(xr + 30, ya, IW, IH, t["recon"])
    d.image(xr + 110, ya, IW, IH, t["target"])
    yl = ya + IH + 4
    d.text(-6, yl, 106, 30, "masked", size=22, color=GRAY)
    d.text(104, yl, xe - 104, 30, "encoder E", size=22, color=TEAL)
    d.text(xe + 30, yl, xr - xe - 30, 30, "decoder", size=22, color=GRAY)
    d.text(xr + 15, yl, IW + 30, 30, "output", size=22, color=GRAY)
    d.text(xr + 95, yl, IW + 30, 30, "target", size=22, color=GRAY)

    # (c) left bottom: inference --------------------------------------------------------------------------
    y0 = 360
    d.text(10, y0, 600, 36, "(c) Inference", size=24, align="left")
    yi = y0 + 46
    d.image(20, yi, IW, IH, t["ant"])
    y1, y2 = yi + 55, yi + IH - 55
    d.arrow(80, yi + IH / 2, 104, y1, points=[(92, yi + IH / 2), (92, y1)])
    d.arrow(80, yi + IH / 2, 104, y2, points=[(92, yi + IH / 2), (92, y2)])
    d.box(108, y1 - 38, 170, 76, "with the<br>anatomy task", fill=TEAL_FILL, stroke=TEAL)
    d.box(108, y2 - 38, 170, 76, "without the<br>anatomy task", fill=INK_FILL, stroke=GRAY)
    ym = yi + IH / 2
    d.arrow(282, y1, 308, ym - 12, points=[(296, y1), (296, ym - 12)])
    d.arrow(282, y2, 308, ym + 12, points=[(296, y2), (296, ym + 12)])
    d.box(312, ym - 42, 100, 84, "mean of<br>4 maps")
    d.arrow(416, ym, 434, ym)
    d.box(438, ym - 26, 66, 52, "&ge; 0.5", stroke=ORANGE, color=ORANGE)
    d.arrow(508, ym, 526, ym)
    d.image(530, yi, IW, IH, t["final"])
    d.text(0, yi + IH + 4, 94, 30, "test view", size=22, color=GRAY)
    d.text(512, yi + IH + 4, 90, 30, "lesions", size=22, color=ORANGE)

    # (b) right: dual-view multi-task training with partial labels ----------------------------------------
    x0 = 640
    d.text(x0, 4, 600, 36, "(b) Stage 2: dual-view multi-task training", size=24, align="left")
    ia, ip = 56, 360
    xi, xs = x0 - 4, x0 + 96  # input thumbnails, first encoder stage
    d.image(xi, ia, IW, IH, t["ant"])
    d.image(xi, ip, IW, IH, t["post"])
    d.text(xi - 20, ia + IH + 2, IW + 40, 30, "anterior", size=22, color=GRAY)
    d.text(xi - 30, ip + IH + 2, IW + 60, 54, "posterior<br>(mirrored)", size=22, color=GRAY)
    ycb = 320
    d.arrow(xi + IW + 4, 230, xs - 4, 230)
    d.arrow(xi + IW + 4, 410, xs - 4, 410)
    enc = [h * 1.25 for h in ENC]
    xe = d.stack(xs, ycb, enc, 20, 6, TEAL_FILL, TEAL)
    d.text(xs - 40, ycb + 132, 174, 56, "encoder E<br>(shared weights)", size=22, color=TEAL)
    yl_, ya_ = 163, 477
    # the bottleneck feeds the first level of each decoder
    d.arrow(xe + 6, ycb - 30, xe + 34, yl_ + 12, points=[(xe + 20, ycb - 30), (xe + 20, yl_ + 12)])
    d.arrow(xe + 6, ycb + 30, xe + 34, ya_ - 12, points=[(xe + 20, ycb + 30), (xe + 20, ya_ - 12)])
    # one of the six skip connections per decoder: the last stage before the bottleneck to the first decoder level
    xk = xs + 5 * 26 + 14
    d.arrow(xk, ycb - enc[5] / 2, xe + 34, yl_ - 22, dashed=1, width=1.8, end="open", points=[(xk, yl_ - 22)])
    d.arrow(xk, ycb + enc[5] / 2, xe + 34, ya_ + 22, dashed=1, width=1.8, end="open", points=[(xk, ya_ + 22)])
    xl = d.stack(xe + 38, yl_, DEC, 18, 6, ORANGE_FILL, ORANGE)
    xa = d.stack(xe + 38, ya_, DEC, 18, 6, INK_FILL, INK)
    for k in range(6):  # deep-supervision outputs above each decoder level
        cx = xe + 38 + k * 24 + 4
        d.ellipse(cx, yl_ - DEC[k] / 2 - 16, 10, ORANGE)
        d.ellipse(cx, ya_ - DEC[k] / 2 - 16, 10, INK)
    xc = (xe + 38 + xl) / 2
    d.text(xc - 100, yl_ + 92, 200, 30, "lesion decoder", size=22, color=ORANGE)
    d.text(xc - 100, ya_ + 92, 200, 30, "anatomy decoder", size=22, color=INK)
    d.arrow(xl + 6, yl_, xl + 26, yl_)
    d.arrow(xa + 6, ya_, xa + 26, ya_)
    d.image(xl + 30, yl_ - IH / 2, IW, IH, t["lesion"])
    d.image(xa + 30, ya_ - IH / 2, IW, IH, t["anatomy"])
    d.text(xl + 6, yl_ + IH / 2 + 2, IW + 60, 30, "lesions", size=22, color=ORANGE)
    d.text(xa + 6, ya_ + IH / 2 + 2, IW + 60, 30, "skeleton", size=22, color=INK)

    # legend -----------------------------------------------------------------------------------------------
    yg = H - 42
    d.box(20, yg + 6, 20, 28, fill=TEAL_FILL, stroke=TEAL, rounded=0)
    d.text(46, yg, 200, 40, "residual stage", size=22, align="left")
    d.ellipse(260, yg + 15, 11, ORANGE)
    d.text(278, yg, 280, 40, "deep-supervision output", size=22, align="left")
    d.arrow(570, yg + 20, 616, yg + 20, dashed=1, width=1.8, end="open")
    d.text(622, yg, 220, 40, "skip connection", size=22, align="left")

    src_dir = PROJECT / "figures"
    src = src_dir / "fig2_model_architecture.drawio"
    old = src_dir / "fig2_model_architecture_v1_simple.drawio"
    if src.exists() and not old.exists():
        src.replace(old)  # keep the earlier simple version
    src.write_text(d.xml(W, H), encoding="utf-8")
    out = PROJECT / "figures" / "Figure2_architecture.png"
    subprocess.run([DRAWIO, "--export", "--format", "png", "--scale", "2", "--border", "10", "--output", str(out),
                    str(src)], check=True, capture_output=True)
    print("saved", src, "and", out)


if __name__ == "__main__":
    main()
