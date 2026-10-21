"""Draws the app icons (the ellipse-and-eigenvectors logo) into public/icons.

    python3 scripts/make_icons.py      (needs Pillow)
"""
from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent.parent / "public" / "icons"
BG = (26, 33, 48, 255)
RING = (47, 199, 147, 255)
FILL = (47, 199, 147, 60)
BLUE = (94, 145, 255, 255)
PINK = (255, 110, 146, 255)
DOT = (232, 236, 244, 255)
SS = 4  # supersampling factor for smooth edges


def logo(size, scale):
    """The 40-unit SVG logo drawn on a transparent square, `scale` = share of the canvas it fills."""
    big = size * SS
    u = big * scale / 40  # pixels per logo unit
    off = (big - 40 * u) / 2
    P = lambda x, y: (off + x * u, off + y * u)

    ring = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    d = ImageDraw.Draw(ring)
    cx, cy = P(20, 20)
    rx, ry = 16.5 * u, 9.5 * u
    d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=FILL, outline=RING, width=round(2.6 * u))
    ring = ring.rotate(30, resample=Image.BICUBIC, center=(cx, cy))

    d = ImageDraw.Draw(ring)
    w = round(3.2 * u)
    for (x, y), color in (((34.3, 11.8), BLUE), ((24.8, 28.2), PINK)):
        a, b = P(20, 20), P(x, y)
        d.line([a, b], fill=color, width=w)
        r = w / 2
        d.ellipse([b[0] - r, b[1] - r, b[0] + r, b[1] + r], fill=color)
    r = 2.8 * u
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=DOT)
    return ring


def icon(size, scale, rounded):
    big = size * SS
    base = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    d = ImageDraw.Draw(base)
    if rounded:
        d.rounded_rectangle([0, 0, big - 1, big - 1], radius=int(big * 0.22), fill=BG)
    else:
        d.rectangle([0, 0, big, big], fill=BG)
    base.alpha_composite(logo(size, scale))
    return base.resize((size, size), Image.LANCZOS)


OUT.mkdir(parents=True, exist_ok=True)
icon(192, 0.78, True).save(OUT / "icon-192.png")
icon(512, 0.78, True).save(OUT / "icon-512.png")
# Maskable icons get cropped to a circle or squircle, so keep the logo inside the central 80%.
icon(512, 0.62, False).save(OUT / "maskable-512.png")
icon(180, 0.72, False).convert("RGB").save(OUT / "apple-touch-icon.png")
icon(32, 0.9, True).save(OUT / "favicon-32.png")
print("wrote", ", ".join(sorted(p.name for p in OUT.glob("*.png"))))
