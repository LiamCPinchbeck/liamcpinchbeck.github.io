#!/usr/bin/env python3
"""Generate social preview cards (1200x630 PNG) for the site and each blog post.

LinkedIn, X, Slack, etc. show these images when a link is shared.

    python3 scripts/make-post-cards.py           # make cards for new posts only
    python3 scripts/make-post-cards.py --force   # regenerate every card

Outputs
    images/og-image.png        default card for the site (used by non-post pages)
    images/og/<post-file>.png  one card per post in _posts/ (picked up by _includes/seo.html)

Requires Pillow and PyYAML (Ubuntu: sudo apt install python3-pil python3-yaml).
A post can opt out of its auto card by setting `header: og_image: some-image.png`.
"""
import argparse
import re
import sys
from datetime import date, datetime
from pathlib import Path

try:
    import yaml
    from PIL import Image, ImageDraw, ImageFont
except ImportError as e:  # pragma: no cover
    sys.exit(f"Missing dependency ({e}). On Ubuntu: sudo apt install python3-pil python3-yaml")

ROOT = Path(__file__).resolve().parent.parent
W, H = 1200, 630

# Mint theme palette (see _sass/theme/_mint_light.scss)
TEAL = (17, 153, 158)
DEEP = (9, 78, 82)
MINT = (48, 227, 202)
WHITE = (255, 255, 255)
SOFT = (214, 243, 240)

AUTHOR = "Liam C. Pinchbeck"
SITE = "liamcpinchbeck.github.io"
TAGLINE = "Bayesian inference · Machine learning · Statistics"

FONT_DIRS = [Path("/usr/share/fonts/truetype/dejavu"), Path("/usr/share/fonts/TTF"), Path("/Library/Fonts")]


def font(bold, size):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for d in FONT_DIRS:
        if (d / name).exists():
            return ImageFont.truetype(str(d / name), size)
    return ImageFont.load_default()


def slugify(s):
    """Match Jekyll's default `slugify` filter."""
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def background():
    """Diagonal teal gradient with a couple of soft decorative circles."""
    img = Image.new("RGB", (W, H), TEAL)
    px = img.load()
    for y in range(H):
        for x in range(0, W, 2):
            t = (x / W) * 0.55 + (y / H) * 0.45
            c = tuple(int(TEAL[i] * (1 - t) + DEEP[i] * t) for i in range(3))
            px[x, y] = c
            if x + 1 < W:
                px[x + 1, y] = c
    over = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    d.ellipse((W - 330, -170, W + 170, 330), fill=MINT + (38,))
    d.ellipse((W - 170, H - 150, W + 90, H + 110), fill=WHITE + (22,))
    img = Image.alpha_composite(img.convert("RGBA"), over)
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, 14, H), fill=MINT + (255,))  # accent bar
    return img, d


def wrap(draw, text, fnt, max_w):
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if draw.textlength(trial, font=fnt) <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def fit_title(draw, title, max_w, max_h):
    for size in (72, 66, 60, 54, 48, 44, 40, 36):
        f = font(True, size)
        lines = wrap(draw, title, f, max_w)
        lh = int(size * 1.18)
        if len(lines) * lh <= max_h:
            return f, lines, lh
    f = font(True, 36)
    lines = wrap(draw, title, f, max_w)[:5]
    lines[-1] = lines[-1].rstrip(".,;:") + "…"
    return f, lines, int(36 * 1.18)


def footer(d):
    d.line((70, H - 92, W - 70, H - 92), fill=(120, 200, 200), width=2)
    d.text((70, H - 72), AUTHOR, font=font(True, 28), fill=WHITE)
    fs = font(False, 24)
    d.text((W - 70 - d.textlength(SITE, font=fs), H - 69), SITE, font=fs, fill=SOFT)


def post_card(title, when, tags, out):
    img, d = background()
    d.text((70, 62), ("BLOG POST · " + when).upper(), font=font(True, 24), fill=MINT)
    f, lines, lh = fit_title(d, title, W - 140, 300)
    y = 112
    for line in lines:
        d.text((70, y), line, font=f, fill=WHITE)
        y += lh
    # tag pills: translucent shapes go on an overlay so they blend, then text on top
    fx, x, ty = font(False, 22), 70, H - 150
    pills = []
    for t in tags[:4]:
        tw = d.textlength(t, font=fx)
        if x + tw + 32 > W - 70:
            break
        pills.append((x, t, tw))
        x += tw + 42
    over = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(over)
    for px, t, tw in pills:
        od.rounded_rectangle((px, ty, px + tw + 30, ty + 38), radius=19, fill=WHITE + (40,), outline=WHITE + (130,), width=2)
    img = Image.alpha_composite(img, over)
    d = ImageDraw.Draw(img)
    for px, t, tw in pills:
        d.text((px + 15, ty + 6), t, font=fx, fill=WHITE)
    footer(d)
    img.convert("RGB").save(out, optimize=True)


def site_card(out):
    img, d = background()
    d.text((70, 170), AUTHOR, font=font(True, 78), fill=WHITE)
    d.text((72, 275), TAGLINE, font=font(False, 34), fill=SOFT)
    d.text((72, 340), "Tutorials, projects and notes", font=font(False, 30), fill=MINT)
    footer(d)
    img.convert("RGB").save(out, optimize=True)


def front_matter(path):
    text = path.read_text(encoding="utf-8")
    m = re.match(r"^---\s*\n(.*?)\n---", text, flags=re.S)
    return yaml.safe_load(m.group(1)) or {} if m else {}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="regenerate existing cards")
    args = ap.parse_args()

    og_dir = ROOT / "images" / "og"
    og_dir.mkdir(parents=True, exist_ok=True)

    default = ROOT / "images" / "og-image.png"
    if args.force or not default.exists() or default.stat().st_size == 46978:  # replace template's default card
        site_card(default)
        print("wrote", default.relative_to(ROOT))

    made = 0
    for post in sorted((ROOT / "_posts").glob("*.md")):
        out = og_dir / (slugify(post.stem) + ".png")
        if out.exists() and not args.force:
            continue
        fm = front_matter(post)
        title = str(fm.get("title", post.stem))
        when = fm.get("date") or post.name[:10]
        if isinstance(when, (date, datetime)):
            when = when.strftime("%d %b %Y")
        tags = [str(t) for t in (fm.get("tags") or [])]
        post_card(title, str(when), tags, out)
        made += 1
        print("wrote", out.relative_to(ROOT))
    print(f"done: {made} post card(s)")


if __name__ == "__main__":
    main()
