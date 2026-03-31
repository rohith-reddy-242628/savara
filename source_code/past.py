import cv2
import numpy as np

def apply_past_theme(input_path, output_path):
    img = cv2.imread(input_path)
    h, w = img.shape[:2]

    # ── 1. UPSCALE for detail headroom ───────────────────────────────────────
    scale = 2
    img = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    h, w = img.shape[:2]

    # ── 2. SEPIA TONE BASE ────────────────────────────────────────────────────
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sepia = np.zeros_like(img, dtype=np.float32)
    sepia[:,:,0] = gray * 0.75   # Blue
    sepia[:,:,1] = gray * 0.95   # Green
    sepia[:,:,2] = gray * 1.15   # Red
    sepia = np.clip(sepia, 0, 255).astype(np.uint8)

    # ── 3. AGED YELLOW-BROWN TINT ─────────────────────────────────────────────
    b, g, r = cv2.split(sepia.astype(np.float32))
    r = np.clip(r * 1.08 + 18, 0, 255)
    g = np.clip(g * 0.92 + 5,  0, 255)
    b = np.clip(b * 0.65 - 10, 0, 255)
    aged = cv2.merge([b, g, r]).astype(np.uint8)

    # ── 4. CONTRAST BOOST (old photo punch) ──────────────────────────────────
    lab = cv2.cvtColor(aged, cv2.COLOR_BGR2LAB)
    l, a, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    base = cv2.cvtColor(cv2.merge([l, a, b_ch]), cv2.COLOR_LAB2BGR)

    # ── 5. FILM GRAIN (analog noise) ─────────────────────────────────────────
    rng = np.random.default_rng(7)
    grain = rng.normal(0, 18, (h, w, 3)).astype(np.float32)
    base = np.clip(base.astype(np.float32) + grain, 0, 255).astype(np.uint8)

    # ── 6. HEAVY VIGNETTE (dark rounded edges like old lens) ─────────────────
    Y, X = np.ogrid[:h, :w]
    cx, cy = w // 2, h // 2
    dist = np.sqrt(((X - cx) / cx) ** 2 + ((Y - cy) / cy) ** 2)
    vignette = np.clip(1.0 - 1.05 * (dist ** 1.6), 0.05, 1.0).astype(np.float32)
    for c in range(3):
        base[:,:,c] = np.clip(base[:,:,c].astype(np.float32) * vignette, 0, 255).astype(np.uint8)

    # ── 7. SCRATCHES & DUST (vertical film scratches) ────────────────────────
    scratch_layer = np.zeros((h, w), np.float32)
    rng2 = np.random.default_rng(21)
    num_scratches = 14
    for _ in range(num_scratches):
        x = rng2.integers(0, w)
        length = rng2.integers(h // 4, h)
        y_start = rng2.integers(0, h - length)
        thickness = 1 if rng2.random() > 0.3 else 2
        intensity = rng2.uniform(0.4, 0.9)
        cv2.line(scratch_layer, (x, y_start), (x, y_start + length),
                 intensity, thickness)

    # Dust spots
    num_dust = 60
    dust_x = rng2.integers(0, w, num_dust)
    dust_y = rng2.integers(0, h, num_dust)
    dust_r = rng2.integers(1, 5, num_dust)
    for i in range(num_dust):
        val = rng2.uniform(0.3, 0.8)
        cv2.circle(scratch_layer, (int(dust_x[i]), int(dust_y[i])),
                   int(dust_r[i]), val, -1)

    scratch_blur = cv2.GaussianBlur(scratch_layer, (3, 3), 0)
    # Apply scratches as bright marks
    for c in range(3):
        base[:,:,c] = np.clip(
            base[:,:,c].astype(np.float32) + scratch_blur * 120, 0, 255
        ).astype(np.uint8)

    # ── 8. HORIZONTAL SCAN LINES (old photo halftone feel) ───────────────────
    for y in range(0, h, 5):
        base[y, :] = (base[y, :].astype(np.float32) * 0.82).astype(np.uint8)

    # ── 9. FADED EDGE BURN (darker corners, lighter center) ──────────────────
    burn = np.ones((h, w), np.float32)
    corner_pts = [(0,0),(w-1,0),(0,h-1),(w-1,h-1)]
    for (bx, by) in corner_pts:
        d = np.sqrt(((X - bx) / w) ** 2 + ((Y - by) / h) ** 2)
        burn *= np.clip(1.0 - 0.5 * np.exp(-d * 3.5), 0.5, 1.0)
    for c in range(3):
        base[:,:,c] = np.clip(base[:,:,c].astype(np.float32) * burn, 0, 255).astype(np.uint8)

    # ── 10. PAPER TEXTURE OVERLAY (crumpled old paper) ───────────────────────
    paper = np.zeros((h, w), np.float32)
    # Multi-frequency Perlin-like noise using blurred random layers
    for freq, amp in [(3,0.08),(7,0.06),(15,0.04),(31,0.03)]:
        small = rng2.random((h // freq + 2, w // freq + 2)).astype(np.float32)
        big   = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        paper += big * amp
    paper = cv2.GaussianBlur(paper, (5, 5), 0)
    for c in range(3):
        base[:,:,c] = np.clip(
            base[:,:,c].astype(np.float32) * (1.0 + paper * 0.25), 0, 255
        ).astype(np.uint8)

    # ── 11. LIGHT LEAK (warm orange bloom top-left corner) ───────────────────
    leak = np.zeros((h, w, 3), np.float32)
    for y in range(h):
        for pass_ in range(1):  # vectorised below
            pass
    Y2, X2 = np.ogrid[:h, :w]
    leak_dist = np.sqrt((X2 / w) ** 2 + (Y2 / h) ** 2)
    leak_mask = np.clip(1.0 - leak_dist * 1.8, 0, 1) ** 2
    leak[:,:,0] = leak_mask * 30    # slight blue
    leak[:,:,1] = leak_mask * 60    # green
    leak[:,:,2] = leak_mask * 120   # strong red-orange
    base = np.clip(base.astype(np.float32) + leak, 0, 255).astype(np.uint8)

    # ── 12. BORDER: AGED PHOTO FRAME ─────────────────────────────────────────
    frame_w = 28
    frame_color = (30, 55, 80)   # dark brownish
    cv2.rectangle(base, (0, 0), (w-1, h-1), frame_color, frame_w * 2)

    # Inner worn border line
    inner = frame_w + 8
    line_color = (60, 90, 115)
    cv2.rectangle(base, (inner, inner), (w-inner, h-inner), line_color, 2)

    # ── 13. PHOTO CAPTION AREA (bottom white strip like old photo) ───────────
    caption_h = 90
    caption_strip = np.ones((caption_h, w, 3), np.float32)
    # Aged paper color for strip
    caption_strip[:,:,0] = 185   # B
    caption_strip[:,:,1] = 200   # G
    caption_strip[:,:,2] = 220   # R  → warm cream
    caption_noise = rng2.normal(0, 8, (caption_h, w, 3))
    caption_strip = np.clip(caption_strip + caption_noise, 0, 255).astype(np.uint8)
    base[h - caption_h:, :] = caption_strip

    # ── 14. HANDWRITTEN-STYLE TEXT ────────────────────────────────────────────
    font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
    font2 = cv2.FONT_HERSHEY_TRIPLEX

    ink = (25, 35, 70)     # dark ink color
    ink_fade = (60, 75, 110)

    cv2.putText(base, "circa 1940s",
                (frame_w + 20, h - caption_h + 42),
                font, 1.1, ink, 2, cv2.LINE_AA)
    cv2.putText(base, "Architecture of the Era  |  Hand Developed",
                (frame_w + 20, h - caption_h + 72),
                font2, 0.52, ink_fade, 1, cv2.LINE_AA)
    cv2.putText(base, "SALVARA :: IMAGIX",
                (w - 420, h - caption_h + 72),
                font2, 0.52, ink_fade, 1, cv2.LINE_AA)

    # ── 15. FINAL SOFT BLUR (old lens softness) ───────────────────────────────
    base = cv2.GaussianBlur(base, (3, 3), 0.6)

    # ── 16. TONE CURVE: Faded highlights (old photo look) ────────────────────
    lut = np.array([
        int(np.clip(255 * (1 - np.exp(-2.2 * (i / 255))) * 0.96 + 8, 0, 255))
        for i in range(256)
    ], dtype=np.uint8)
    base = cv2.LUT(base, lut)

    # ── 17. DOWNSCALE BACK ────────────────────────────────────────────────────
    result = cv2.resize(base, (w // scale, h // scale), interpolation=cv2.INTER_AREA)

    cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, 97])
    print(f"Done → {output_path}")


if __name__ == "__main__":
    apply_past_theme(
        "source_images/image_1.jpeg", #source image path
        "processed_images/past_style.png" #result image path
    )