import cv2
import numpy as np
import math

def apply_futuristic_v2(input_path, output_path):
    img = cv2.imread(input_path)
    h, w = img.shape[:2]

    # ── 1. UPSCALE for more detail headroom ──────────────────────────────────
    scale = 2
    img = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    h, w = img.shape[:2]

    # ── 2. BASE COLOR GRADE: deep night-city cyan-blue ────────────────────────
    b, g, r = cv2.split(img.astype(np.float32))
    r = np.clip(r * 0.45, 0, 255)
    g = np.clip(g * 0.75, 0, 255)
    b = np.clip(b * 1.1 + 30, 0, 255)
    tinted = cv2.merge([b, g, r]).astype(np.uint8)

    # ── 3. CLAHE contrast on luminance ───────────────────────────────────────
    lab = cv2.cvtColor(tinted, cv2.COLOR_BGR2LAB)
    l, a, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    base = cv2.cvtColor(cv2.merge([l, a, b_ch]), cv2.COLOR_LAB2BGR)

    # ── 4. MULTI-LAYER NEON EDGE GLOW ────────────────────────────────────────
    gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)

    # Fine edges
    edges_fine = cv2.Canny(gray, 40, 120)
    # Coarse structural edges
    edges_coarse = cv2.Canny(gray, 15, 60)
    edges_coarse = cv2.dilate(edges_coarse, np.ones((3,3), np.uint8), iterations=1)

    def make_glow(edges, color_bgr, blur_sizes):
        glow = np.zeros((h, w, 3), np.float32)
        for bsz, strength in blur_sizes:
            layer = np.zeros((h, w, 3), np.float32)
            for c, val in enumerate(color_bgr):
                layer[:,:,c] = edges.astype(np.float32) * (val / 255.0)
            blurred = cv2.GaussianBlur(layer, (bsz, bsz), 0)
            glow += blurred * strength
        return np.clip(glow, 0, 255).astype(np.uint8)

    # Cyan primary glow
    glow_cyan = make_glow(edges_fine, (255, 255, 0),
                          [(3,0.6), (9,0.4), (21,0.25), (41,0.1)])
    # Purple secondary glow on structure
    glow_purple = make_glow(edges_coarse, (255, 0, 200),
                            [(5,0.3), (15,0.2), (35,0.1)])
    # White hot core
    glow_white = make_glow(edges_fine, (255, 255, 255),
                           [(3,0.5)])

    result = cv2.addWeighted(base, 1.0, glow_cyan, 0.7, 0)
    result = cv2.addWeighted(result, 1.0, glow_purple, 0.4, 0)
    result = cv2.addWeighted(result, 1.0, glow_white, 0.3, 0)

    # ── 5. ATMOSPHERIC VIGNETTE (heavy outer darkness) ───────────────────────
    Y, X = np.ogrid[:h, :w]
    cx, cy = w // 2, h // 2
    dist = np.sqrt(((X - cx) / cx) ** 2 + ((Y - cy) / cy) ** 2)
    vignette = np.clip(1.0 - 0.9 * (dist ** 1.4), 0.1, 1.0).astype(np.float32)
    for c in range(3):
        result[:,:,c] = np.clip(result[:,:,c].astype(np.float32) * vignette, 0, 255).astype(np.uint8)

    # ── 6. PERSPECTIVE GRID (ground plane hologram) ──────────────────────────
    grid = np.zeros((h, w, 3), np.float32)
    horizon_y = int(h * 0.72)
    vp_x = w // 2  # vanishing point

    # Vertical perspective lines radiating from vanishing point
    num_vlines = 22
    for i in range(num_vlines + 1):
        t = i / num_vlines
        base_x = int(w * t)
        alpha = 1.0 - abs(t - 0.5) * 1.6
        alpha = max(0.0, alpha)
        color = (0, int(200 * alpha), int(255 * alpha))
        cv2.line(grid, (vp_x, horizon_y), (base_x, h), color, 1)

    # Horizontal lines (foreshortened)
    num_hlines = 14
    for i in range(num_hlines):
        t = (i / num_hlines) ** 2  # quadratic for perspective
        y = int(horizon_y + t * (h - horizon_y))
        alpha = t ** 0.5
        color = (0, int(150 * alpha), int(255 * alpha))
        cv2.line(grid, (0, y), (w, y), color, 1)

    grid_blur = cv2.GaussianBlur(grid, (3, 3), 0)
    result = cv2.addWeighted(result, 1.0, grid_blur.astype(np.uint8), 0.22, 0)

    # ── 7. SCANLINES (CRT/hologram texture) ──────────────────────────────────
    for y in range(0, h, 4):
        result[y, :] = (result[y, :].astype(np.float32) * 0.75).astype(np.uint8)

    # ── 8. CHROMATIC ABERRATION (sci-fi lens fringe) ─────────────────────────
    shift = 4
    b2, g2, r2 = cv2.split(result)
    M_r = np.float32([[1, 0, shift], [0, 1, 0]])
    M_b = np.float32([[1, 0, -shift], [0, 1, 0]])
    r2 = cv2.warpAffine(r2, M_r, (w, h))
    b2 = cv2.warpAffine(b2, M_b, (w, h))
    result = cv2.merge([b2, g2, r2])

    # ── 9. LIGHT STREAKS on bright windows ───────────────────────────────────
    bright_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)[1]
    streak_layer = np.zeros((h, w, 3), np.float32)
    # Horizontal motion blur for streak
    streak_kernel = np.zeros((1, 61), np.float32)
    streak_kernel[0, :] = np.linspace(0, 1, 31).tolist() + np.linspace(1, 0, 30).tolist()
    streak_kernel /= streak_kernel.sum()
    for c, col in enumerate([(0, 200, 255), (100, 0, 255), (0, 255, 200)]):
        lyr = np.zeros((h, w), np.float32)
        lyr[bright_mask > 0] = col[c]
        streaked = cv2.filter2D(lyr, -1, streak_kernel)
        streak_layer[:,:,c] = streaked
    streak_blur = cv2.GaussianBlur(streak_layer, (1, 7), 0)
    result = cv2.addWeighted(result, 1.0,
                             np.clip(streak_blur, 0, 255).astype(np.uint8), 0.35, 0)

    # ── 10. PARTICLE DOTS (floating data points) ─────────────────────────────
    rng = np.random.default_rng(42)
    num_particles = 280
    px = rng.integers(0, w, num_particles)
    py = rng.integers(0, int(h * 0.75), num_particles)
    sizes = rng.integers(1, 5, num_particles)
    for i in range(num_particles):
        brightness = rng.random()
        if brightness > 0.7:
            color = (int(255*brightness), int(255*brightness), 0)   # cyan
        elif brightness > 0.4:
            color = (int(200*brightness), 0, int(255*brightness))   # purple
        else:
            color = (int(255*brightness), int(255*brightness), int(255*brightness))  # white
        cv2.circle(result, (int(px[i]), int(py[i])), int(sizes[i]), color, -1)

    # ── 11. HUD OVERLAY ELEMENTS ─────────────────────────────────────────────
    hud = np.zeros((h, w, 4), np.uint8)  # BGRA for alpha compositing

    def draw_hud_rect(img, x1, y1, x2, y2, color, alpha=0.18):
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)

    # Top status bar
    draw_hud_rect(result, 0, 0, w, 55, (0, 180, 255), 0.25)
    # Bottom data bar
    draw_hud_rect(result, 0, h - 65, w, h, (0, 100, 200), 0.25)

    # Corner HUD brackets — thick & detailed
    bracket_c = (0, 255, 220)
    bracket_c2 = (100, 100, 255)
    L = 90; T = 2; mg = 20
    for (cx2, cy2, sx, sy) in [(mg, mg, 1, 1), (w-mg, mg, -1, 1),
                                (mg, h-mg, 1, -1), (w-mg, h-mg, -1, -1)]:
        cv2.line(result, (cx2, cy2), (cx2 + sx*L, cy2), bracket_c, T+1)
        cv2.line(result, (cx2, cy2), (cx2, cy2 + sy*L), bracket_c, T+1)
        cv2.line(result, (cx2 + sx*8, cy2 + sy*8),
                 (cx2 + sx*L, cy2 + sy*8), bracket_c2, 1)
        cv2.line(result, (cx2 + sx*8, cy2 + sy*8),
                 (cx2 + sx*8, cy2 + sy*L), bracket_c2, 1)
        cv2.circle(result, (cx2, cy2), 4, (255, 255, 255), -1)

    # Center crosshair
    cx2, cy2 = w//2, h//2 - 40
    cv2.line(result, (cx2-30, cy2), (cx2+30, cy2), (0,255,200), 1)
    cv2.line(result, (cx2, cy2-30), (cx2, cy2+30), (0,255,200), 1)
    cv2.circle(result, (cx2, cy2), 18, (0,255,200), 1)
    cv2.circle(result, (cx2, cy2), 5, (0,255,200), -1)

    # Side tick marks
    for y_tick in range(100, h-100, 40):
        cv2.line(result, (mg, y_tick), (mg+15, y_tick), (0, 200, 255), 1)
        cv2.line(result, (w-mg, y_tick), (w-mg-15, y_tick), (0, 200, 255), 1)

    # ── 12. HUD TEXT ─────────────────────────────────────────────────────────
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_m = cv2.FONT_HERSHEY_DUPLEX

    def put_text_glow(img, text, pos, scale, color, thickness=1):
        glow_col = tuple(min(255, int(c * 1.5)) for c in color)
        cv2.putText(img, text, pos, font, scale, glow_col, thickness+2, cv2.LINE_AA)
        cv2.putText(img, text, pos, font, scale, color, thickness, cv2.LINE_AA)

    # Top bar texts
    put_text_glow(result, "SALVARA :: IMAGIX CHALLENGE", (mg+5, 36),
                  0.75, (0, 255, 220), 1)
    put_text_glow(result, "SCAN ID: SLV-2087-FTR", (w - 380, 36),
                  0.65, (180, 255, 255), 1)

    # Bottom bar texts
    put_text_glow(result, "FUTURE SCAN ACTIVE", (mg+5, h-20),
                  0.6, (0, 255, 200), 1)
    put_text_glow(result, "STRUCTURE: ANALYZED", (w//2 - 150, h-20),
                  0.6, (100, 200, 255), 1)
    put_text_glow(result, "SYS: ONLINE  //  RES: 4K", (w-340, h-20),
                  0.55, (0, 180, 255), 1)

    # Floating data tags near building features
    tags = [
        (int(w*0.38), int(h*0.12), "ROOFTOP SENSOR ARRAY"),
        (int(w*0.55), int(h*0.30), "SOLAR GLASS: ACTIVE"),
        (int(w*0.20), int(h*0.55), "ACCESS NODE"),
        (int(w*0.65), int(h*0.60), "STRUCTURAL LOAD: 98%"),
    ]
    for tx, ty, label in tags:
        cv2.line(result, (tx, ty), (tx+60, ty-25), (0,200,255), 1)
        put_text_glow(result, label, (tx+62, ty-28), 0.38, (0,230,255), 1)
        cv2.circle(result, (tx, ty), 4, (0,255,200), -1)
        cv2.circle(result, (tx, ty), 8, (0,255,200), 1)

    # ── 13. FINAL TONE CURVE (slight S-curve punch) ───────────────────────────
    lut = np.array([
        np.clip(((i/255.0)**0.85) * 255, 0, 255)
        for i in range(256)
    ], dtype=np.uint8)
    result = cv2.LUT(result, lut)

    # ── 14. DOWNSCALE BACK to original size ──────────────────────────────────
    result = cv2.resize(result, (w//scale, h//scale), interpolation=cv2.INTER_AREA)

    cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, 97])
    print(f"Done → {output_path}")

if __name__ == "__main__":
    apply_futuristic_v2(
        "source_images/image_1.jpeg",
        "processed_images/futuristic_style.png"
    )