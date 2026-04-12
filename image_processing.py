from PIL import Image

RESIZE_METHODS = {
    'NEAREST': Image.Resampling.NEAREST,
    'BOX': Image.Resampling.BOX,
    'BILINEAR': Image.Resampling.BILINEAR,
    'HAMMING': Image.Resampling.HAMMING,
    'BICUBIC': Image.Resampling.BICUBIC,
    'LANCZOS': Image.Resampling.LANCZOS,
}


def _apply_sharpen(img, conf):
    if conf['SHARPEN']:
        from PIL import ImageFilter
        img = img.filter(ImageFilter.UnsharpMask(
            radius=conf['SHARPEN_RADIUS'],
            percent=conf['SHARPEN_PERCENT'],
            threshold=conf['SHARPEN_THRESHOLD']
        )) 
    return img


def _apply_blur(img, conf):
    if conf['GAUSSIAN_BLUR']:
        from PIL import ImageFilter
        img = img.filter(ImageFilter.GaussianBlur(radius=conf['GAUSSIAN_BLUR_RADIUS']))
    return img


def _apply_color_to_transparent(img, conf):
    if not conf.get('COLOR_TO_TRANSPARENT'):
        return img
    tc = _parse_color(conf.get('COLOR_TO_TRANSPARENT_COLOR', [0, 0, 0]))
    threshold = conf.get('COLOR_TO_TRANSPARENT_THRESHOLD', 0)
    img = img.convert('RGBA')
    pixels = img.load()
    for y in range(img.height):
        for x in range(img.width):
            r, g, b, a = pixels[x, y]
            if abs(r - tc[0]) < threshold and abs(g - tc[1]) < threshold and abs(b - tc[2]) < threshold:
                pixels[x, y] = (r, g, b, 0)
    return img


def _apply_contrast(img, conf):
    if conf['CONTRAST']:
        from PIL import ImageEnhance
        img = ImageEnhance.Contrast(img).enhance(conf['CONTRAST_FACTOR'])
    return img


def _apply_exposure(img, conf):
    if conf['EXPOSURE']:
        import numpy as np
        factor = conf['EXPOSURE_FACTOR']
        arr = np.array(img, dtype=np.float32)
        if img.mode == 'RGBA':
            arr[..., :3] = np.clip(arr[..., :3] * factor, 0, 255)
        else:
            arr = np.clip(arr * factor, 0, 255)
        img = Image.fromarray(arr.astype(np.uint8), img.mode)
    return img


def _apply_gamma(img, conf):
    if conf['GAMMA']:
        import numpy as np
        gamma = conf['GAMMA_VALUE']
        inv_gamma = 1.0 / gamma
        arr = np.array(img, dtype=np.float32)
        if img.mode == 'RGBA':
            arr[..., :3] = np.clip(255.0 * (arr[..., :3] / 255.0) ** inv_gamma, 0, 255)
        else:
            arr = np.clip(255.0 * (arr / 255.0) ** inv_gamma, 0, 255)
        img = Image.fromarray(arr.astype(np.uint8), img.mode)
    return img


def _apply_alpha_outline(img, conf):
    if conf['ALPHA_OUTLINE']:
        import numpy as np

        thickness = conf['ALPHA_OUTLINE_THICKNESS']
        outline_color = _parse_color(conf['ALPHA_OUTLINE_COLOR'])

        img = img.convert('RGBA')
        arr = np.array(img)
        alpha = arr[..., 3]
        opaque = alpha > 0

        source = opaque if thickness < 0 else ~opaque
        dist = np.zeros(source.shape, dtype=np.float32)
        dist[source] = float('inf')
        remaining = source.copy()
        for i in range(1, abs(thickness) + 1):
            eroded = remaining.copy()
            eroded[1:] &= remaining[:-1]
            eroded[:-1] &= remaining[1:]
            eroded[:, 1:] &= remaining[:, :-1]
            eroded[:, :-1] &= remaining[:, 1:]
            border = remaining & ~eroded
            dist[border] = i
            remaining = eroded

        abs_t = abs(thickness)
        mask = (dist >= 1) & (dist <= abs_t)
        mode = conf['ALPHA_OUTLINE_LERPCOLOR']
        oc = np.array(outline_color, dtype=np.float32)

        edge_rgb = arr[..., :3].copy()
        if thickness > 0 and mode != 'newcolor':
            filled = opaque.copy()
            for _ in range(abs_t):
                expanded = filled.copy()
                for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                    shifted = np.roll(np.roll(filled, dy, 0), dx, 1)
                    shifted_rgb = np.roll(np.roll(edge_rgb, dy, 0), dx, 1)
                    new_pixels = shifted & ~expanded
                    for c in range(3):
                        edge_rgb[..., c][new_pixels] = shifted_rgb[..., c][new_pixels]
                    expanded |= shifted
                filled = expanded
        elif thickness < 0 and mode != 'newcolor':
            border1 = opaque & ~(np.roll(opaque,1,0) & np.roll(opaque,-1,0) & np.roll(opaque,1,1) & np.roll(opaque,-1,1))
            filled = border1.copy()
            for _ in range(abs_t):
                expanded = filled.copy()
                for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                    shifted = np.roll(np.roll(filled, dy, 0), dx, 1)
                    shifted_rgb = np.roll(np.roll(edge_rgb, dy, 0), dx, 1)
                    new_pixels = shifted & ~expanded & opaque
                    for c in range(3):
                        edge_rgb[..., c][new_pixels] = shifted_rgb[..., c][new_pixels]
                    expanded |= shifted
                filled = expanded

        grad = np.clip(1.0 - (dist[mask] - 1) / max(abs_t - 1, 1), 0, 1)

        if mode == 'bordercolor':
            for c in range(3):
                arr[..., c][mask] = edge_rgb[..., c][mask]
        elif mode == 'newcolor':
            for c in range(3):
                arr[..., c][mask] = int(oc[c])
        elif mode == 'border_to_new':
            for c in range(3):
                orig = edge_rgb[..., c][mask].astype(np.float32)
                arr[..., c][mask] = (orig * grad + oc[c] * (1.0 - grad)).astype(np.uint8)
        elif mode == 'new_to_border':
            for c in range(3):
                orig = edge_rgb[..., c][mask].astype(np.float32)
                arr[..., c][mask] = (oc[c] * grad + orig * (1.0 - grad)).astype(np.uint8)

        if thickness > 0:
            arr[..., 3][mask] = 255

        img = Image.fromarray(arr, 'RGBA')
    return img


COLOR_MAP = {
    'black': (0, 0, 0), 'white': (255, 255, 255), 'red': (255, 0, 0),
    'green': (0, 255, 0), 'blue': (0, 0, 255), 'yellow': (255, 255, 0),
    'cyan': (0, 255, 255), 'magenta': (255, 0, 255), 'light_gray': (192, 192, 192),
    'dark_gray': (64, 64, 64), 'orange': (255, 165, 0), 'purple': (128, 0, 128)
}


def _parse_color(c):
    if isinstance(c, (list, tuple)) and len(c) == 3:
        return tuple(int(x) for x in c)
    if isinstance(c, str) and c.startswith('#'):
        h = c.lstrip('#')
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    if isinstance(c, str):
        return COLOR_MAP.get(c, (0, 0, 0))
    return (0, 0, 0)


# INVERT examples for .custom_processing:
#   Invert all pixels:
#     INVERT = True
#   Invert only near a specific color:
#     INVERT = True
#     INVERT_COLOR = white
#     INVERT_THRESHOLD = 40
def _apply_invert(img, conf):
    if conf.get('INVERT'):
        import numpy as np
        img = img.convert('RGBA')
        arr = np.array(img)
        color = conf.get('INVERT_COLOR', [0, 0, 0])
        threshold = conf.get('INVERT_THRESHOLD', 0)
        if threshold > 0:
            tc = np.array(_parse_color(color), dtype=np.float32)
            diff = np.abs(arr[..., :3].astype(np.float32) - tc)
            mask = np.all(diff < threshold, axis=-1)
            arr[..., :3][mask] = 255 - arr[..., :3][mask]
        else:
            arr[..., :3] = 255 - arr[..., :3]
        img = Image.fromarray(arr, 'RGBA')
    return img


# PIXELATE examples for .custom_processing:
#   Basic pixelate (divide resolution by 4):
#     PIXELATE = True
#     PIXELATE_LEVEL = 4
#   Pixelate with specific resampling:
#     PIXELATE = True
#     PIXELATE_LEVEL = 8
#     PIXELATE_DOWN_METHOD = NEAREST
#     PIXELATE_UP_METHOD = NEAREST
def _apply_pixelate(img, conf):
    if conf.get('PIXELATE'):
        level = conf.get('PIXELATE_LEVEL', 2)
        down_method = RESIZE_METHODS.get(conf.get('PIXELATE_DOWN_METHOD', 'NEAREST').upper(), Image.Resampling.NEAREST)
        up_method = RESIZE_METHODS.get(conf.get('PIXELATE_UP_METHOD', 'NEAREST').upper(), Image.Resampling.NEAREST)
        w, h = img.size
        small_w, small_h = max(1, w // level), max(1, h // level)
        img = img.resize((small_w, small_h), down_method).resize((w, h), up_method)
    return img


# FILL examples for .custom_processing:
#   Fill transparent pixels with black:
#     FILL = True
#     FILL_COLOR = black
#     FILL_THRESHOLD = 128
#   Fill with hex color, low threshold:
#     FILL = True
#     FILL_COLOR = #ff0000
#     FILL_THRESHOLD = 50
def _apply_fill(img, conf):
    if conf.get('FILL'):
        import numpy as np
        fill_color = _parse_color(conf.get('FILL_COLOR', [0, 0, 0]))
        threshold = conf.get('FILL_THRESHOLD', 128)
        img = img.convert('RGBA')
        arr = np.array(img)
        mask = arr[..., 3] < threshold
        arr[..., 0][mask] = fill_color[0]
        arr[..., 1][mask] = fill_color[1]
        arr[..., 2][mask] = fill_color[2]
        arr[..., 3][mask] = 255
        img = Image.fromarray(arr, 'RGBA')
    return img


def _apply_colorize(img, conf):
    if not conf.get('COLORIZE'):
        return img
    import numpy as np
    color = _parse_color(conf.get('COLORIZE_COLOR', [255, 255, 255]))
    threshold = conf.get('COLORIZE_THRESHOLD', 0)
    img = img.convert('RGBA')
    arr = np.array(img)
    mask = arr[..., 3] > threshold
    arr[..., 0][mask] = color[0]
    arr[..., 1][mask] = color[1]
    arr[..., 2][mask] = color[2]
    img = Image.fromarray(arr, 'RGBA')
    return img


# COLOR_REPLACE examples for .custom_processing:
#   Replace red with white:
#     COLOR_REPLACE_SRC = red
#     COLOR_REPLACE_THRESHOLD = 100
#     COLOR_REPLACE_DST = white
def _apply_color_replace(img, conf):
    if not conf.get('COLOR_REPLACE'):
        return img
    import numpy as np
    thresh = conf.get('COLOR_REPLACE_THRESHOLD', 100)
    img = img.convert('RGBA')
    arr = np.array(img)
    sc = np.array(_parse_color(conf.get('COLOR_REPLACE_SRC', [0, 0, 0])), dtype=np.float32)
    dc = _parse_color(conf.get('COLOR_REPLACE_DST', [0, 0, 0]))
    diff = np.abs(arr[..., :3].astype(np.float32) - sc)
    mask = np.all(diff < thresh, axis=-1) if thresh > 0 else np.all(arr[..., :3] == sc.astype(np.uint8), axis=-1)
    arr[..., 0][mask] = dc[0]
    arr[..., 1][mask] = dc[1]
    arr[..., 2][mask] = dc[2]
    img = Image.fromarray(arr, 'RGBA')
    return img


# RECTANGLES examples for .custom_processing:
#   Overlay with fill and border:
#     RECT_AX = 0.1
#     RECT_AY = 0.1
#     RECT_BX = 0.9
#     RECT_BY = 0.9
#     RECT_MODE = overlay
#     RECT_FILL = True
#     RECT_FILL_COLOR = red
#     RECT_BORDER = 3
#     RECT_BORDER_COLOR = white
#     RECT_ROUNDNESS = 0.2
#
# All rectangle options:
#   RECT_AX, RECT_AY, RECT_BX, RECT_BY — corners, normalized 0.0-1.0
#   RECT_MODE       — "overlay", "subtract", "intersect"
#   RECT_FILL       — true/false
#   RECT_FILL_COLOR — color name, hex "#ff0000"
#   RECT_BORDER     — pixel thickness (overlay only)
#   RECT_BORDER_COLOR — color (overlay only)
#   RECT_CUT_BORDER — pixel thickness at new alpha edge (subtract/intersect only)
#   RECT_CUT_BORDER_COLOR — color (subtract/intersect only)
#   RECT_ROUNDNESS  — 0.0 (sharp) to 1.0 (fully rounded)
def _apply_rectangle(img, conf):
    if not conf.get('RECTANGLE'):
        return img
    import numpy as np
    img = img.convert('RGBA')
    w, h = img.size
    ax = conf.get('RECT_AX', 0.0)
    ay = conf.get('RECT_AY', 0.0)
    bx = conf.get('RECT_BX', 1.0)
    by = conf.get('RECT_BY', 1.0)
    mode = conf.get('RECT_MODE', 'overlay')
    fill = conf.get('RECT_FILL', False)
    fill_color = _parse_color(conf.get('RECT_FILL_COLOR', [255, 255, 255]))
    border = conf.get('RECT_BORDER', 0)
    border_color = _parse_color(conf.get('RECT_BORDER_COLOR', [255, 255, 255]))
    roundness = conf.get('RECT_ROUNDNESS', 0.0)

    x0, x1 = int(min(ax, bx) * w), int(max(ax, bx) * w)
    y0, y1 = int(min(ay, by) * h), int(max(ay, by) * h)
    x0, x1 = max(0, x0), min(w, x1)
    y0, y1 = max(0, y0), min(h, y1)

    rw, rh = x1 - x0, y1 - y0
    if rw <= 0 or rh <= 0:
        return img

    yy, xx = np.mgrid[y0:y1, x0:x1]
    if roundness > 0:
        radius = roundness * min(rw, rh) / 2
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        hw, hh = rw / 2 - radius, rh / 2 - radius
        dx = np.clip(np.abs(xx - cx) - hw, 0, None)
        dy = np.clip(np.abs(yy - cy) - hh, 0, None)
        inside = (dx**2 + dy**2) <= radius**2
    else:
        inside = np.ones((rh, rw), dtype=bool)

    if border > 0:
        inner = np.zeros_like(inside)
        iy0, iy1 = border, rh - border
        ix0, ix1 = border, rw - border
        if iy1 > iy0 and ix1 > ix0:
            sub_yy = yy[iy0:iy1, ix0:ix1]
            sub_xx = xx[iy0:iy1, ix0:ix1]
            if roundness > 0:
                ir = max(0, radius - border)
                ihw, ihh = max(0, hw), max(0, hh)
                idx = np.clip(np.abs(sub_xx - cx) - ihw, 0, None)
                idy = np.clip(np.abs(sub_yy - cy) - ihh, 0, None)
                inner[iy0:iy1, ix0:ix1] = (idx**2 + idy**2) <= ir**2
            else:
                inner[iy0:iy1, ix0:ix1] = True
        border_mask_local = inside & ~inner
    else:
        border_mask_local = np.zeros_like(inside)
        inner = inside

    fill_mask_local = inner if fill else np.zeros_like(inside)

    arr = np.array(img)

    combined = np.zeros_like(inside)
    if fill:
        combined |= fill_mask_local
    if border > 0:
        combined |= border_mask_local
    full_shape = np.zeros((h, w), dtype=bool)
    full_shape[y0:y1, x0:x1] = combined

    if mode == 'overlay':
        if fill:
            fm = np.zeros((h, w), dtype=bool)
            fm[y0:y1, x0:x1] = fill_mask_local
            for c in range(3):
                arr[..., c][fm] = fill_color[c]
            arr[..., 3][fm] = 255
        if border > 0:
            bm = np.zeros((h, w), dtype=bool)
            bm[y0:y1, x0:x1] = border_mask_local
            for c in range(3):
                arr[..., c][bm] = border_color[c]
            arr[..., 3][bm] = 255
    elif mode == 'subtract':
        alpha_before = arr[..., 3].copy()
        arr[..., 3][full_shape] = 0
        cut_border = conf.get('RECT_CUT_BORDER', 0)
        if cut_border > 0:
            cut_color = _parse_color(conf.get('RECT_CUT_BORDER_COLOR', [255, 255, 255]))
            new_edge = (alpha_before > 0) & (arr[..., 3] == 0)
            edge_band = new_edge.copy()
            for _ in range(cut_border - 1):
                expanded = edge_band.copy()
                expanded[1:] |= edge_band[:-1]
                expanded[:-1] |= edge_band[1:]
                expanded[:, 1:] |= edge_band[:, :-1]
                expanded[:, :-1] |= edge_band[:, 1:]
                edge_band = expanded
            edge_band &= (alpha_before > 0) & (arr[..., 3] == 0)
            inward = edge_band.copy()
            for _ in range(cut_border):
                shrunk = inward.copy()
                shrunk[1:] &= inward[:-1]
                shrunk[:-1] &= inward[1:]
                shrunk[:, 1:] &= inward[:, :-1]
                shrunk[:, :-1] &= inward[:, 1:]
                inward = shrunk
            neighbor_opaque = np.zeros((h, w), dtype=bool)
            neighbor_opaque[1:] |= (alpha_before[:-1] > 0) & (arr[..., 3][:-1] > 0)
            neighbor_opaque[:-1] |= (alpha_before[1:] > 0) & (arr[..., 3][1:] > 0)
            neighbor_opaque[:, 1:] |= (alpha_before[:, :-1] > 0) & (arr[..., 3][:, :-1] > 0)
            neighbor_opaque[:, :-1] |= (alpha_before[:, 1:] > 0) & (arr[..., 3][:, 1:] > 0)
            seed = new_edge & neighbor_opaque
            cut_mask = seed.copy()
            for _ in range(cut_border - 1):
                expanded = cut_mask.copy()
                expanded[1:] |= cut_mask[:-1]
                expanded[:-1] |= cut_mask[1:]
                expanded[:, 1:] |= cut_mask[:, :-1]
                expanded[:, :-1] |= cut_mask[:, 1:]
                cut_mask = expanded
            cut_mask &= (alpha_before > 0)
            for c in range(3):
                arr[..., c][cut_mask] = cut_color[c]
            arr[..., 3][cut_mask] = 255
    elif mode == 'intersect':
        alpha_before = arr[..., 3].copy()
        arr[..., 3][~full_shape] = 0
        cut_border = conf.get('RECT_CUT_BORDER', 0)
        if cut_border > 0:
            cut_color = _parse_color(conf.get('RECT_CUT_BORDER_COLOR', [255, 255, 255]))
            still_opaque = (arr[..., 3] > 0)
            lost = (alpha_before > 0) & ~still_opaque
            neighbor_lost = np.zeros((h, w), dtype=bool)
            neighbor_lost[1:] |= lost[:-1]
            neighbor_lost[:-1] |= lost[1:]
            neighbor_lost[:, 1:] |= lost[:, :-1]
            neighbor_lost[:, :-1] |= lost[:, 1:]
            seed = still_opaque & neighbor_lost
            cut_mask = seed.copy()
            for _ in range(cut_border - 1):
                expanded = cut_mask.copy()
                expanded[1:] |= cut_mask[:-1]
                expanded[:-1] |= cut_mask[1:]
                expanded[:, 1:] |= cut_mask[:, :-1]
                expanded[:, :-1] |= cut_mask[:, 1:]
                cut_mask = expanded
            cut_mask &= still_opaque
            for c in range(3):
                arr[..., c][cut_mask] = cut_color[c]
            arr[..., 3][cut_mask] = 255

    img = Image.fromarray(arr, 'RGBA')
    return img


# LINES examples for .custom_processing:
#   Diagonal line corner to corner:
#     LINE_AX = 0.0
#     LINE_AY = 0.0
#     LINE_BX = 1.0
#     LINE_BY = 1.0
#     LINE_COLOR = white
#     LINE_THICKNESS = 2
#   Coordinates are normalized 0.0-1.0 (fraction of image size)
def _apply_line(img, conf):
    if not conf.get('LINE'):
        return img
    color_name = conf.get('LINE_COLOR', [255, 255, 255])
    thickness = conf.get('LINE_THICKNESS', 1)
    lax = conf.get('LINE_AX', 0.0)
    lay = conf.get('LINE_AY', 0.0)
    lbx = conf.get('LINE_BX', 1.0)
    lby = conf.get('LINE_BY', 1.0)
    import numpy as np
    img = img.convert('RGBA')
    arr = np.array(img)
    h, w = arr.shape[:2]
    color = _parse_color(color_name)
    x0, y0 = int(lax * w), int(lay * h)
    x1, y1 = int(lbx * w), int(lby * h)
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    half = thickness // 2
    while True:
        for oy in range(-half, half + 1):
            for ox in range(-half, half + 1):
                px, py = x0 + ox, y0 + oy
                if 0 <= px < w and 0 <= py < h:
                    arr[py, px, :3] = color
                    arr[py, px, 3] = 255
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    img = Image.fromarray(arr, 'RGBA')
    return img


# EDGE_OUTLINE examples for .custom_processing:
#   White edge outline, 2px:
#     EDGE_OUTLINE = True
#     EDGE_OUTLINE_THICKNESS = 2
#     EDGE_OUTLINE_COLOR = white
#   Red edge outline from image border inward:
#     EDGE_OUTLINE = True
#     EDGE_OUTLINE_THICKNESS = 5
#     EDGE_OUTLINE_COLOR = #ff0000
def _apply_edge_outline(img, conf):
    if conf.get('EDGE_OUTLINE'):
        import numpy as np
        img = img.convert('RGBA')
        arr = np.array(img)
        h, w = arr.shape[:2]
        alpha = arr[..., 3]
        thickness = conf.get('EDGE_OUTLINE_THICKNESS', 2)
        color = _parse_color(conf.get('EDGE_OUTLINE_COLOR', [255, 255, 255]))
        edge_mask = np.zeros((h, w), dtype=bool)
        edge_mask[0, :] = True
        edge_mask[h-1, :] = True
        edge_mask[:, 0] = True
        edge_mask[:, w-1] = True
        seeds = edge_mask & (alpha > 0)
        outline = seeds.copy()
        for _ in range(thickness - 1):
            expanded = outline.copy()
            expanded[1:] |= outline[:-1]
            expanded[:-1] |= outline[1:]
            expanded[:, 1:] |= outline[:, :-1]
            expanded[:, :-1] |= outline[:, 1:]
            expanded &= (alpha > 0)
            outline = expanded
        for c in range(3):
            arr[..., c][outline] = color[c]
        arr[..., 3][outline] = 255
        img = Image.fromarray(arr, 'RGBA')
    return img


# GRAIN examples for .custom_processing:
#   Subtle B&W grain:
#     GRAIN = True
#     GRAIN_MODE = bw
#     GRAIN_SIZE = 1
#     GRAIN_ROUGHNESS = 0.2
#   Coarse color grain:
#     GRAIN = True
#     GRAIN_MODE = color
#     GRAIN_SIZE = 4
#     GRAIN_ROUGHNESS = 0.8
def _apply_grain(img, conf):
    if conf.get('GRAIN'):
        import numpy as np
        img = img.convert('RGBA')
        arr = np.array(img, dtype=np.float32)
        h, w = arr.shape[:2]
        mode = conf.get('GRAIN_MODE', 'bw')
        size = max(1, conf.get('GRAIN_SIZE', 1))
        roughness = conf.get('GRAIN_ROUGHNESS', 0.5)
        gh, gw = max(1, -(-h // size)), max(1, -(-w // size))
        if mode == 'bw':
            noise = np.random.uniform(-1, 1, (gh, gw)).astype(np.float32)
            noise = np.repeat(np.repeat(noise, size, axis=0), size, axis=1)[:h, :w]
            noise = noise[..., np.newaxis] * np.ones(3)
        else:
            noise = np.random.uniform(-1, 1, (gh, gw, 3)).astype(np.float32)
            noise = np.repeat(np.repeat(noise, size, axis=0), size, axis=1)[:h, :w]
        arr[..., :3] = np.clip(arr[..., :3] + noise * roughness * 255, 0, 255)
        img = Image.fromarray(arr.astype(np.uint8), 'RGBA')
    return img


def _apply_dither(img, conf):
    if conf['DITHERING']:
        dither_map = {
            'floyd_steinberg': Image.Dither.FLOYDSTEINBERG,
            'ordered': Image.Dither.ORDERED,
            'none': Image.Dither.NONE
        }
        
        has_alpha = img.mode == 'RGBA'
        alpha_channel = img.split()[3] if has_alpha else None
        
        if conf['DITHER_MODE'] == 'bw':
            rgb_img = img.convert('L')
            dithered = rgb_img.convert('1', dither=dither_map[conf['DITHER_METHOD']])
            dithered = dithered.convert('RGB')
        
        elif conf['DITHER_MODE'] == 'color_reduce':
            rgb_img = img.convert('RGB')
            dithered = rgb_img.quantize(colors=conf['DITHER_COLORS'], dither=dither_map[conf['DITHER_METHOD']])
            dithered = dithered.convert('RGB')
        
        elif conf['DITHER_MODE'] == 'custom_palette':
            palette_colors = []
            for hex_color in conf['CUSTOM_PALETTE']:
                hex_color = hex_color.lstrip('#')
                r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                palette_colors.extend([r, g, b])
            
            while len(palette_colors) < 768:
                palette_colors.extend([0, 0, 0])
            
            palette_img = Image.new('P', (1, 1))
            palette_img.putpalette(palette_colors)
            
            rgb_img = img.convert('RGB')
            dithered = rgb_img.quantize(palette=palette_img, dither=dither_map[conf['DITHER_METHOD']])
            dithered = dithered.convert('RGB')
        
        if has_alpha:
            dithered = dithered.convert('RGBA')
            dithered.putalpha(alpha_channel)
        
        img = dithered
    return img


PIPELINE_FNS = {
    'sharpen': _apply_sharpen,
    'blur': _apply_blur,
    'contrast': _apply_contrast,
    'exposure': _apply_exposure,
    'gamma': _apply_gamma,
    'color_to_transparent': _apply_color_to_transparent,
    'alpha_outline': _apply_alpha_outline,
    'dither': _apply_dither,
    'invert': _apply_invert,
    'pixelate': _apply_pixelate,
    'fill': _apply_fill,
    'colorize': _apply_colorize,
    'color_replace': _apply_color_replace,
    'rectangle': _apply_rectangle,
    'line': _apply_line,
    'edge_outline': _apply_edge_outline,
    'grain': _apply_grain,
}

DEFAULT_PIPELINE_ORDER = ['sharpen', 'blur', 'contrast', 'exposure', 'gamma', 'color_to_transparent', 'alpha_outline', 'dither', 'invert', 'pixelate', 'fill', 'colorize', 'color_replace', 'rectangle', 'line', 'edge_outline', 'grain']


def apply_filter(img, conf):
    order = conf.get('PIPELINE_ORDER', DEFAULT_PIPELINE_ORDER)
    for step in order:
        if isinstance(step, dict):
            step_name = step.get('step', '')
            enabled = step.get('enabled', True)
            if not enabled:
                continue
            fn = PIPELINE_FNS.get(step_name)
            if fn:
                merged = dict(conf)
                merged.update(step.get('params', {}))
                if step_name == 'dither':
                    merged['DITHERING'] = True
                elif step_name in ('sharpen','blur','contrast','exposure','gamma','alpha_outline','invert','pixelate','fill','colorize','edge_outline','grain','color_to_transparent','color_replace','rectangle','line'):
                    bool_map = {'sharpen':'SHARPEN','blur':'GAUSSIAN_BLUR','contrast':'CONTRAST','exposure':'EXPOSURE','gamma':'GAMMA','alpha_outline':'ALPHA_OUTLINE','invert':'INVERT','pixelate':'PIXELATE','fill':'FILL','colorize':'COLORIZE','edge_outline':'EDGE_OUTLINE','grain':'GRAIN','color_to_transparent':'COLOR_TO_TRANSPARENT','color_replace':'COLOR_REPLACE','rectangle':'RECTANGLE','line':'LINE'}
                    bk = bool_map.get(step_name)
                    if bk:
                        merged[bk] = True
                img = fn(img, merged)
        else:
            fn = PIPELINE_FNS.get(step)
            if fn:
                img = fn(img, conf)
    return img


def resize_image(img, target_size, conf):
    force_square = conf.get('SQUARE_IMAGES', True)
    resize_key = conf.get('RESIZE_METHOD', 'LANCZOS')
    resize_method = RESIZE_METHODS.get(resize_key.upper(), Image.Resampling.LANCZOS)
    
    if force_square:
        if img.size != (target_size, target_size):
            img = img.resize((target_size, target_size), resize_method)
        return img

    w, h = img.size
    longest = max(w, h)
    if longest != target_size:
        scale = target_size / longest
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), resize_method)
    return img
