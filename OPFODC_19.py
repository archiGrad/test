from pathlib import Path
import json
import ast
import subprocess
import shutil
from PIL import Image, ImageDraw, ImageFont
from natsort import natsorted
import numpy as np


# ==========================================
# GLOBAL DEFAULT CONFIGURATION
# ==========================================

DEFAULTS = {
    'SPRITE_SIZE': 256,
    'RESIZE_METHOD': 'LANCZOS',
    
    # Image Processing Defaults
    'SHARPEN': True,
    'SHARPEN_RADIUS': 1,
    'SHARPEN_PERCENT': 20,
    'SHARPEN_THRESHOLD': 3,

    'GAUSSIAN_BLUR': False,
    'GAUSSIAN_BLUR_RADIUS': 2,
    
    'COLOR_TO_TRANSPARENT': [('blue', 0),('green', 0,)],

    'DITHERING': False,
    'DITHER_MODE': 'custom_palette',
    'DITHER_METHOD': 'ordered',
    'DITHER_COLORS': 256,
    'CUSTOM_PALETTE': ['#000000', '#FF0000', '#00FF00'],

    'PIPELINE_ORDER': ['sharpen', 'blur', 'color_to_transparent', 'dither'],

    'MAX_GIF_FRAMES': 30,
    'GIF_SPEED': 0.6, 


    # Folder Label Defaults (Renamed & Updated)
    'IMAGE_SETTINGS_LABELS': True,       
    'IMAGE_SETTINGS_COLOR': 'white',    
    'IMAGE_SETTINGS_FONTSIZE': 3,      
    'IMAGE_SETTINGS_CUSTOMTEXT': "",  
    'LABEL_SPRITE_SIZE': 512,       

    #square image processing or not
    'SQUARE_IMAGES': True,
}

RESIZE_METHODS = {
    'NEAREST': Image.Resampling.NEAREST,
    'BOX': Image.Resampling.BOX,
    'BILINEAR': Image.Resampling.BILINEAR,
    'HAMMING': Image.Resampling.HAMMING,
    'BICUBIC': Image.Resampling.BICUBIC,
    'LANCZOS': Image.Resampling.LANCZOS,
}

# ==========================================
# FIXED CONSTANTS (not overridable via .custom_processing)
# ==========================================
TEXT_IMAGE_RESOLUTION = 256
KTX2_ENCODE = 'etc1s'   # 'uastc' (high quality, larger files) or 'etc1s' (smaller files, lower quality). Default: 'uastc'
KTX2_QUALITY = 100        # uastc: 0-4 (0=fastest, 4=best). etc1s: 1-255 (higher=better). Default: 2
KTX2_ZCMP = 20           # Zstandard supercompression level 1-20 (higher=smaller but slower). Ignored for etc1s. Default: 5
SPRITESHEET_SIZE = 1024 * 4
LOD_TARGET_SPRITE_SIZES = [None, 32, 8]
LOD_VIEWHEIGHT_THRESHOLDS = [20, 40]
STACK_DIM_OPACITY = 0.2
SEED = -1000000000000
ORDERED_GRID_LAYOUT = True
ROTATION_SPEED = 0.000015
RANDOM_ZOOM = False
RANDOM_TEXTDIV_POSITION = False
DIV_RATIO_HALF = False
MAX_LABELS = 200
SCREEN_BUFFER = 100
COLORED_HTML_DOTS = False
BREADCRUMB_HTML_DOTS = True
SHOW_NAV_ACCESSORIES = False
WIRE_STRAIGHT = False
STACK_SPACING = 0.09
STACK_REVERSE = False
ZOOM_VALUE = 0.4
SPRITE_BG = True
SPRITE_BG_OPACITY = 1.0

# ==========================================
# PROCESSING FLAGS
# ==========================================

PROCESS_IMG_LOWRES = False  # Overrides all sprite sizes to DEFAULTS values; dotfiles cannot override

if PROCESS_IMG_LOWRES:
    DEFAULTS['SPRITE_SIZE'] = 8
    DEFAULTS['LABEL_SPRITE_SIZE'] = 8

# ==========================================
# CONFIGURATION PARSING
# ==========================================

def parse_custom_processing(path, current_config):
    new_config = current_config.copy()
    
    custom_file = path / '.custom_processing'
    if custom_file.exists():
        try:
            content = custom_file.read_text()
            for line in content.splitlines():
                line = line.strip()
                if not line or line.startswith('#'): continue
                if '=' in line:
                    key, value = [x.strip() for x in line.split('=', 1)]
                    if key in DEFAULTS:
                        try:
                            new_config[key] = ast.literal_eval(value)
                        except:
                            new_config[key] = value
        except Exception as e:
            print(f"Warning: Error parsing .custom_processing in {path}: {e}")
            
    if PROCESS_IMG_LOWRES:
        new_config['SPRITE_SIZE'] = DEFAULTS['SPRITE_SIZE']
        new_config['LABEL_SPRITE_SIZE'] = DEFAULTS['LABEL_SPRITE_SIZE']

    return new_config


# ==========================================
# IMAGE PROCESSING FUNCTIONS
# ==========================================

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
    if conf['COLOR_TO_TRANSPARENT']:
        color_map = {
            'black': (0, 0, 0), 'white': (255, 255, 255), 'red': (255, 0, 0),
            'green': (0, 255, 0), 'blue': (0, 0, 255), 'yellow': (255, 255, 0),
            'cyan': (0, 255, 255), 'magenta': (255, 0, 255), 'light_gray': (192, 192, 192),
            'dark_gray': (64, 64, 64), 'orange': (255, 165, 0), 'purple': (128, 0, 128)
        }
        targets = [(color_map[c], t) for c, t in conf['COLOR_TO_TRANSPARENT'] if c in color_map]
        if targets:
            img = img.convert('RGBA')
            pixels = img.load()
            for y in range(img.height):
                for x in range(img.width):
                    r, g, b, a = pixels[x, y]
                    for (tr, tg, tb), thresh in targets:
                        if abs(r - tr) < thresh and abs(g - tg) < thresh and abs(b - tb) < thresh:
                            pixels[x, y] = (r, g, b, 0)
                            break
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
    'color_to_transparent': _apply_color_to_transparent,
    'dither': _apply_dither,
}

DEFAULT_PIPELINE_ORDER = ['sharpen', 'blur', 'color_to_transparent', 'dither']


def apply_filter(img, conf):
    order = conf.get('PIPELINE_ORDER', DEFAULT_PIPELINE_ORDER)
    for step in order:
        fn = PIPELINE_FNS.get(step)
        if fn:
            img = fn(img, conf)
    return img

def resize_image(img, target_size, conf=None):
    force_square = conf.get('SQUARE_IMAGES', False) if conf else DEFAULTS['SQUARE_IMAGES']
    resize_key = conf.get('RESIZE_METHOD', DEFAULTS['RESIZE_METHOD']) if conf else DEFAULTS['RESIZE_METHOD']
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

# ==========================================
# FILE SCANNING
# ==========================================



def scan_folder(path, parent_hidden=False, inherited_spacing=None, inherited_zoom=None, inherited_keywords=None, inherited_reverse=None, inherited_sprite_bg=None, inherited_sprite_bg_opacity=None, ignore=['venv', '__pycache__', '.git', 'fonts',  'spritesheets', 'images', 'backup', 'geo']):
    if path.name in ignore:
        return None
    
    images = []
    texts = []
    children = []
    grid_layout = None
    no_accum = False
    stop_accum = False
    
    manual_spacing = inherited_spacing
    manual_zoom = inherited_zoom
    zoom_propagate = False
    manual_reverse = inherited_reverse
    manual_sprite_bg = inherited_sprite_bg
    manual_sprite_bg_opacity = inherited_sprite_bg_opacity
    keywords = list(inherited_keywords) if inherited_keywords else []
    
    is_hidden = parent_hidden or (path / '.hidden').exists()
    
    if path.is_dir():
        kw_file = path / '.keywords'
        if kw_file.exists():
            try:
                lines = [l.strip() for l in kw_file.read_text().splitlines() if l.strip() and not l.strip().startswith('#')]
                keywords.extend(lines)
            except Exception as e:
                print(f"Warning: Error reading .keywords in {path}: {e}")

        grid_file = path / '.grid_layout'
        if grid_file.exists():
            grid_layout = grid_file.read_text().strip()
        
        # Check .custom_processing for STACK_SPACING
        custom_file = path / '.custom_processing'
        if custom_file.exists():
            try:
                content = custom_file.read_text()
                for line in content.splitlines():
                    line = line.strip()
                    if not line or line.startswith('#'): continue
                    if '=' in line:
                        key, value = [x.strip() for x in line.split('=', 1)]
                        if key == 'STACK_SPACING':
                            try:
                                manual_spacing = ast.literal_eval(value)
                            except:
                                pass
                        if key == 'STACK_REVERSE':
                            try:
                                manual_reverse = ast.literal_eval(value)
                            except:
                                pass
                        if key == 'ZOOM_VALUE':
                            try:
                                manual_zoom = float(ast.literal_eval(value))
                            except:
                                pass
                        if key == 'ZOOM_PROPAGATE':
                            try:
                                zoom_propagate = ast.literal_eval(value)
                            except:
                                pass
                        if key == 'SPRITE_BG':
                            try:
                                manual_sprite_bg = ast.literal_eval(value)
                            except:
                                pass
                        if key == 'SPRITE_BG_OPACITY':
                            try:
                                manual_sprite_bg_opacity = float(ast.literal_eval(value))
                            except:
                                pass
            except:
                pass
        
        # .stack_spacing file overrides .custom_processing
        # spacing_file = path / '.stack_spacing'
        # if spacing_file.exists():
        #     try:
        #         manual_spacing = float(spacing_file.read_text().strip())
        #     except Exception as e:
        #         print(f"Warning: Invalid .stack_spacing in {path}: {e}")
        
        no_accum = (path / '.html_only_here').exists()
        stop_accum = (path / '.html_stops_here').exists()
            
        for item in natsorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name)):
            if item.name in ignore or item.name.startswith('.'): 
                continue
            if item.is_file():
                if item.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
                    images.append(item.relative_to('.').as_posix())
                elif item.suffix.lower() == '.html':
                    texts.append(item.relative_to('.').as_posix())
            elif item.is_dir():
                child = scan_folder(item, parent_hidden=is_hidden, inherited_spacing=manual_spacing, inherited_zoom=manual_zoom if zoom_propagate else None, inherited_keywords=keywords, inherited_reverse=manual_reverse, inherited_sprite_bg=manual_sprite_bg, inherited_sprite_bg_opacity=manual_sprite_bg_opacity, ignore=ignore)
                if child:
                    children.append(child)
        
        all_images = images.copy()
        all_texts = texts.copy()
        if not stop_accum:
            for child in children:
                all_images.extend(child['ai'])
                if not child.get('na', False) and not child.get('sa', False):
                    all_texts.extend(child['at'])
        else:
            for child in children:
                all_images.extend(child['ai'])        

    content_type = 'empty'
    if all_images and all_texts: content_type = 'mixed'
    elif all_images: content_type = 'images'
    elif all_texts: content_type = 'text'
    
    result = {
        'name': path.name,
        'path': path.relative_to('.').as_posix(),
        'type': content_type,
        'children': children,
        'ai': all_images, 
        'at': all_texts,
        'oi': images,
        'ot': texts,
        'na': no_accum,
        'sa': stop_accum,
        'hid': is_hidden,
        'msp': manual_spacing,
        'mrv': manual_reverse,
        'mzm': manual_zoom,
        'sbg': manual_sprite_bg,
        'sbo': manual_sprite_bg_opacity,
        'sk': keywords if keywords else []
    }
    if grid_layout: result['grid_layout'] = grid_layout
    return result


# root = scan_folder(Path('.'))
# root['name'] = Path('.').resolve().name or "Root"


# Check if the specific project folder exists and use it as the root
target_root = Path('archiGrad.io')

if target_root.exists() and target_root.is_dir():
    root = scan_folder(target_root, inherited_spacing=None)


# ==========================================
# LABEL GENERATION
# ==========================================

def load_custom_font(size):
    # 1. Check specifically for the requested font in ./fonts/
    requested_font = Path('fonts/UbuntuMono-Regular.ttf')
    if requested_font.exists():
        try:
            return ImageFont.truetype(str(requested_font), int(size))
        except IOError:
            print(f"Warning: Found {requested_font} but could not load it.")
            
    # 2. Fallback List
    font_candidates = [
        "UbuntuMono-Regular.ttf", 
        "UbuntuMono-R.ttf",
        "DejaVuSansMono.ttf", 
        "FreeMono.ttf", 
        "Consolas.ttf",
        "arial.ttf"
    ]
    
    for font_name in font_candidates:
        try:
            return ImageFont.truetype(font_name, int(size))
        except IOError:
            continue
            
    # 3. Final Fallback
    try:
        return ImageFont.load_default(size=int(size))
    except TypeError:
        return ImageFont.load_default()

def get_dotfile_content(path):
    ignored_dotfiles = [
        '.html_only_here', '.html_stops_here', '.hidden', 
        '.DS_Store', '.git', '.gitignore', '__pycache__'
    ]
    
    candidates = []
    try:
        for item in path.iterdir():
            if item.is_file() and item.name.startswith('.') and item.name not in ignored_dotfiles:
                candidates.append(item)
    except Exception:
        pass
    
    if not candidates:
        return None
        
    target = sorted(candidates, key=lambda x: x.name)[0]
    
    try:
        content = target.read_text(encoding='utf-8')
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        
        if not lines:
            return None
            
        return {'name': target.name, 'lines': lines}
    except Exception as e:
        print(f"Error reading dotfile {target}: {e}")
        return None

def create_label_image(title, body_lines, color, output_path, img_resolution, font_size):
    try:
        # Use the passed resolution
        img = Image.new('RGBA', (img_resolution, img_resolution), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Use the scaled font size
        font_title = load_custom_font(int(font_size * 1.5))
        font_body = load_custom_font(int(font_size))
        
        def get_text_size(text, font):
            try:
                left, top, right, bottom = font.getbbox(text)
                return right - left, bottom - top
            except AttributeError:
                return font.getsize(text)

        # Layout Configuration
        scale_factor = img_resolution / 64.0
        padding_left = int(10 * scale_factor)
        padding_top = int(10 * scale_factor)
        
        # --- CHANGED: Set spacing to minimum (0) ---
        line_spacing = 0 
        title_body_gap = int(2 * scale_factor) if (title and body_lines) else 0
        # -------------------------------------------
        
        current_y = padding_top
        
        if title:
            draw.text((padding_left, current_y), title, fill=color, font=font_title)
            title_w, title_h = get_text_size(title, font_title)
            current_y += title_h + title_body_gap
        
        for line in body_lines:
            draw.text((padding_left, current_y), line, fill=color, font=font_body)
            line_w, line_h = get_text_size(line, font_body)
            current_y += line_h + line_spacing
            
        img.save(output_path)
        return True
    except Exception as e:
        print(f"Error creating label {output_path}: {e}")
        return False

def is_label_image(filename):
    return (filename.startswith('ZZ_') and filename.endswith('_top.png')) or \
           (filename.startswith('AA_') and filename.endswith('_bottom.png'))

def manage_folder_labels(node, parent_config):
    node_path = Path(node['path'])
    current_config = parse_custom_processing(node_path, parent_config)
    
    folder_name = node['name']
    top_name = f"ZZ_{folder_name}_top.png"
    bottom_name = f"AA_{folder_name}_bottom.png"
    
    for f in node_path.iterdir():
        if f.is_file() and is_label_image(f.name) and f.name not in [top_name, bottom_name]:
            try: f.unlink()
            except: pass
    
    top_path = node_path / top_name
    bottom_path = node_path / bottom_name
    
    real_images = []
    for img_str in node['oi']:
        if not is_label_image(Path(img_str).name):
            real_images.append(img_str)
    
    real_images = natsorted(real_images, key=lambda x: Path(x).name)
            
    should_generate = current_config.get('IMAGE_SETTINGS_LABELS', False)
    has_real_images = len(real_images) > 0
    
    new_oi_list = []

    if should_generate and has_real_images:
        color = current_config.get('IMAGE_SETTINGS_COLOR', 'white')
        
        # Configs from user/dotfile
        base_font_size = current_config.get('IMAGE_SETTINGS_FONTSIZE', 16)
        custom_text = current_config.get('IMAGE_SETTINGS_CUSTOMTEXT', "")
        
        # FORCE High Resolution
        text_resolution = TEXT_IMAGE_RESOLUTION
        
        # SCALE Font Size
        font_scale = text_resolution / 64.0
        effective_font_size = int(base_font_size * font_scale)
        
        # -------------------------------------------------
        # PRIORITY LOGIC
        # -------------------------------------------------
        if custom_text and custom_text.strip():
            lines = custom_text.strip().splitlines()
            top_title = ""
            top_body = lines
            bottom_title = ""
            bottom_body = lines
        else:
            dotfile_data = get_dotfile_content(node_path)
            if dotfile_data:
                top_title = dotfile_data['name']
                top_body = dotfile_data['lines']
                bottom_title = dotfile_data['name']
                bottom_body = dotfile_data['lines']
            else:
                top_title = folder_name
                top_body = ["top"]
                bottom_title = folder_name
                bottom_body = ["bottom"]
        
        node_reverse = node.get('mrv') or False
        first_label = (top_title, top_body, top_path) if node_reverse else (bottom_title, bottom_body, bottom_path)
        last_label = (bottom_title, bottom_body, bottom_path) if node_reverse else (top_title, top_body, top_path)

        if create_label_image(first_label[0], first_label[1], color, first_label[2], text_resolution, effective_font_size):
            new_oi_list.append(first_label[2].relative_to('.').as_posix())
        
        new_oi_list.extend(real_images)
        
        if create_label_image(last_label[0], last_label[1], color, last_label[2], text_resolution, effective_font_size):
            new_oi_list.append(last_label[2].relative_to('.').as_posix())
            
        node['oi'] = new_oi_list
        
    else:
        node['oi'] = real_images

    for child in node['children']:
        manage_folder_labels(child, current_config)

# Execute Label Generation
manage_folder_labels(root, DEFAULTS)


# ==========================================
# FLATTENING AND CONFIG ASSIGNMENT
# ==========================================

all_image_items = [] 

def collect_images_with_config(node, parent_config):
    node_path = Path(node['path'])
    current_config = parse_custom_processing(node_path, parent_config)
    
    label_conf = None
    for img_path in node['oi']:
        if is_label_image(Path(img_path).name):
            if not label_conf:
                label_conf = current_config.copy()
                label_conf['SPRITE_SIZE'] = current_config.get('LABEL_SPRITE_SIZE', DEFAULTS['LABEL_SPRITE_SIZE'])
            all_image_items.append({'path': img_path, 'conf': label_conf})
        else:
            all_image_items.append({'path': img_path, 'conf': current_config})
        
    for child in node['children']:
        collect_images_with_config(child, current_config)

collect_images_with_config(root, DEFAULTS)

all_image_items.sort(key=lambda x: (x['conf']['SPRITE_SIZE'], x['path']))

# ==========================================
# SPRITESHEET GENERATION
# ==========================================

Path('spritesheets').mkdir(exist_ok=True)
for file in Path('spritesheets').glob('*'):
    file.unlink()

LOD_LEVELS = 3

sprite_data = {} 
sheet_idx = 0
current_sheet_size = 0 
sheet = None
slot_idx = 0
saved_sheets = []
empty_slot = {}
img_dedup = {}


# def _is_fully_transparent(img):
#     return img.convert('RGBA').getextrema()[3][1] == 0

def _is_fully_transparent(img, threshold=0.005):
    a = np.array(img.convert('RGBA'))[:, :, 3]
    return np.count_nonzero(a) / a.size < threshold


def _get_empty_slot(sheet_idx):
    global slot_idx
    if sheet_idx not in empty_slot:
        empty_slot[sheet_idx] = slot_idx
        slot_idx += 1
    return empty_slot[sheet_idx]

def _convert_to_ktx2(png_path):
    ktx2_path = png_path.replace('.png', '.ktx2')
    cmd = ['toktx', '--t2', '--encode', KTX2_ENCODE]
    if KTX2_ENCODE == 'uastc':
        cmd += ['--uastc_quality', str(KTX2_QUALITY)]
    else:
        cmd += ['--qlevel', str(KTX2_QUALITY)]
    if KTX2_ENCODE != 'etc1s':
        cmd += ['--zcmp', str(KTX2_ZCMP)]
    cmd += ['--lower_left_maps_to_s0t0', ktx2_path, png_path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"toktx error for {png_path}:\n  stdout: {result.stdout}\n  stderr: {result.stderr}")
            raise RuntimeError(f"toktx failed with exit code {result.returncode}")
    except FileNotFoundError:
        print("ERROR: 'toktx' not found. Install KTX-Software: https://github.com/KhronosGroup/KTX-Software/releases")
        raise
    return ktx2_path

def save_sheet_with_lod(sheet, sheet_idx, sprite_size, conf=None):
    resize_key = conf.get('RESIZE_METHOD', DEFAULTS['RESIZE_METHOD']) if conf else DEFAULTS['RESIZE_METHOD']
    resize_method = RESIZE_METHODS.get(resize_key.upper(), Image.Resampling.LANCZOS)
    for lod in range(LOD_LEVELS):
        if lod == 0:
            lod_sheet = sheet
        else:
            target_sprite = LOD_TARGET_SPRITE_SIZES[lod]
            scale_factor = 1.0 if target_sprite >= sprite_size else target_sprite / sprite_size
            new_size = max(1, int(sheet.width * scale_factor))
            lod_sheet = sheet.resize((new_size, new_size), resize_method)
        png_path = f'spritesheets/sprites_{sheet_idx}_lod{lod}.png'
        lod_sheet.save(png_path)
        _convert_to_ktx2(png_path)
        webp_path = png_path.replace('.png', '.webp')
        lod_sheet.save(webp_path, 'WEBP', quality=80)
    saved_sheets.append(sheet_idx)
    lod1_size = max(1, int(sheet.width * min(1.0, LOD_TARGET_SPRITE_SIZES[1] / sprite_size)))
    lod2_size = max(1, int(sheet.width * min(1.0, LOD_TARGET_SPRITE_SIZES[2] / sprite_size)))
    print(f"  Saved spritesheet {sheet_idx} (sprite_size={sprite_size}) with {LOD_LEVELS} LOD levels ({sheet.width}px -> {lod1_size}px -> {lod2_size}px)")

def lod_paths(sheet_idx):
    return [f'spritesheets/sprites_{sheet_idx}_lod{lod}.ktx2' for lod in range(LOD_LEVELS)]

total_items = len(all_image_items)
for i, item in enumerate(all_image_items):
    pct = (i + 1) / total_items
    bar = ('█' * int(pct * 20)).ljust(20)
    #print(f"\r  [{bar}] {i+1}/{total_items} {Path(item['path']).name[:40]:<40}", end='', flush=True)
    img_path = item['path']
    conf = item['conf']
    
    target_size = conf['SPRITE_SIZE']
    sheet_dim = SPRITESHEET_SIZE
    
    sprites_per_row = sheet_dim // target_size
    sprites_per_sheet = sprites_per_row * sprites_per_row
    
    if sheet is None or target_size != current_sheet_size or slot_idx >= sprites_per_sheet:
        if sheet is not None:
            save_sheet_with_lod(sheet, sheet_idx, current_sheet_size, conf)
            sheet_idx += 1
        
        sheet = Image.new('RGBA', (sheet_dim, sheet_dim), (0, 0, 0, 0))
        slot_idx = 0
        current_sheet_size = target_size

    speed_val = max(0.0, min(1.0, conf['GIF_SPEED']))
    # gif_delay_ms = int(500 - (speed_val * 480))
    # gif_delay_ms = int(2000 * (1 - speed_val)**3 + 10)
    gif_delay_ms = int(10000 * (1 - speed_val)**3 + 10)



    is_gif = img_path.lower().endswith('.gif')
    
    if is_gif:
        try:
            gif = Image.open(img_path)
            frame_count = min(gif.n_frames, conf['MAX_GIF_FRAMES']) 

            frames = []
            for frame_idx in range(frame_count):
                gif.seek(frame_idx)
                frame = gif.convert('RGBA')
                frame = resize_image(frame, target_size, conf)
                frame = apply_filter(frame, conf)
                frames.append(frame)

            new_hashes = [hash(f.tobytes()) for f in frames if not _is_fully_transparent(f)]
            unique_new = sum(1 for h in new_hashes if (sheet_idx, h) not in img_dedup)
            slots_needed = unique_new + (1 if len(new_hashes) < frame_count and sheet_idx not in empty_slot else 0)

            if slot_idx + slots_needed > sprites_per_sheet:
                save_sheet_with_lod(sheet, sheet_idx, current_sheet_size, conf)
                sheet_idx += 1
                sheet = Image.new('RGBA', (sheet_dim, sheet_dim), (0, 0, 0, 0))
                slot_idx = 0

            fm = []
            for frame in frames:
                if _is_fully_transparent(frame):
                    fm.append(_get_empty_slot(sheet_idx))
                else:
                    h = hash(frame.tobytes())
                    key = (sheet_idx, h)
                    if key in img_dedup:
                        fm.append(img_dedup[key])
                    else:
                        col = slot_idx % sprites_per_row
                        row = slot_idx // sprites_per_row
                        sheet.paste(frame, (col * target_size, row * target_size))
                        img_dedup[key] = slot_idx
                        fm.append(slot_idx)
                        slot_idx += 1
                        
            sprite_data[img_path] = {
                'ss': lod_paths(sheet_idx),
                'fm': fm,
                'anim': True,
                'w': target_size,
                'h': target_size,
                'sz': target_size,
                'gd': gif_delay_ms,
                'path': img_path
            }
        except Exception as e:
            print(f"Error processing GIF {img_path}: {e}")
    else:
        try:
            img = Image.open(img_path).convert('RGBA')
            img = resize_image(img, target_size, conf)
            img = apply_filter(img, conf)       

            if _is_fully_transparent(img):
                idx = _get_empty_slot(sheet_idx)
            else:
                h = hash(img.tobytes())
                key = (sheet_idx, h)
            if key in img_dedup:
                idx = img_dedup[key]
            else:
                col = slot_idx % sprites_per_row
                row = slot_idx // sprites_per_row
                sheet.paste(img, (col * target_size, row * target_size))
                idx = slot_idx
                img_dedup[key] = idx
                slot_idx += 1
            
            sprite_data[img_path] = {
                'ss': lod_paths(sheet_idx),
                'idx': idx,
                'w': img.width,
                'h': img.height,
                'sz': target_size,
                'path': img_path
            }
        except Exception as e:
            print(f"Error processing Image {img_path}: {e}")

print()
if sheet:
    save_sheet_with_lod(sheet, sheet_idx, current_sheet_size, conf)


#see how many duplicates we had and are now referenced in a single spritesheet slot
total_static = sum(1 for item in all_image_items if not item['path'].lower().endswith('.gif'))
total_gif_frames = sum(len(sprite_data[item['path']].get('fm', [])) for item in all_image_items if item['path'].lower().endswith('.gif') and item['path'] in sprite_data)
total_all = total_static + total_gif_frames
unique_slots = len(img_dedup)
print(f"  Dedup: {total_all - unique_slots} frames reused existing slots ({unique_slots} unique out of {total_all})")


# ==========================================
# TREE PROCESSING & SAVING (OPTIMIZED)
# ==========================================

# 1. Assign 'gi' (Global Index) to the original sprite objects 
#    We do this BEFORE converting the tree to IDs, while we still have paths.
global_index_counter = 0

def assign_gi_and_filter(node):
    global global_index_counter
    # We filter and assign 'gi' at the same time
    
    # Process 'ai' (Active Images)
    valid_ai = []
    for path in node['ai']:
        if path in sprite_data:
            sprite_data[path]['gi'] = global_index_counter
            global_index_counter += 1
            valid_ai.append(path)
    node['ai'] = valid_ai
            
    # Process 'oi' (Other Images)
    valid_oi = []
    for path in node['oi']:
        if path in sprite_data:
            sprite_data[path]['gi'] = global_index_counter
            global_index_counter += 1
            valid_oi.append(path)
    node['oi'] = valid_oi

    for child in node['children']:
        assign_gi_and_filter(child)

assign_gi_and_filter(root)

# 2. Create the flat "Database" of all sprite objects
image_database_list = list(sprite_data.values())

# 3. Create a lookup map: Image Path -> ID (Integer)
path_to_id = { item['path']: i for i, item in enumerate(image_database_list) }

# 4. Inject IDs into the tree instead of full objects (NORMALIZATION)
def replace_images_with_ids(node):
    # 'ai' and 'oi' will now store Integers (0, 1, 2) instead of full Objects
    node['ai'] = [path_to_id[p] for p in node['ai'] if p in path_to_id]
    node['oi'] = [path_to_id[p] for p in node['oi'] if p in path_to_id]
    
    for child in node['children']:
        replace_images_with_ids(child)

replace_images_with_ids(root)

sprite_config = {
    'spritesheet_size': SPRITESHEET_SIZE,
    'stack_spacing': STACK_SPACING,
    'stack_reverse': STACK_REVERSE,
    'stack_dim_opacity': STACK_DIM_OPACITY,
    'seed': SEED,
    'ordered_grid_layout': ORDERED_GRID_LAYOUT,
    'rotation_speed': ROTATION_SPEED,
    'random_textdiv_position': RANDOM_TEXTDIV_POSITION,
    'div_ratio_half': DIV_RATIO_HALF,
    'random_zoom': RANDOM_ZOOM,
    'zoom_value': ZOOM_VALUE,
    'lod_viewheight_thresholds': LOD_VIEWHEIGHT_THRESHOLDS,
    'max_labels': MAX_LABELS,
    'screen_buffer': SCREEN_BUFFER,
    'colored_html_dots': COLORED_HTML_DOTS,
    'breadcrumb_html_dots': BREADCRUMB_HTML_DOTS,
    'show_nav_accessories': SHOW_NAV_ACCESSORIES,

    'wire_straight': WIRE_STRAIGHT,

    'sprite_bg': SPRITE_BG,
    'sprite_bg_opacity': SPRITE_BG_OPACITY,
    'pipeline_order': DEFAULTS.get('PIPELINE_ORDER', DEFAULT_PIPELINE_ORDER),
}

# 5. Save BOTH the tree (lightweight) and the database (heavy)
# with open('data.json', 'w') as f:
with open('data.json', 'w', encoding='utf-8') as f:
    json.dump({
        'tree': root, 
        'database': image_database_list, 
        'sprite_config': sprite_config
    }, f, indent=None) # indent=None makes the file significantly smaller

# ==========================================
# HTML GENERATION
# ==========================================


#with open('index.html', 'w') as f:
with open('index.html', 'w', encoding='utf-8') as f:
    f.write(f'''<!DOCTYPE html>
<html>
<head>
<link rel="stylesheet" href="styles.css">
<script src="https://unpkg.com/stats.js@0.17.0/build/stats.min.js"></script>
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<meta charset="utf-8">

            

</head>
<body>
<div id="tree">
    <div id="tree-toolbar" style="display:flex;align-items:center;margin-bottom:6px;">
        <input id="tree-search" type="text" placeholder="search..." />
        <span id="nav-back">&lt;</span>
        <span id="nav-fwd">&gt;</span>
        <span id="tree-expand-all">+</span>
    </div>
    <div id="tree-content"></div>
    <div id="tree-legend">
        <span><span style="color:#4f4;">●</span> text_stop</span>
        <span><span style="color:#f44;">●</span> text_hard_stop</span>
        <span><span style="color:#fff;">●</span> text page</span>
    </div>
</div>
<div id="content"></div>
<script type="importmap">
{{
  "imports": {{
    "three": "https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/"
  }}
}}
</script>

<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';
import {{ mergeGeometries }} from 'three/addons/utils/BufferGeometryUtils.js';
import {{ KTX2Loader }} from 'three/addons/loaders/KTX2Loader.js';
import {{ Line2 }} from 'three/addons/lines/Line2.js';
import {{ LineMaterial }} from 'three/addons/lines/LineMaterial.js';
import {{ LineGeometry }} from 'three/addons/lines/LineGeometry.js';
THREE.ColorManagement.enabled = false;

(() => {{
    const tree = document.getElementById('tree');
    const treeContent = document.getElementById('tree-content');
    const searchInput = document.getElementById('tree-search');
    let dragging = false, ox = 0, oy = 0, tabIndex = -1, visibleMatches = [];

    const navBack = document.getElementById('nav-back');
    const navFwd = document.getElementById('nav-fwd');

    navBack.onclick = (e) => {{ e.stopPropagation(); navigateHistory(-1); }};
    navFwd.onclick = (e) => {{ e.stopPropagation(); navigateHistory(1); }};

    const expandAllBtn = document.getElementById('tree-expand-all');
    let treeExpanded = false;
    expandAllBtn.onclick = (e) => {{
        e.stopPropagation();
        treeExpanded = !treeExpanded;
        expandAllBtn.textContent = treeExpanded ? '-' : '+';
        document.querySelectorAll('.tree-link').forEach(link => {{
            const cc = link._childContainer;
            if (cc) cc.style.display = treeExpanded ? 'block' : 'none';
        }});
        document.querySelectorAll('.tree-caret').forEach(caret => {{
            const link = caret.closest('.tree-link');
            const cc = link?._childContainer;
            if (cc) caret.textContent = treeExpanded ? 'v' : '>';
        }});
    }};

    tree.style.left = Math.floor(Math.random() * Math.max(0, innerWidth - 300)) + 'px';
    tree.style.top = Math.floor(Math.random() * Math.max(0, innerHeight - 400)) + 'px';

    tree.addEventListener('mousedown', (e) => {{
        if (e.target.closest('.tree-link, #tree-search, #nav-back, #nav-fwd, #tree-expand-all')) return;
        dragging = true; ox = e.clientX - tree.offsetLeft; oy = e.clientY - tree.offsetTop;
        tree.style.cursor = 'grabbing'; document.body.style.userSelect = 'none';
    }});
    document.addEventListener('mousemove', (e) => {{ if (dragging) {{ tree.style.left = (e.clientX - ox) + 'px'; tree.style.top = (e.clientY - oy) + 'px'; }} }});
    document.addEventListener('mouseup', () => {{ if (dragging) {{ dragging = false; tree.style.cursor = 'grab'; document.body.style.userSelect = ''; }} }});

    function clearSearch() {{
        visibleMatches.forEach(l => l.style.background = '');
        activeScenes.forEach(s => {{ if (s.clearWireNav) s.clearWireNav(); }});
        searchInput.value = ''; tabIndex = -1;
        treeContent.querySelectorAll('.tree-item').forEach(el => el.style.display = '');
        treeContent.querySelectorAll('.tree-children').forEach(el => el.style.display = 'none');
        updateTreeState(true);
    }}

    function highlightMatch(idx) {{
        visibleMatches.forEach(l => l.style.background = '');
        if (idx >= 0 && idx < visibleMatches.length) {{
            visibleMatches[idx].style.background = '#333';
            const p = visibleMatches[idx].dataset.path;
            if (p) activeScenes.forEach(s => {{ if (s.driveWireNav) s.driveWireNav(p); }});
            const item = visibleMatches[idx].closest('.tree-item');
            if (item) {{
                const top = item.offsetTop, bot = top + item.offsetHeight;
                if (bot > treeContent.scrollTop + treeContent.clientHeight) treeContent.scrollTop = bot - treeContent.clientHeight;
                else if (top < treeContent.scrollTop) treeContent.scrollTop = top;
            }}
        }} else {{
            activeScenes.forEach(s => {{ if (s.clearWireNav) s.clearWireNav(); }});
        }}
    }}

    function collectVisible() {{
        visibleMatches = Array.from(treeContent.querySelectorAll('.tree-link')).filter(link => {{
            if (link.style.textDecoration === 'line-through') return false;
            let el = link.closest('.tree-item');
            while (el && el !== treeContent) {{ if (el.style.display === 'none') return false; el = el.parentElement; }}
            return true;
        }});
    }}

    searchInput.addEventListener('input', () => {{
        const q = searchInput.value.toLowerCase().trim();
        tabIndex = -1;
        if (!q) {{ clearSearch(); return; }}
        treeContent.querySelectorAll('.tree-item').forEach(el => el.style.display = 'none');
        treeContent.querySelectorAll('.tree-children').forEach(el => el.style.display = 'none');
        treeContent.querySelectorAll('.tree-link').forEach(link => {{
            if (!link.textContent.toLowerCase().includes(q) && !(link.dataset.keywords && link.dataset.keywords.includes(q))) return;
            const item = link.closest('.tree-item');
            if (!item) return;
            item.style.display = '';
            const sib = item.nextElementSibling;
            if (sib && sib.classList.contains('tree-children')) {{
                sib.querySelectorAll('.tree-item').forEach(el => el.style.display = '');
                sib.querySelectorAll('.tree-children').forEach(el => el.style.display = 'none');
            }}
            let p = item.parentElement;
            while (p && p !== treeContent) {{
                if (p.classList.contains('tree-children')) p.style.display = 'block';
                if (p.classList.contains('tree-item')) p.style.display = '';
                p = p.parentElement;
            }}
        }});
        collectVisible();
        updateTreeState();
    }});

    searchInput.addEventListener('mousedown', (e) => e.stopPropagation());
    searchInput.addEventListener('keydown', (e) => {{
        if (e.key === 'Tab') {{
            e.preventDefault(); collectVisible();
            if (!visibleMatches.length) return;
            tabIndex = e.shiftKey ? (tabIndex <= 0 ? visibleMatches.length - 1 : tabIndex - 1) : (tabIndex + 1) % visibleMatches.length;
            highlightMatch(tabIndex);
        }} else if (e.key === 'Enter') {{
            e.preventDefault();
            const target = (tabIndex >= 0 && tabIndex < visibleMatches.length) ? visibleMatches[tabIndex] : (collectVisible(), visibleMatches[0]);
            const name = target?.querySelector('.tree-name');
            if (name) name.click();
            clearSearch();
        }} else if (e.key === 'Escape') {{ clearSearch(); searchInput.blur(); }}
    }});
    document.addEventListener('keydown', (e) => {{ if (e.key === 'Escape' && document.activeElement !== searchInput) clearSearch(); }});
}})();

const loader = document.createElement('div');
loader.id = 'loader';
loader.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;background:black;color:white;display:flex;align-items:center;justify-content:center;z-index:99999;font-family:monospace;';
loader.textContent = 'initializing...';
document.body.appendChild(loader);

            
const GRID_COLS = window.innerWidth > 768 ? 20 : 1;
const GRID_ROWS = window.innerWidth > 768 ? 1 : 20;
const GRID_TOTAL = GRID_COLS * GRID_ROWS;

const _gridOrder = Array.from({{length: GRID_TOTAL}}, (_, i) => i);
(function shuffle(a) {{
    let s = 9301;
    for (let i = a.length - 1; i > 0; i--) {{
        s = (s * 16807 + 1) & 0x7fffffff;
        const j = s % (i + 1);
        [a[i], a[j]] = [a[j], a[i]];
    }}
}})(_gridOrder);

let progress = {{ss: 0, ssTotal: 0, stacks: 0, stacksTotal: 0, imgs: 0, imgsTotal: 0}};

function makeGrid(percent) {{
    const filled = Math.floor((percent / 100) * GRID_TOTAL);
    const active = new Set(_gridOrder.slice(0, filled));
    let rows = '';
    for (let r = 0; r < GRID_ROWS; r++) {{
        let row = '';
        for (let c = 0; c < GRID_COLS; c++) {{
            row += active.has(r * GRID_COLS + c) ? '\u00B7' : '\u00A0';
        }}
        rows += row + '\\n';
    }}
    return rows;
}}

function updateLoader() {{
    const l = document.getElementById('loader');
    if (!l) return;
    
    const ssPercent = progress.ssTotal > 0 ? Math.floor((progress.ss / progress.ssTotal) * 100) : 0;
    const stacksPercent = progress.stacksTotal > 0 ? Math.floor((progress.stacks / progress.stacksTotal) * 100) : 0;
    const imgsPercent = progress.imgsTotal > 0 ? Math.floor((progress.imgs / progress.imgsTotal) * 100) : 0;
    
    const linkStyle = 'color:white;text-decoration:underline;display:block;text-align:right;line-height:1.8;';

    l.innerHTML = `<div style="font-family:monospace;">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:24px;">
            <div>
                <div style="margin-bottom:4px;">archigrad.io</div>
                <div>office for architecture and computation</div>
            </div>
            <div style="text-align:right;">
                <a href="https://github.com/archiGrad" target="_blank" style="${{linkStyle}}">github</a>
                <a href="#" style="${{linkStyle}}">whatsapp</a>
                <a href="#" style="${{linkStyle}}">line</a>
                <a href="mailto:putteneersjoris@gmail.com" style="${{linkStyle}}">email</a>
            </div>
        </div>
        <div style="display:flex;gap:24px;">
            <div style="flex:1;">
                <div style="margin-bottom:6px;">spritesheets:<br>${{progress.ss}}/${{progress.ssTotal}}</div>
                <pre style="margin:0;font-size:var(--font-loading-grid, 10px);line-height:1.2;letter-spacing:2px;">${{makeGrid(ssPercent)}}</pre>
                <div style="margin-top:4px;">${{ssPercent}}%</div>
            </div>
            <div style="flex:1;">
                <div style="margin-bottom:6px;">stacks:<br>${{progress.stacks}}/${{progress.stacksTotal}}</div>
                <pre style="margin:0;font-size:var(--font-loading-grid, 10px);line-height:1.2;letter-spacing:2px;">${{makeGrid(stacksPercent)}}</pre>
                <div style="margin-top:4px;">${{stacksPercent}}%</div>
            </div>
            <div style="flex:1;">
                <div style="margin-bottom:6px;">images:<br>${{progress.imgs}}/${{progress.imgsTotal}}</div>
                <pre style="margin:0;font-size:var(--font-loading-grid, 10px);line-height:1.2;letter-spacing:2px;">${{makeGrid(imgsPercent)}}</pre>
                <div style="margin-top:4px;">${{imgsPercent}}%</div>
            </div>
        </div>
    </div>`;
}}

async function updateLoaderAsync() {{
    updateLoader();
    await new Promise(r => requestAnimationFrame(r));
}}

let dataTree;
let spriteConfig;
let activeScenes = [];
let currentPage = 0;
let allFilteredChildren = [];
function getPageSize() {{ return window.innerWidth <= 768 ? 6 : 16; }}
const textureCache = {{}};
const pendingLoads = {{}};
const stackMaterialCache = {{}};
const lodAvailable = {{}};
const LOD_COUNT = 3;

function ssBase(ssArray) {{ return ssArray[0].replace('_lod0', ''); }}

function seededRandom(seed) {{
    let state = seed;
    state = (state * 1664525 + 1013904223) % 4294967296;
    return state / 4294967296;
}}

function seededColorFromPath(path) {{
    const hash = path.split('').reduce((a, c) => a + c.charCodeAt(0), 0);
    let state = hash * spriteConfig.seed;
    function next() {{ state = (state * 1664525 + 1013904223) % 4294967296; return state / 4294967296; }}
    return `hsl(${{Math.floor(next() * 360)}},${{60 + Math.floor(next() * 30)}}%,${{50 + Math.floor(next() * 20)}}%)`;
}}

function buildHtmlDots(atArray, sizeClass) {{
    if (!atArray || atArray.length === 0) return '';
    return atArray.map(p => {{
        const c = spriteConfig.colored_html_dots ? seededColorFromPath(p) : '#fff';
        return `<span class="${{sizeClass}}" style="color:${{c}};vertical-align:middle;margin-left:1px;">●</span>`;
    }}).join('');
}}

function dotSpan(color, cls) {{
    return `<span class="${{cls}}" style="color:${{color}};vertical-align:middle;margin-left:1px;">●</span>`;
}}

function navDeco(node, cls, showHtmlDots) {{
    if (!node) return '';
    let d = '';
    if (node.na) d += ' ' + dotSpan('#4f4', cls);
    if (node.sa) d += ' ' + dotSpan('#f44', cls);
    if (showHtmlDots !== false && node.at && node.at.length > 0) d += ' ' + buildHtmlDots(node.at, cls);
    if (node._dc > 0) d += ` <span class="nav-count">(${{node._dc}})</span>`;
    return d;
}}

function breadcrumbHtml(path, activePath) {{
    const parts = path.split('/');
    return parts.map((part, idx) => {{
        const partPath = parts.slice(0, idx + 1).join('/');
        const color = (activePath && partPath === activePath) ? '#44f' : 'white';
        const pNode = findNodeByPath(dataTree, partPath);
        const acc = spriteConfig.show_nav_accessories;
        const deco = acc ? navDeco(pNode, 'dot-md', spriteConfig.breadcrumb_html_dots) : '';
        return `<span style="color:${{color}};cursor:pointer" data-path="${{partPath}}">${{part}}${{deco}}</span>`;
    }}).join(' <span style="color:white">&gt;</span> ');
}}

function assembleHtmlParts(parts, paths) {{
    const total = parts.length;
    return parts.map((part, i) => {{
        const dot = (spriteConfig.show_nav_accessories && spriteConfig.colored_html_dots) ? `<span class="page-dot" style="color:${{seededColorFromPath(paths[i])}}">●</span>` : '';
        const pageLabel = total > 1 ? `<div class="page-label">${{i+1}}/${{total}}${{dot}}</div>` : '';
        const separator = i < total - 1 ? '<hr class="page-separator">' : '';
        return `<div class=\"text-page\"><div class=\"text-page-inner\">${{part}}</div>${{pageLabel}}</div>${{separator}}`;
    }}).join('');
}}


let ktx2Loader = null;
function initKTX2(renderer) {{
    if (ktx2Loader) return;
    ktx2Loader = new KTX2Loader();
    ktx2Loader.setTranscoderPath('https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/libs/basis/');
    ktx2Loader.detectSupport(renderer);
}}

async function loadSpritesheet(path) {{
    if (textureCache[path]) return textureCache[path];
    if (pendingLoads[path]) return pendingLoads[path];
    pendingLoads[path] = (async () => {{
        const texture = await ktx2Loader.loadAsync(path);
        texture.minFilter = THREE.NearestFilter;
        texture.magFilter = THREE.NearestFilter;
        texture.generateMipmaps = false;
        textureCache[path] = texture;
        delete pendingLoads[path];
        const base = path.replace(/_lod\\d/, '');
        if (!lodAvailable[base]) lodAvailable[base] = {{}};
        const lodMatch = path.match(/_lod(\\d)/);
        if (lodMatch) lodAvailable[base][parseInt(lodMatch[1])] = path;
        if (lodMatch && parseInt(lodMatch[1]) < LOD_COUNT - 1) progress.ss++;
        updateLoader();
        return texture;
    }})();
    return pendingLoads[path];
}}

function getBestLod(ssArray, maxLod) {{
    const base = ssArray[0].replace(/_lod\\d/, '');
    const avail = lodAvailable[base] || {{}};
    for (let l = maxLod; l < LOD_COUNT; l++) {{
        if (avail[l] && textureCache[avail[l]]) return textureCache[avail[l]];
    }}
    for (let l = maxLod - 1; l >= 0; l--) {{
        if (avail[l] && textureCache[avail[l]]) return textureCache[avail[l]];
    }}
    return null;
}}

function getStackMaterial(ssArray, stackName) {{
    const key = `${{ssBase(ssArray)}}::${{stackName}}`;
    if (!stackMaterialCache[key]) {{
        const tex = getBestLod(ssArray, LOD_COUNT - 1) || textureCache[ssArray[LOD_COUNT - 1]];
        stackMaterialCache[key] = new THREE.MeshBasicMaterial({{
            map: tex,
            side: THREE.DoubleSide,
            transparent: true,
            alphaTest: 0.1,
            opacity: 1
        }});
        stackMaterialCache[key]._ssArray = ssArray;
    }}
    return stackMaterialCache[key];
}}

function createAnimInstancedMaterial(ssArray, frameMapTex, frameMapWidth) {{
    const tex = getBestLod(ssArray, LOD_COUNT - 1) || textureCache[ssArray[LOD_COUNT - 1]];
    const SS = spriteConfig.spritesheet_size;
    const mat = new THREE.ShaderMaterial({{
        uniforms: {{
            map: {{ value: tex }},
            time: {{ value: 0.0 }},
            frameMapTex: {{ value: frameMapTex }},
            frameMapWidth: {{ value: frameMapWidth }},
            spritesheetSize: {{ value: SS }}
        }},
        vertexShader: `
            uniform float time;
            uniform sampler2D frameMapTex;
            uniform float frameMapWidth;
            uniform float spritesheetSize;
            attribute float frameMapOffset;
            attribute float frameMapLength;
            attribute float gifDuration;
            attribute float spriteSize;
            attribute float spritesPerRow;
            attribute float instanceOpacity;
            varying vec2 vUv;
            varying float vOpacity;
            void main() {{
                float frame = mod(floor(time / gifDuration), frameMapLength);
                float texIdx = frameMapOffset + frame;
                float tx = (mod(texIdx, frameMapWidth) + 0.5) / frameMapWidth;
                float ty = (floor(texIdx / frameMapWidth) + 0.5) / 1.0;
                float spriteIdx = floor(texture2D(frameMapTex, vec2(tx, ty)).r + 0.5);
                float col = mod(spriteIdx, spritesPerRow);
                float row = floor(spriteIdx / spritesPerRow);
                float sz = spriteSize / spritesheetSize;
                float u0 = col * sz;
                float v0 = 1.0 - (row + 1.0) * sz;
                vUv = vec2(u0 + uv.x * sz, v0 + uv.y * sz);
                vOpacity = instanceOpacity;
                gl_Position = projectionMatrix * modelViewMatrix * instanceMatrix * vec4(position, 1.0);
            }}
        `,
        fragmentShader: `
            uniform sampler2D map;
            varying vec2 vUv;
            varying float vOpacity;
            void main() {{
                vec4 c = texture2D(map, vUv);
                if (c.a < 0.1) discard;
                c.a *= vOpacity;
                gl_FragColor = c;
                #include <colorspace_fragment>
            }}
        `,
        side: THREE.DoubleSide,
        transparent: true
    }});
    mat._ssArray = ssArray;
    return mat;
}}





fetch('data.json')
    .then(r => r.json())
    .then(async d => {{
        const db = d.database;
        dataTree = d.tree;
        spriteConfig = d.sprite_config;

        function hydrate(node) {{
            if (node.ai) node.ai = node.ai.map(id => db[id]);
            if (node.oi) node.oi = node.oi.map(id => db[id]);
            if (node.children) node.children.forEach(hydrate);
        }}
        hydrate(dataTree);
        precomputeCounts(dataTree);

        buildTree(dataTree, document.getElementById('tree-content'));
        progress = {{ss: 0, ssTotal: 0, stacks: 0, stacksTotal: 0, imgs: 0, imgsTotal: 0}};

        const params = new URLSearchParams(window.location.search);
        const path = params.get('path');

        const targetNode = path ? findNodeByPath(dataTree, path) : dataTree;
        
        await renderContent(targetNode || dataTree);
        
        isInitialLoad = false;
        const loaderEl = document.getElementById('loader');
        if (loaderEl) loaderEl.remove();

            const hash = window.location.hash.slice(2);

        if (hash) {{

            const node = findNodeByPath(dataTree, hash);

            if (node) await renderContent(node);

        }}

    }});











window.addEventListener('hashchange', () => {{
    const hash = window.location.hash.slice(2);
    const node = hash ? findNodeByPath(dataTree, hash) : dataTree;
    if (node) renderContent(node);
}});

window.matchMedia('(max-width: 768px)').addEventListener('change', () => {{ if (currentNode) renderContent(currentNode); }});


function precomputeCounts(node) {{
    let count = 0;
    if (node.children) {{
        for (const child of node.children) {{
            count += 1 + precomputeCounts(child);
        }}
    }}
    node._dc = count;
    return count;
}}

function buildTree(node, container, depth = 0, isLast = true, prefix = '') {{
    const connector = '';
    const item = document.createElement('div');
    item.className = 'tree-item';
    item.innerHTML = prefix + connector;
    const link = document.createElement('span');
    link.className = 'tree-link';
    
    const hasChildren = node.children && node.children.length > 0;
    const deco = navDeco(node, 'dot-sm');

    if (hasChildren) {{
        const caret = document.createElement('span');
        caret.className = 'tree-caret';
        caret.textContent = '>';
        caret.onclick = (e) => {{
            e.stopPropagation();
            const cc = link._childContainer;
            if (cc) {{ const opening = cc.style.display === 'none'; cc.style.display = opening ? 'block' : 'none'; updateTreeState(opening); }}
        }};
        link.appendChild(caret);
        link.appendChild(document.createTextNode(' '));
    }}

    const nameSpan = document.createElement('span');
    nameSpan.className = 'tree-name';
    nameSpan.innerHTML = node.name + deco;
    link.appendChild(nameSpan);
    link.dataset.path = node.path;
    if (node.sk && node.sk.length) link.dataset.keywords = node.sk.join(',').toLowerCase();
    
    if (node.hid) {{
        link.style.textDecoration = 'line-through';
        link.style.color = '#666';
        link.style.cursor = 'not-allowed';
        link.title = 'Hidden';
    }} else {{
        nameSpan.onclick = (e) => {{
            e.stopPropagation();
            if (!currentNode || currentNode.path !== node.path) {{
                renderContent(node);
                updateTreeState(true);
            }}
        }};
        link.addEventListener('mouseenter', () => {{
            if (window.innerWidth <= 768) return;
            activeScenes.forEach(s => {{ if (s.driveWireNav) s.driveWireNav(node.path); }});
        }});
        link.addEventListener('mouseleave', () => {{
            if (window.innerWidth <= 768) return;
            activeScenes.forEach(s => {{ if (s.clearWireNav) s.clearWireNav(); }});
        }});
    }}
    
    item.appendChild(link);
    container.appendChild(item);
    
    if (hasChildren) {{
        const childContainer = document.createElement('div');
        childContainer.className = 'tree-children';
        childContainer.style.display = 'none';
        link._childContainer = childContainer;
        const newPrefix = prefix + '  ';
        node.children.forEach((child, i) => {{
            buildTree(child, childContainer, depth + 1, i === node.children.length - 1, newPrefix);
        }});
        container.appendChild(childContainer);
    }}
}}

function updateTreeState(expand) {{
    if (!currentNode) {{
        document.querySelectorAll('.tree-caret').forEach(caret => {{
            const link = caret.closest('.tree-link');
            const cc = link?._childContainer;
            if (cc) caret.textContent = cc.style.display === 'none' ? '>' : 'v';
        }});
        return;
    }}
    const ancestorSet = new Set(), descendantSet = new Set();
    const parts = currentNode.path.split('/');
    for (let i = 1; i < parts.length; i++) ancestorSet.add(parts.slice(0, i).join('/'));
    (function collect(n) {{ n.children.forEach(c => {{ descendantSet.add(c.path); collect(c); }}); }})(currentNode);

    document.querySelectorAll('.tree-link').forEach(link => {{
        const p = link.dataset.path;
        if (expand && ancestorSet.has(p)) {{
            const cc = link._childContainer;
            if (cc) cc.style.display = 'block';
        }}
        let color;
        if (link.style.textDecoration === 'line-through') color = '#666';
        else if (p === currentNode.path) color = '#44f';
        else if (ancestorSet.has(p) || descendantSet.has(p)) color = 'white';
        else color = '#444';
        link.style.color = color;
        const countSpan = link.querySelector('.tree-count');
        if (countSpan) countSpan.style.color = color;
    }});
    document.querySelectorAll('.tree-caret').forEach(caret => {{
        const link = caret.closest('.tree-link');
        const cc = link?._childContainer;
        if (cc) caret.textContent = cc.style.display === 'none' ? '>' : 'v';
    }});
}}

let _treeHoverPath = null;
function _updateTreeCarets() {{
    document.querySelectorAll('.tree-link').forEach(link => {{
        const cc = link._childContainer, caret = cc && link.querySelector('.tree-caret');
        if (caret) caret.textContent = cc.style.display === 'none' ? '>' : 'v';
    }});
}}
function highlightTreePath(path) {{
    if (window.innerWidth <= 768) return;
    if (_treeHoverPath === path) return;
    clearTreeHighlight();
    _treeHoverPath = path;
    const parts = path.split('/'), ancestors = new Set();
    for (let i = 1; i < parts.length; i++) ancestors.add(parts.slice(0, i).join('/'));
    document.querySelectorAll('.tree-link').forEach(link => {{
        const p = link.dataset.path, cc = link._childContainer;
        if (ancestors.has(p) && cc && cc.style.display === 'none') {{ cc.style.display = 'block'; cc._treeHoverOpened = true; }}
        if (p === path) {{ link.classList.add('tree-hover'); link.scrollIntoView({{ block: 'nearest', behavior: 'smooth' }}); }}
    }});
    _updateTreeCarets();
}}
function clearTreeHighlight() {{
    if (!_treeHoverPath) return;
    _treeHoverPath = null;
    document.querySelectorAll('.tree-link').forEach(link => {{
        link.classList.remove('tree-hover');
        const cc = link._childContainer;
        if (cc && cc._treeHoverOpened) {{ cc.style.display = 'none'; cc._treeHoverOpened = false; }}
    }});
    _updateTreeCarets();
}}
            

function disposeScene(sceneData) {{
    if (sceneData.animationId) cancelAnimationFrame(sceneData.animationId);
    if (sceneData.resizeHandler) window.removeEventListener('resize', sceneData.resizeHandler);
    if (sceneData.labelContainer && sceneData.labelContainer.parentNode) {{
        sceneData.labelContainer.parentNode.removeChild(sceneData.labelContainer);
    }}
    if (sceneData.countDiv && sceneData.countDiv.parentNode) {{
        sceneData.countDiv.parentNode.removeChild(sceneData.countDiv);
    }}
    if (sceneData.ssBg) sceneData.ssBg.dispose();
    if (sceneData.renderer) {{
        sceneData.renderer.dispose();
        if (sceneData.renderer.domElement.parentNode) {{
            sceneData.renderer.domElement.parentNode.removeChild(sceneData.renderer.domElement);
        }}
    }}
    if (sceneData.scene) {{
        sceneData.scene.traverse((object) => {{
            if (object.geometry) object.geometry.dispose();
        }});
        sceneData.scene.clear();
    }}
}}

function createSpritesheetBg(container, images, SS, scene, renderer, opacity) {{
    const slotMap = [];
    const keys = new Set();
    images.forEach(img => keys.add(img.ss[0].replace('_lod0.ktx2', '')));
    const ssKeys = [...keys];
    const _tmpMat4 = new THREE.Matrix4();
    images.forEach(img => {{
        if (img.anim && img.fm) return;
        const key = img.ss[0].replace('_lod0.ktx2', '');
        const sz = img.sz;
        const spr = Math.floor(SS / sz);
        const idx = img.idx;
        if (idx === undefined) return;
        slotMap.push({{ key, px: (idx % spr) * sz, py: Math.floor(idx / spr) * sz, pw: sz, ph: sz }});
    }});

    const el = document.createElement('div');
    el.className = 'ss-bg';
    if (opacity < 1) el.style.opacity = opacity;
    const wrap = document.createElement('div');
    wrap.className = 'ss-img-wrapper';
    ssKeys.forEach(key => {{
        const img = document.createElement('img');
        img.src = key + '_lod2.webp';
        img.className = 'ss-bg-img';
        img.dataset.ssKey = key;
        img._lod = 2;
        wrap.appendChild(img);
        [1, 0].forEach(lod => {{
            const pre = new Image();
            pre.onload = () => {{ if (img._lod > lod) {{ img.src = pre.src; img._lod = lod; }} }};
            pre.src = key + '_lod' + lod + '.webp';
        }});
    }});
    const highlight = document.createElement('div');
    highlight.className = 'ss-highlight';
    const info = document.createElement('div');
    info.className = 'ss-info-label';
    wrap.appendChild(highlight);
    wrap.appendChild(info);
    el.appendChild(wrap);
    container.appendChild(el);

    const resize = () => {{
        const cw = container.clientWidth * 0.8;
        const ch = container.clientHeight * 0.8;
        const gap = (ssKeys.length - 1) * 4;
        const sz = Math.floor(Math.min((cw - gap) / ssKeys.length, ch));
        wrap.querySelectorAll('.ss-bg-img').forEach(img => {{ img.style.width = sz + 'px'; img.style.height = sz + 'px'; }});
    }};
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(container);

    renderer.setClearColor(0x000000, 0);
    renderer.domElement.style.position = 'relative';
    renderer.domElement.style.zIndex = '1';

    const mouse = new THREE.Vector2(-999, -999);
    const raycaster = new THREE.Raycaster();
    let dirty = false;
    let meshCache = null;
    let wireframe = null;

    const onMove = (e) => {{
        const r = renderer.domElement.getBoundingClientRect();
        mouse.x = ((e.clientX - r.left) / r.width) * 2 - 1;
        mouse.y = -((e.clientY - r.top) / r.height) * 2 + 1;
        dirty = true;
    }};
    const onLeave = () => {{
        dirty = false;
        highlight.style.display = 'none';
        info.style.display = 'none';
        if (wireframe) wireframe.visible = false;
    }};
    renderer.domElement.addEventListener('mousemove', onMove);
    renderer.domElement.addEventListener('mouseleave', onLeave);

    function update(camera, time) {{
        if (!dirty) return;
        dirty = false;
        if (!meshCache) {{
            meshCache = [];
            scene.traverse(obj => {{ if (obj.isMesh || obj.isInstancedMesh) meshCache.push(obj); }});
        }}
        scene.updateMatrixWorld();
        raycaster.setFromCamera(mouse, camera);
        const hits = raycaster.intersectObjects(meshCache);
        if (hits.length > 0) {{
            const hit = hits[0];
            const mat = hit.object.material;
            const isInstanced = hit.object.isInstancedMesh;
            const key = mat._ssArray ? mat._ssArray[0].replace('_lod0.ktx2', '') : null;

            let slot = null;
            if (isInstanced && hit.instanceId != null && hit.object._animInstances) {{
                const inst = hit.object._animInstances[hit.instanceId];
                if (inst) {{
                    const imgData = inst.imgData;
                    const sz = imgData.sz;
                    const spr = Math.floor(SS / sz);
                    const frame = Math.floor(time / imgData.gd) % imgData.fm.length;
                    const idx = imgData.fm[frame];
                    slot = {{ key, px: (idx % spr) * sz, py: Math.floor(idx / spr) * sz, pw: sz, ph: sz }};
                    dirty = true;
                }}
            }} else if (hit.uv) {{
                const px = hit.uv.x * SS;
                const py = (1 - hit.uv.y) * SS;
                for (const s of slotMap) {{
                    if (s.key === key && px >= s.px && px < s.px + s.pw && py >= s.py && py < s.py + s.ph) {{ slot = s; break; }}
                }}
            }}
            const imgEl = key ? wrap.querySelector('img[data-ss-key="' + key + '"]') : null;
            if (slot && imgEl) {{
                const ir = imgEl.getBoundingClientRect();
                const wr = wrap.getBoundingClientRect();
                const sc = ir.width / SS;
                highlight.style.display = 'block';
                highlight.style.left = (ir.left - wr.left + slot.px * sc) + 'px';
                highlight.style.top = (ir.top - wr.top + slot.py * sc) + 'px';
                highlight.style.width = (slot.pw * sc) + 'px';
                highlight.style.height = (slot.ph * sc) + 'px';
                const u0 = (slot.px / SS).toFixed(3);
                const v0 = (slot.py / SS).toFixed(3);
                const u1 = ((slot.px + slot.pw) / SS).toFixed(3);
                const v1 = ((slot.py + slot.ph) / SS).toFixed(3);
                info.textContent = key.split('/').pop() + '  uv [' + u0 + ',' + v0 + '] [' + u1 + ',' + v1 + ']  ' + slot.pw + 'x' + slot.ph;
                info.style.display = 'block';
            }} else {{
                highlight.style.display = 'none';
                info.style.display = 'none';
            }}

            let pts;
            if (isInstanced && hit.instanceId != null) {{
                const geo = hit.object.geometry;
                const pos = geo.attributes.position;
                hit.object.getMatrixAt(hit.instanceId, _tmpMat4);
                const _v = new THREE.Vector3();
                const order = [0, 1, 3, 2];
                pts = [];
                for (let i = 0; i < 4; i++) {{
                    _v.set(pos.getX(order[i]), pos.getY(order[i]), pos.getZ(order[i]));
                    _v.applyMatrix4(_tmpMat4);
                    pts.push(_v.x, _v.y, _v.z);
                }}
                pts.push(pts[0], pts[1], pts[2]);
            }} else {{
                const geo = hit.object.geometry;
                const pos = geo.attributes.position;
                const vi = Math.floor(hit.faceIndex / 2) * 4;
                const order = [0, 1, 3, 2];
                pts = [];
                for (let i = 0; i < 4; i++) pts.push(pos.getX(vi + order[i]), pos.getY(vi + order[i]), pos.getZ(vi + order[i]));
                pts.push(pts[0], pts[1], pts[2]);
            }}
            if (!wireframe) {{
                const wGeo = new LineGeometry();
                wGeo.setPositions(pts);
                const wMat = new LineMaterial({{ color: 0x4444ff, linewidth: 3 }});
                wMat.resolution.set(container.clientWidth, container.clientHeight);
                wireframe = new Line2(wGeo, wMat);
                wireframe.renderOrder = 999;
                wireframe.frustumCulled = false;
                wireframe.computeLineDistances();
                scene.add(wireframe);
            }} else {{
                wireframe.geometry.setPositions(pts);
                wireframe.computeLineDistances();
                wireframe.material.resolution.set(container.clientWidth, container.clientHeight);
            }}
            wireframe.visible = true;
        }} else {{
            highlight.style.display = 'none';
            info.style.display = 'none';
            if (wireframe) wireframe.visible = false;
        }}
    }}

    function dispose() {{
        renderer.domElement.removeEventListener('mousemove', onMove);
        renderer.domElement.removeEventListener('mouseleave', onLeave);
        ro.disconnect();
        if (wireframe) {{ wireframe.geometry.dispose(); wireframe.material.dispose(); }}
        if (el.parentNode) el.parentNode.removeChild(el);
    }}

    return {{ el, update, dispose }};
}}

async function createThreeScene(container, images, node, highlightPath) {{
    const scene = new THREE.Scene();
    const grouped = {{}};
    const stackGroups = {{}};
    const animInstances = [];

    const SPRITESHEET_SIZE = spriteConfig.spritesheet_size;
    const STACK_SPACING = spriteConfig.stack_spacing;
    const STACK_REVERSE = spriteConfig.stack_reverse;
    const STACK_DIM_OPACITY = spriteConfig.stack_dim_opacity;
    const SEED = spriteConfig.seed;
    const ORDERED_GRID_LAYOUT = spriteConfig.ordered_grid_layout;
    const ROTATION_SPEED = spriteConfig.rotation_speed;

    images.forEach(imgData => {{
        const parts = imgData.path.split('/');
        const folder = parts.slice(0, -1).join('/') || 'root';
        if (!grouped[folder]) grouped[folder] = [];
        grouped[folder].push(imgData);
    }});
    
    const folders = Object.keys(grouped);
    const isLastStack = folders.length === 1;
    folders.forEach(folder => {{ grouped[folder].sort((a, b) => a.gi - b.gi); }});
   
    progress.imgsTotal += images.length;
    progress.stacksTotal += folders.length;
    const uniqueSS = new Set();
    images.forEach(i => i.ss.forEach(p => uniqueSS.add(p)));
    const uniqueSSBase = new Set();
    images.forEach(i => {{ for (let l = 0; l < LOD_COUNT - 1; l++) if (i.ss[l]) uniqueSSBase.add(i.ss[l]); }});
    progress.ssTotal += uniqueSSBase.size;
    uniqueSSBase.forEach(ss => {{ if (textureCache[ss]) progress.ss++; }});
    updateLoader();
 
    const baseGeometry = new THREE.PlaneGeometry(1, 1);
    
    const spacing = 1;
    let cols, rows, offsetX, offsetZ;
    let gridGroups;
    
    if (node.grid_layout) {{
        const parts = node.grid_layout.split('x');
        cols = parseInt(parts[0]);
        rows = parseInt(parts[1]);
        offsetX = (cols - 1) * spacing / 2;
        offsetZ = (rows - 1) * spacing / 2;
    }} else {{
        gridGroups = [];
        const processedFolders = new Set();
        function collectGridChildren(n) {{
            if (n.hid) return;
            const childFolders = folders.filter(f => f.startsWith(n.path + '/') || f === n.path);
            if (n.grid_layout && n.ai.length > 0 && childFolders.length > 0) {{
                const [gCols, gRows] = ORDERED_GRID_LAYOUT ? n.grid_layout.split('x').map(Number) : [1, 1];
                gridGroups.push({{ folders: childFolders, cols: gCols, rows: gRows, path: n.path }});
                childFolders.forEach(f => processedFolders.add(f));
            }} else if (n.children.length > 0) {{
                n.children.forEach(child => collectGridChildren(child));
            }} else if (childFolders.length > 0) {{
                gridGroups.push({{ folders: childFolders, cols: 1, rows: 1, path: n.path }});
                childFolders.forEach(f => processedFolders.add(f));
            }}
        }}
        node.children.forEach(child => collectGridChildren(child));
        
        if (gridGroups.length > 0) {{
            const occupiedGrid = new Map();
            const isOccupied = (gx, gy, w, h) => {{
                for (let dy = 0; dy < h; dy++) {{
                    for (let dx = 0; dx < w; dx++) {{
                        if (occupiedGrid.has(`${{gx + dx}},${{gy + dy}}`)) return true;
                    }}
                }}
                return false;
            }};
            const occupy = (gx, gy, w, h) => {{
                for (let dy = 0; dy < h; dy++) {{
                    for (let dx = 0; dx < w; dx++) {{
                        occupiedGrid.set(`${{gx + dx}},${{gy + dy}}`, true);
                    }}
                }}
            }};
            const findSpiralPosition = (w, h) => {{
                if (occupiedGrid.size === 0) return {{ x: 0, y: 0 }};
                let radius = 1;
                while (radius < 100) {{
                    for (let dy = -radius; dy <= radius; dy++) {{
                        for (let dx = -radius; dx <= radius; dx++) {{
                            if (Math.abs(dx) === radius || Math.abs(dy) === radius) {{
                                if (!isOccupied(dx, dy, w, h)) return {{ x: dx, y: dy }};
                            }}
                        }}
                    }}
                    radius++;
                }}
                return {{ x: 0, y: 0 }};
            }};
            gridGroups.forEach(group => {{
                const pos = findSpiralPosition(group.cols, group.rows);
                group.gridX = pos.x;
                group.gridY = pos.y;
                occupy(pos.x, pos.y, group.cols, group.rows);
            }});
            let minGridX = Infinity, maxGridX = -Infinity;
            let minGridZ = Infinity, maxGridZ = -Infinity;
            gridGroups.forEach(group => {{
                minGridX = Math.min(minGridX, group.gridX);
                maxGridX = Math.max(maxGridX, group.gridX + group.cols - 1);
                minGridZ = Math.min(minGridZ, group.gridY);
                maxGridZ = Math.max(maxGridZ, group.gridY + group.rows - 1);
            }});
            offsetX = ((maxGridX + minGridX) / 2) * spacing;
            offsetZ = ((maxGridZ + minGridZ) / 2) * spacing;
        }} else {{
            cols = Math.ceil(Math.sqrt(folders.length));
            rows = Math.ceil(folders.length / cols);
            offsetX = (cols - 1) * spacing / 2;
            offsetZ = (rows - 1) * spacing / 2;
        }}
    }}

 let minX = Infinity, maxX = -Infinity;
    let minZ = Infinity, maxZ = -Infinity;
    
    const calculateBounds = (x, z) => {{
        minX = Math.min(minX, x);
        maxX = Math.max(maxX, x);
        minZ = Math.min(minZ, z);
        maxZ = Math.max(maxZ, z);
    }};
    const folderPositions = {{}};
    
    // --- 2. Iterate to Calculate Scene Size ---
    if (node.grid_layout) {{
        folders.forEach((folder, stackIdx) => {{
            const row = Math.floor(stackIdx / cols);
            const col = stackIdx % cols;
            const x = col * spacing - offsetX;
            const z = row * spacing - offsetZ;
            folderPositions[folder] = {{x, z}};
            calculateBounds(x, z);
        }});
    }} else if (gridGroups && gridGroups.length > 0) {{
        gridGroups.forEach(group => {{
            group.folders.forEach((folder, localIdx) => {{
                const row = Math.floor(localIdx / group.cols);
                const col = localIdx % group.cols;
                const x = (group.gridX + col) * spacing - offsetX;
                const z = (group.gridY + row) * spacing - offsetZ;
                folderPositions[folder] = {{x, z}};
                calculateBounds(x, z);
            }});
        }});
    }} else {{
        folders.forEach((folder, stackIdx) => {{
            const row = Math.floor(stackIdx / cols);
            const col = stackIdx % cols;
            const x = col * spacing - offsetX;
            const z = row * spacing - offsetZ;
            folderPositions[folder] = {{x, z}};
            calculateBounds(x, z);
        }});
    }}

    folders.forEach((folder, stackIdx) => {{
        if (folderPositions[folder]) return;
        const col = stackIdx % (cols || 1);
        const row = Math.floor(stackIdx / (cols || 1));
        const x = col * spacing - (offsetX || 0);
        const z = row * spacing - (offsetZ || 0);
        folderPositions[folder] = {{x, z}};
        calculateBounds(x, z);
    }});

    const meshMaxSize = 1.5 * 1.5;
    const geomWidth = maxX - minX + meshMaxSize;
    const geomDepth = maxZ - minZ + meshMaxSize;
    
    // Add margin (was 15 previously)
    const maxSceneDim = Math.max(geomWidth, geomDepth) + 5;

    // Calculate Aspect Ratio from the Container
    const aspectRatio = container.clientWidth / container.clientHeight;

    // FIT LOGIC: 
    // If Portrait (aspect < 1), div is narrow -> Zoom out (increase frustum) to fit width.
    // If Landscape (aspect >= 1), div is wide -> Height is the limiter, use maxSceneDim.
    const fitFrustumSize = aspectRatio < 1 ? maxSceneDim / aspectRatio : maxSceneDim;

    // ZOOM LOGIC:
    const RANDOM_ZOOM = spriteConfig.random_zoom;
    const GLOBAL_ZOOM_VALUE = spriteConfig.zoom_value;
    const nodeZoom = (node && node.mzm != null) ? node.mzm : null;

    let randomZoom;
    if (nodeZoom != null) {{
        randomZoom = nodeZoom;
    }} else if (!RANDOM_ZOOM) {{
        randomZoom = GLOBAL_ZOOM_VALUE;
    }} else {{
        const seed = images.map(img => img.gi).reduce((a, b) => a + b, 0);
        const rand = seededRandom(seed * SEED);
        if (rand < 0.33) {{
            randomZoom = 0.1;
        }} else if (rand < 0.66) {{
            randomZoom = 0.4;
        }} else {{
            randomZoom = 1.0;
        }}
    }}
    


    // Apply the random multiplier to the perfect fit
    const frustumSize = fitFrustumSize * randomZoom;

    // --- 4. Create Camera ---
    const camera = new THREE.OrthographicCamera(
        frustumSize * aspectRatio / -2,
        frustumSize * aspectRatio / 2,
        frustumSize / 2,
        frustumSize / -2,
        0.1,
        1000
    );
    
    const renderer = new THREE.WebGLRenderer({{ 
        alpha: true, 
        antialias: true,
        powerPreference: 'high-performance'
    }});
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x000000);
    initKTX2(renderer);

    const spriteBgEnabled = node.sbg != null ? node.sbg : spriteConfig.sprite_bg;
    const spriteBgOpacity = node.sbo != null ? node.sbo : spriteConfig.sprite_bg_opacity;
    const ssBg = isLastStack && spriteBgEnabled ? createSpritesheetBg(container, images, SPRITESHEET_SIZE, scene, renderer, spriteBgOpacity) : null;

    container.appendChild(renderer.domElement);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enablePan = true;
    controls.zoomSpeed = 3.0;

    let maxStackHeight = 0;
    folders.forEach(folder => {{
        // Find the specific node for this folder to check for custom spacing
        const folderNode = findNodeByPath(dataTree, folder);
        const localSpacing = (folderNode && folderNode.msp != null) ? folderNode.msp : STACK_SPACING;
        
        maxStackHeight = Math.max(maxStackHeight, grouped[folder].length * localSpacing);
        console.log(folder, grouped[folder].length, 'h:', grouped[folder].length * localSpacing);

            
    }});
    const midHeight = maxStackHeight / 2;

    camera.position.set(10, 10 + midHeight, 10);
    controls.target.set(0, midHeight, 0);

    const labelContainer = document.createElement('div');
    labelContainer.id = 'label-container';
    container.appendChild(labelContainer);

    const wireSvg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    wireSvg.classList.add('nav-wire-svg');
    wireSvg.setAttribute('width', '100%');
    wireSvg.setAttribute('height', '100%');
    wireSvg.style.overflow = 'visible';
    container.appendChild(wireSvg);
    function makeSvgPath(cls) {{ const p = document.createElementNS('http://www.w3.org/2000/svg', 'path'); p.classList.add(cls); wireSvg.appendChild(p); return p; }}
    const wirePathDim = makeSvgPath('nav-wire-path-dim');
    const wirePathAncestor = makeSvgPath('nav-wire-path-ancestor');
    const wirePath = makeSvgPath('nav-wire-path');
    const wireArrowG = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    wireSvg.appendChild(wireArrowG);
    let mouseX = 0, mouseY = 0, mouseIsDown = false;
    document.addEventListener('pointerdown', (e) => {{ if (e.button === 0 || e.button === 1) mouseIsDown = true; }});
    document.addEventListener('pointerup', (e) => {{ if (e.button === 0 || e.button === 1) mouseIsDown = false; }});
    container.addEventListener('mousemove', (e) => {{
        const rect = container.getBoundingClientRect();
        mouseX = e.clientX - rect.left;
        mouseY = e.clientY - rect.top;
    }});

 
    const toggleBtn = document.createElement('span');
    toggleBtn.id = 'label-toggle';
    toggleBtn.textContent = 'hide labels';
    toggleBtn.style.cssText = 'position:absolute;bottom:5px;left:5px;color:#444;font-family:monospace;font-size:11px;cursor:pointer;z-index:100;user-select:none;';
    let labelsVisible = true;
    toggleBtn.onclick = () => {{
        labelsVisible = !labelsVisible;
        labelContainer.style.display = labelsVisible ? 'block' : 'none';
        wireSvg.style.display = labelsVisible ? '' : 'none';
        toggleBtn.textContent = labelsVisible ? 'hide labels' : 'show labels';
    }};
    container.appendChild(toggleBtn);

    const countDiv = document.createElement('div');
    countDiv.style.cssText = 'position:absolute;bottom:5px;right:5px;color:white;font-family:monospace;';
    container.appendChild(countDiv);

    let loadedStacks = 0;
    let loadedImages = 0;
    const totalStacks = folders.length;
    const totalImages = images.length;

const updateCount = () => {{
        const gridInfo = node.grid_layout ? ` [${{node.grid_layout}}]` : '';
        countDiv.innerHTML = `<span id="info-zoom">zoom: ${{randomZoom.toFixed(2)}}${{gridInfo}}</span><span class="info-sep"> | </span><span id="info-stacks">${{loadedStacks + 1}}/${{totalStacks}} stacks</span><span class="info-sep"> | </span><span id="info-images">${{loadedImages}}/${{totalImages}} images</span>`;
    }};
            
    updateCount();

    const stackLabels = [];
    let currentFocusedParent = null;
    let activeLabelPath = null;
    let activeGroupPath = null;
    function focusStackGroup(parentFolder) {{
        if (currentFocusedParent === parentFolder) {{
            unfocusAll();
            currentFocusedParent = null;
        }} else {{
            for (const [folder, data] of Object.entries(stackGroups)) {{
                const op = (folder.startsWith(parentFolder + '/') || folder === parentFolder) ? 1.0 : STACK_DIM_OPACITY;
                data.materials.forEach(m => m.opacity = op);
            }}
            for (const ai of animInstances) {{
                let dirty = false;
                for (let i = 0; i < ai.instances.length; i++) {{
                    const fn = ai.instances[i].folderName;
                    const op = (fn.startsWith(parentFolder + '/') || fn === parentFolder) ? 1.0 : STACK_DIM_OPACITY;
                    if (ai.opAttr.array[i] !== op) {{ ai.opAttr.array[i] = op; dirty = true; }}
                }}
                if (dirty) ai.opAttr.needsUpdate = true;
            }}
            stackLabels.forEach(sl => {{
                const match = sl.folderName.startsWith(parentFolder + '/') || sl.folderName === parentFolder;
                sl.element.style.opacity = match ? '1.0' : String(STACK_DIM_OPACITY);
            }});
            currentFocusedParent = parentFolder;
        }}
    }}
    function unfocusAll() {{
        for (const [folder, data] of Object.entries(stackGroups)) {{
            data.materials.forEach(m => m.opacity = 1.0);
        }}
        for (const ai of animInstances) {{
            ai.opAttr.array.fill(1.0);
            ai.opAttr.needsUpdate = true;
        }}
        stackLabels.forEach(sl => sl.element.style.opacity = '1.0');
        currentFocusedParent = null;
    }}

    const loadingPromise = (async () => {{
        const allSSArrays = new Set();
        folders.forEach(folder => {{
            grouped[folder].forEach(img => allSSArrays.add(JSON.stringify(img.ss)));
        }});
        const uniqueSSArrays = [...allSSArrays].map(s => JSON.parse(s));
        
        const lod2Paths = uniqueSSArrays.map(ss => ss[LOD_COUNT - 1]);
        const lod1Paths = uniqueSSArrays.map(ss => ss[LOD_COUNT - 2]);
        await Promise.all([...lod2Paths, ...lod1Paths].map(p => loadSpritesheet(p)));
        
        const groupLabelData = {{}};
        const animBuckets = {{}};
        for (let stackIdx = 0; stackIdx < folders.length; stackIdx++) {{
            const folderName = folders[stackIdx];


            // --- NEW: Retrieve Custom Spacing for this specific stack ---
            const folderNode = findNodeByPath(dataTree, folderName);
            const localSpacing = (folderNode && folderNode.msp != null) ? folderNode.msp : STACK_SPACING;
            const localReverse = (folderNode && folderNode.mrv != null) ? folderNode.mrv : STACK_REVERSE;
            // ------------------------------------------------------------


            const stackImages = grouped[folderName];
            const pos = folderPositions[folderName];
            const xPos = pos.x;
            const zPos = pos.z;

            const stackGroup = new THREE.Group();
            const stackMaterials = new Set();
            const mergeBuckets = {{}};

            for (let i = 0; i < stackImages.length; i++) {{
                const imgData = stackImages[i];
                stackMaterials.add(getStackMaterial(imgData.ss, folderName));
                
                const aspect = imgData.w / imgData.h;
                const height = 1;
                const width = height * aspect;

                const y = localReverse ? (stackImages.length - 1 - i) * localSpacing : i * localSpacing;
                const matrix = new THREE.Matrix4();
                const position = new THREE.Vector3(xPos, y, zPos);
                const rotation = new THREE.Euler(Math.PI / 2, Math.PI, Math.PI);
                const quaternion = new THREE.Quaternion().setFromEuler(rotation);
                const scale = new THREE.Vector3(width, height, 1);
                matrix.compose(position, quaternion, scale);
                
                const currentSpriteSize = imgData.sz; 
                const SPRITES_PER_ROW = Math.floor(SPRITESHEET_SIZE / currentSpriteSize);

                if (imgData.anim) {{
                    const bKey = ssBase(imgData.ss);
                    if (!animBuckets[bKey]) animBuckets[bKey] = {{ ssArray: imgData.ss, instances: [] }};
                    animBuckets[bKey].instances.push({{
                        matrix, imgData, folderName,
                        spritesPerRow: SPRITES_PER_ROW,
                        lastFrame: -1
                    }});
                }} else {{
                    const bKey = ssBase(imgData.ss);
                    if (!mergeBuckets[bKey]) mergeBuckets[bKey] = {{ geoms: [], ssArray: imgData.ss }};
                    
                    const geometry = baseGeometry.clone();
                    geometry.applyMatrix4(matrix);
                    
                    const idx = imgData.idx;
                    const sprite_col = idx % SPRITES_PER_ROW;
                    const sprite_row = Math.floor(idx / SPRITES_PER_ROW);
                    const u_start = (sprite_col * currentSpriteSize) / SPRITESHEET_SIZE;
                    const u_end = u_start + imgData.w / SPRITESHEET_SIZE;
                    const v_start = 1 - ((sprite_row + 1) * currentSpriteSize) / SPRITESHEET_SIZE;
                    const v_end = 1 - (sprite_row * currentSpriteSize) / SPRITESHEET_SIZE;
                    
                    const uvs = geometry.attributes.uv;
                    uvs.setXY(0, u_start, v_end);
                    uvs.setXY(1, u_end, v_end);
                    uvs.setXY(2, u_start, v_start);
                    uvs.setXY(3, u_end, v_start);
                    
                    mergeBuckets[bKey].geoms.push(geometry);
                }}
                loadedImages++;
                progress.imgs++;
            }}

            for (const [bKey, bucket] of Object.entries(mergeBuckets)) {{
                if (bucket.geoms.length > 0) {{
                    const mergedGeo = mergeGeometries(bucket.geoms);
                    const mat = getStackMaterial(bucket.ssArray, folderName);
                    const mesh = new THREE.Mesh(mergedGeo, mat);
                    stackGroup.add(mesh);
                    bucket.geoms.forEach(g => g.dispose());
                }}
            }}
            scene.add(stackGroup);
            stackGroups[folderName] = {{ group: stackGroup, materials: stackMaterials }};

            updateCount();
            loadedStacks++;
            progress.stacks++;
            if (progress.stacks % 5 === 0) await updateLoaderAsync();


            // Collect group data for labels
            const topY = (stackImages.length - 1) * localSpacing;
            const scenePath = node.path || '';
            let relPath = '';
            if (folderName.startsWith(scenePath + '/')) {{
                relPath = folderName.slice(scenePath.length + 1);
            }} else if (scenePath === '') {{
                relPath = folderName;
            }}
            const groupKey = relPath.split('/').filter(p => p)[0] || folderName.split('/').pop();
            if (!groupLabelData[groupKey]) {{
                groupLabelData[groupKey] = {{ maxY: topY, xPos, zPos, folders: [], scenePath }};
            }}
            if (topY > groupLabelData[groupKey].maxY) {{
                groupLabelData[groupKey].maxY = topY;
                groupLabelData[groupKey].xPos = xPos;
                groupLabelData[groupKey].zPos = zPos;
            }}
            groupLabelData[groupKey].folders.push({{ folderName, relPath, imageCount: stackImages.length, xPos, zPos, topY }});
        }}

        for (const [bKey, bucket] of Object.entries(animBuckets)) {{
            const count = bucket.instances.length;
            if (count === 0) continue;

            const allFrames = [];
            const offsets = new Float32Array(count);
            const lengths = new Float32Array(count);
            const durations = new Float32Array(count);
            const sizes = new Float32Array(count);
            const sprs = new Float32Array(count);
            const opacities = new Float32Array(count);

            for (let i = 0; i < count; i++) {{
                const inst = bucket.instances[i];
                offsets[i] = allFrames.length;
                lengths[i] = inst.imgData.fm.length;
                durations[i] = inst.imgData.gd;
                sizes[i] = inst.imgData.sz;
                sprs[i] = inst.spritesPerRow;
                opacities[i] = 1.0;
                for (const f of inst.imgData.fm) allFrames.push(f);
            }}

            const fmWidth = allFrames.length;
            const fmData = new Float32Array(allFrames);
            const frameMapTex = new THREE.DataTexture(fmData, fmWidth, 1, THREE.RedFormat, THREE.FloatType);
            frameMapTex.minFilter = THREE.NearestFilter;
            frameMapTex.magFilter = THREE.NearestFilter;
            frameMapTex.needsUpdate = true;

            const mat = createAnimInstancedMaterial(bucket.ssArray, frameMapTex, fmWidth);
            const geo = baseGeometry.clone();
            const iMesh = new THREE.InstancedMesh(geo, mat, count);
            for (let i = 0; i < count; i++) iMesh.setMatrixAt(i, bucket.instances[i].matrix);
            iMesh.instanceMatrix.needsUpdate = true;

            geo.setAttribute('frameMapOffset', new THREE.InstancedBufferAttribute(offsets, 1));
            geo.setAttribute('frameMapLength', new THREE.InstancedBufferAttribute(lengths, 1));
            geo.setAttribute('gifDuration', new THREE.InstancedBufferAttribute(durations, 1));
            geo.setAttribute('spriteSize', new THREE.InstancedBufferAttribute(sizes, 1));
            geo.setAttribute('spritesPerRow', new THREE.InstancedBufferAttribute(sprs, 1));
            const opAttr = new THREE.InstancedBufferAttribute(opacities, 1);
            opAttr.setUsage(THREE.DynamicDrawUsage);
            geo.setAttribute('instanceOpacity', opAttr);

            iMesh.frustumCulled = false;
            iMesh._animInstances = bucket.instances;
            scene.add(iMesh);
            animInstances.push({{ mesh: iMesh, instances: bucket.instances, opAttr, ssArray: bucket.ssArray, mat }});
        }}

        // Create spatial labels from recursive tree
        const scenePath = node.path || '';


        function applyDimming(targetPath) {{
            for (const [folder, data] of Object.entries(stackGroups)) {{
                const isMatch = folder.startsWith(targetPath + '/') || folder === targetPath;
                const isAncestor = targetPath.startsWith(folder + '/');
                const op = (isMatch || isAncestor) ? 1.0 : STACK_DIM_OPACITY;
                data.materials.forEach(m => m.opacity = op);
            }}
            for (const ai of animInstances) {{
                let dirty = false;
                for (let i = 0; i < ai.instances.length; i++) {{
                    const fn = ai.instances[i].folderName;
                    const isMatch = fn.startsWith(targetPath + '/') || fn === targetPath;
                    const isAncestor = targetPath.startsWith(fn + '/');
                    const op = (isMatch || isAncestor) ? 1.0 : STACK_DIM_OPACITY;
                    if (ai.opAttr.array[i] !== op) {{ ai.opAttr.array[i] = op; dirty = true; }}
                }}
                if (dirty) ai.opAttr.needsUpdate = true;
            }}
            stackLabels.forEach(sl => {{
                const isMatch = sl.folderName.startsWith(targetPath + '/') || sl.folderName === targetPath;
                const isAncestor = targetPath.startsWith(sl.folderName + '/');
                sl.element.style.opacity = (isMatch || isAncestor) ? '1.0' : String(STACK_DIM_OPACITY);
            }});
        }}

        function clearDimming() {{
            for (const data of Object.values(stackGroups)) data.materials.forEach(m => m.opacity = 1.0);
            for (const ai of animInstances) {{
                ai.opAttr.array.fill(1.0);
                ai.opAttr.needsUpdate = true;
            }}
            stackLabels.forEach(sl => sl.element.style.opacity = '1.0');
        }}

        function setNavVisible(parentPath) {{
            activeLabelPath = parentPath;
            stackLabels.forEach(sl => {{
                if (sl.isGroup) return;
                const isDirectChild = sl.navParentPath === parentPath;
                const isAncestorChild = parentPath.startsWith(sl.navParentPath + '/');
                sl.navVisible = isDirectChild || isAncestorChild;
            }});
        }}

        function clearNav() {{
            activeLabelPath = null;
            activeGroupPath = null;
            stackLabels.forEach(sl => {{ if (!sl.isGroup) sl.navVisible = false; }});
        }}

        let treeDrivenNav = false;

        function resetNav() {{
            treeDrivenNav = false;
            clearNav();
            clearDimming();
            clearTreeHighlight();
        }}

        function resetAll() {{
            resetNav();
            unfocusAll();
        }}

        sceneData.isTreeDriven = () => treeDrivenNav;

        sceneData.driveWireNav = function(targetPath) {{
            if (currentFocusedParent || mouseIsDown) return;
            treeDrivenNav = true;
            const group = stackLabels.find(sl => sl.isGroup && (targetPath === sl.folderName || targetPath.startsWith(sl.folderName + '/')));
            if (!group) return;
            activeGroupPath = group.folderName;
            setNavVisible(targetPath);
            applyDimming(targetPath);
        }};

        sceneData.clearWireNav = function() {{
            if (currentFocusedParent) return;
            resetNav();
        }};

        document.addEventListener('keydown', (e) => {{ if (e.key === 'Escape') resetAll(); }});

        let _ptrDown = null;
        container.addEventListener('pointerdown', (e) => {{ _ptrDown = {{ x: e.clientX, y: e.clientY }}; }});
        container.addEventListener('click', (e) => {{
            if (!_ptrDown) return;
            const dx = e.clientX - _ptrDown.x, dy = e.clientY - _ptrDown.y;
            if (dx * dx + dy * dy > 25 || e.target.closest('.stack-label, .div-nav, button, #label-toggle')) return;
            resetAll();
        }});

        // Build a position tree from flat folder list per group
        function buildPositionTree(folders, basePath) {{
            const root = {{ children: {{}}, path: basePath, maxY: 0, xPos: 0, zPos: 0 }};
            for (const f of folders) {{
                const sub = f.relPath.split('/').filter(p => p).slice(1);
                let cur = root;
                let curPath = basePath;
                for (let i = 0; i < sub.length; i++) {{
                    curPath = curPath + '/' + sub[i];
                    if (!cur.children[sub[i]]) {{
                        cur.children[sub[i]] = {{ children: {{}}, path: curPath, maxY: 0, xPos: f.xPos, zPos: f.zPos }};
                    }}
                    cur = cur.children[sub[i]];
                }}
                if (f.topY > cur.maxY) {{
                    cur.maxY = f.topY;
                    cur.xPos = f.xPos;
                    cur.zPos = f.zPos;
                }}
            }}
            function propagate(n) {{
                for (const child of Object.values(n.children)) {{
                    propagate(child);
                    if (child.maxY > n.maxY) {{
                        n.maxY = child.maxY;
                        n.xPos = child.xPos;
                        n.zPos = child.zPos;
                    }}
                }}
            }}
            propagate(root);
            return root;
        }}

        function createLabel(name, path, worldPos, isGroup, navParentPath, rootGroupPath) {{
            const el = document.createElement('span');
            el.className = 'stack-label' + (isGroup ? '' : ' nav-child-label');
            el.dataset.folderName = path;
            const isHighlight = path === highlightPath;
            const colorStyle = isHighlight ? 'color:#44f;' : '';
            if (isHighlight) el.style.borderColor = '#44f';
            el.innerHTML = `<span style="padding:2px 4px;cursor:pointer;${{colorStyle}}" data-path="${{path}}">${{name}}</span>`;
            el.addEventListener('wheel', (e) => {{ e.stopPropagation(); renderer.domElement.dispatchEvent(new WheelEvent('wheel', e)); }}, {{ passive: false }});
            labelContainer.appendChild(el);

            const entry = {{ element: el, position: worldPos, xPos: worldPos.x, zPos: worldPos.z, folderName: path, isGroup, isAncestor: false, navParentPath, navVisible: false }};
            stackLabels.push(entry);

            el.addEventListener('mouseenter', () => {{
                if (currentFocusedParent || mouseIsDown) return;
                treeDrivenNav = false;
                activeGroupPath = rootGroupPath;
                setNavVisible(path);
                applyDimming(path);
                highlightTreePath(path);
            }});
            el.addEventListener('mouseleave', () => {{
                if (currentFocusedParent) return;
                if (activeGroupPath === rootGroupPath) return;
                resetNav();
            }});

            el.querySelector('span[data-path]').onclick = (e) => {{
                e.stopPropagation();
                const clickedNode = findNodeByPath(dataTree, path);
                if (clickedNode && clickedNode.path === node.path) renderContent(clickedNode);
                else if (clickedNode) renderContent(clickedNode);
            }};

            return entry;
        }}

        function createChildLabels(treeNode, parentPath, rootGroupPath) {{
            for (const [name, child] of Object.entries(treeNode.children)) {{
                const wp = new THREE.Vector3(child.xPos, child.maxY, child.zPos);
                createLabel(name, child.path, wp, false, parentPath, rootGroupPath);
                if (Object.keys(child.children).length > 0) {{
                    createChildLabels(child, child.path, rootGroupPath);
                }}
            }}
        }}

        for (const [groupKey, gData] of Object.entries(groupLabelData)) {{
            const rawGroupPath = scenePath ? scenePath + '/' + groupKey : groupKey;
            const isLeafDup = rawGroupPath !== scenePath && scenePath.endsWith('/' + groupKey) && findNodeByPath(dataTree, rawGroupPath) === null;
            const groupPath = isLeafDup ? scenePath : rawGroupPath;
            const worldPos = new THREE.Vector3(gData.xPos, gData.maxY, gData.zPos);
            const entry = createLabel(groupKey, groupPath, worldPos, true, null, groupPath);

            const ancestorParts = scenePath ? scenePath.split('/') : [];
            const ancestors = isLeafDup ? ancestorParts.slice(0, -1) : ancestorParts;
            for (let ai = 0; ai < ancestors.length; ai++) {{
                const aPath = ancestorParts.slice(0, ai + 1).join('/');
                const aName = ancestorParts[ai];
                const aEl = document.createElement('span');
                aEl.className = 'stack-label nav-ancestor-label';
                aEl.dataset.folderName = aPath;
                const aColor = aPath === highlightPath ? '#44f' : '#888';
                aEl.style.borderColor = aColor;
                aEl.innerHTML = `<span style="padding:2px 4px;cursor:pointer;color:${{aColor}}" data-path="${{aPath}}">${{aName}}</span>`;
                aEl.addEventListener('wheel', (e) => {{ e.stopPropagation(); renderer.domElement.dispatchEvent(new WheelEvent('wheel', e)); }}, {{ passive: false }});
                labelContainer.appendChild(aEl);
                aEl.addEventListener('mouseenter', () => {{
                    highlightTreePath(aPath);
                }});
                aEl.addEventListener('mouseleave', () => {{
                    if (currentFocusedParent) return;
                    clearTreeHighlight();
                }});
                aEl.querySelector('span[data-path]').onclick = (e) => {{
                    e.stopPropagation();
                    const clickedNode = findNodeByPath(dataTree, aPath);
                    if (clickedNode) renderContent(clickedNode);
                }};
                const ancestorOffset = (ancestors.length - ai) * 28;
                stackLabels.push({{ element: aEl, position: worldPos, xPos: gData.xPos, zPos: gData.zPos, folderName: aPath, isGroup: false, isAncestor: true, ancestorOffset, anchorGroup: groupPath, navParentPath: null, navVisible: false }});
            }}

            const posTree = buildPositionTree(gData.folders, groupPath);
            createChildLabels(posTree, groupPath, groupPath);
        }}
    }})();

    const bgLodPromise = loadingPromise.then(async () => {{
        const allSSArrays = new Set();
        folders.forEach(folder => {{
            grouped[folder].forEach(img => allSSArrays.add(JSON.stringify(img.ss)));
        }});
        const uniqueSSArrays = [...allSSArrays].map(s => JSON.parse(s));
        const waitFrame = () => new Promise(r => requestAnimationFrame(r));
        const lod0Paths = uniqueSSArrays.map(ss => ss[0]);
        for (const p of lod0Paths) {{
            await loadSpritesheet(p);
            await waitFrame();
        }}
    }});

    const sceneMaterials = new Set();
    loadingPromise.then(() => {{
        for (const data of Object.values(stackGroups))
            for (const mat of data.materials) sceneMaterials.add(mat);
    }});




    const stats = new Stats();
    stats.showPanel(0);
    container.appendChild(stats.dom);
    stats.dom.style.position = 'absolute';
    stats.dom.style.top = '30px';
    stats.dom.style.left = '5px';

    const resizeHandler = () => {{
        const newAspect = container.clientWidth / container.clientHeight;
        camera.left = frustumSize * newAspect / -2;
        camera.right = frustumSize * newAspect / 2;
        camera.updateProjectionMatrix();
        renderer.setSize(container.clientWidth, container.clientHeight);
    }};

    new ResizeObserver(() => resizeHandler()).observe(container);
            


    const sceneData = {{ scene, renderer, camera, controls, animationId: null, resizeHandler, labelContainer, countDiv, stackGroups, ssBg }};

    const frustum = new THREE.Frustum();
    const cameraViewProjectionMatrix = new THREE.Matrix4();

    let frameCount = 0;
    const animStartTime = Date.now();
    
    // OPTIMIZATION SETTINGS


    const MAX_LABELS =  spriteConfig.max_labels;
    const SCREEN_BUFFER = spriteConfig.screen_buffer;
    const _labelVec = new THREE.Vector3();
    const _worldBox = new THREE.Box3();

    function animate() {{
        stats.begin();
        sceneData.animationId = requestAnimationFrame(animate);
        controls.update();
        const rotationAngle = Date.now() * ROTATION_SPEED;
        scene.rotation.y = rotationAngle;
        
        if (frameCount % 5 === 0) {{
            camera.updateMatrixWorld();
            cameraViewProjectionMatrix.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
            frustum.setFromProjectionMatrix(cameraViewProjectionMatrix);
            for (const [folder, data] of Object.entries(stackGroups)) {{
                if (!data.localBox) {{
                    data.localBox = new THREE.Box3().setFromObject(data.group);
                    data.group.updateMatrixWorld();
                    data.localBox.applyMatrix4(data.group.matrixWorld.clone().invert());
                }}
                _worldBox.copy(data.localBox).applyMatrix4(data.group.matrixWorld);
                data.group.visible = frustum.intersectsBox(_worldBox);
            }}
        }}

        if (frameCount % 15 === 0) {{
            const viewHeight = (camera.top - camera.bottom) / camera.zoom;
            const targetLod = viewHeight < spriteConfig.lod_viewheight_thresholds[0] ? 0 : viewHeight < spriteConfig.lod_viewheight_thresholds[1] ? 1 : 2;
            for (const mat of sceneMaterials) {{
                if (!mat._ssArray) continue;
                const best = getBestLod(mat._ssArray, targetLod);
                if (best && mat.map !== best) {{ mat.map = best; mat.needsUpdate = true; }}
            }}
            for (const ai of animInstances) {{
                const best = getBestLod(ai.ssArray, targetLod);
                if (best && ai.mat.uniforms.map.value !== best) ai.mat.uniforms.map.value = best;
            }}
        }}

        const now = Date.now() - animStartTime;
        for (const ai of animInstances) ai.mat.uniforms.time.value = now;
        
        // OPTIMIZED LABEL UPDATE (Every frame, but efficient)
        if (labelsVisible) {{
            const cWidth = container.clientWidth;
            const cHeight = container.clientHeight;
            const cCenterX = cWidth / 2;
            const cCenterY = cHeight / 2;

            const labelCandidates = [];

            const cosR = Math.cos(rotationAngle);
            const sinR = Math.sin(rotationAngle);

            // 1. Calculate positions and filter roughly
            for (let i = 0; i < stackLabels.length; i++) {{
                const lbl = stackLabels[i];

                if (lbl.isAncestor) {{
                    if (activeGroupPath !== lbl.anchorGroup) {{
                        if (lbl.element.style.display !== 'none') lbl.element.style.display = 'none';
                        continue;
                    }}
                }} else if (!lbl.isGroup && !lbl.navVisible) {{
                    if (lbl.element.style.display !== 'none') lbl.element.style.display = 'none';
                    continue;
                }}
                
                // Manual World Rotation
                const rX = lbl.xPos * cosR + lbl.zPos * sinR;
                const rZ = -lbl.xPos * sinR + lbl.zPos * cosR;
                
                // Quick Depth Check (is it behind camera?)
                // Simple heuristic since we know camera is roughly at +Z
                // But camera rotates. So we use projection.
                
                const vec = _labelVec.set(rX, lbl.position.y, rZ);
                vec.project(camera);
                
                // vec.x and vec.y are now -1 to 1
                // Check bounds with buffer
                if (vec.z < 1 && vec.x >= -1.2 && vec.x <= 1.2 && vec.y >= -1.2 && vec.y <= 1.2) {{
                    const screenX = (vec.x * 0.5 + 0.5) * cWidth;
                    const screenY = (-(vec.y * 0.5) + 0.5) * cHeight;
                    
                    // Calculate distance to center of screen for priority
                    const dist = Math.abs(screenX - cCenterX) + Math.abs(screenY - cCenterY);
                    
                    labelCandidates.push({{
                        el: lbl.element,
                        x: screenX,
                        y: lbl.isAncestor ? screenY - lbl.ancestorOffset : screenY,
                        anchorY: screenY,
                        d: dist
                    }});
                }} else {{
                    if (lbl.element.style.display !== 'none') lbl.element.style.display = 'none';
                }}
            }}

            // 1b. Resolve overlapping labels with predictable vertical offset
            const OVERLAP_THRESH = 30;
            const LABEL_OFFSET_Y = 28;
            labelCandidates.sort((a, b) => a.el.dataset.folderName < b.el.dataset.folderName ? -1 : 1);
            for (let i = 0; i < labelCandidates.length; i++) {{
                for (let j = i + 1; j < labelCandidates.length; j++) {{
                    const dx = Math.abs(labelCandidates[i].x - labelCandidates[j].x);
                    const dy = Math.abs(labelCandidates[i].y - labelCandidates[j].y);
                    if (dx < OVERLAP_THRESH && dy < OVERLAP_THRESH) {{
                        labelCandidates[j].y = labelCandidates[i].y + LABEL_OFFSET_Y;
                    }}
                }}
            }}

            // 2. Sort by distance to center screen
            labelCandidates.sort((a, b) => a.d - b.d);

            // 3. Render top N, hide rest
            for (let i = 0; i < labelCandidates.length; i++) {{
                const cand = labelCandidates[i];
                if (i < MAX_LABELS) {{
                    if (cand.el.style.display !== 'block') cand.el.style.display = 'block';
                    cand.el.style.transform = `translate3d(${{cand.x.toFixed(1)}}px, ${{cand.y.toFixed(1)}}px, 0)`;
                    cand.el._screenX = cand.x;
                    cand.el._screenY = cand.y;
                    cand.el._screenVisible = true;
                }} else {{
                    if (cand.el.style.display !== 'none') cand.el.style.display = 'none';
                    cand.el._screenVisible = false;
                }}
            }}

            // 4. Wire system
            const WIRE = {{ r: 12, pad: 20, biasMin: 80, biasRange: 120, arrowSize: 6, arrowMinLen: 400 }};

            function roundCorner(d, a, c, b, maxR) {{
                const d1x = c.x-a.x, d1y = c.y-a.y, d2x = b.x-c.x, d2y = b.y-c.y;
                const l1 = Math.hypot(d1x, d1y), l2 = Math.hypot(d2x, d2y);
                const r = Math.min(maxR, l1*0.45, l2*0.45);
                if (r < 1) return d + ` L ${{c.x}} ${{c.y}}`;
                return d + ` L ${{c.x-d1x/l1*r}} ${{c.y-d1y/l1*r}} Q ${{c.x}} ${{c.y}}, ${{c.x+d2x/l2*r}} ${{c.y+d2y/l2*r}}`;
            }}

            function routeWire(a, b, dir) {{
                const {{ r: R, pad: P, biasMin: bm, biasRange: br }} = WIRE;
                const flip = dir === -1;
                const ra = flip ? {{ x: -a.x, y: a.y }} : a;
                const rb = flip ? {{ x: -b.x, y: b.y }} : b;
                let wp;
                if (rb.x > ra.x + P) {{
                    const mx = (ra.x + rb.x) / 2;
                    wp = [{{ x: mx, y: ra.y }}, {{ x: mx, y: rb.y }}];
                }} else {{
                    const bias = 0.5 + Math.min(Math.max((Math.abs(rb.y - ra.y) - bm) / br, 0), 1) * 0.35;
                    const rx = Math.max(ra.x, rb.x) + P, lx = Math.min(ra.x, rb.x) - P;
                    wp = [{{ x: rx, y: ra.y }}, {{ x: rx, y: ra.y+(rb.y-ra.y)*bias }}, {{ x: lx, y: ra.y+(rb.y-ra.y)*bias }}, {{ x: lx, y: rb.y }}];
                }}
                const pts = [ra, ...wp, rb];
                if (flip) pts.forEach(p => p.x = -p.x);
                let d = '';
                for (let i = 1; i < pts.length - 1; i++) d = roundCorner(d, pts[i-1], pts[i], pts[i+1], R);
                return {{ d: d + ` L ${{pts[pts.length-1].x}} ${{pts[pts.length-1].y}}`, pts }};
            }}

            const WIRE_STRAIGHT = spriteConfig.wire_straight;

            function wireD(a, b, dir) {{
                if (WIRE_STRAIGHT) {{
                    const pts = [a, b];
                    return {{ d: `M ${{a.x}} ${{a.y}} L ${{b.x}} ${{b.y}}`, pts }};
                }}
                const w = routeWire(a, b, dir); return {{ d: `M ${{a.x}} ${{a.y}}` + w.d, pts: w.pts }};
            }}
            function labelR(el) {{ return {{ x: (el._screenX||0) + (el.offsetWidth||60), y: (el._screenY||0) + (el.offsetHeight||18)/2 }}; }}
            function labelL(el) {{ return {{ x: el._screenX||0, y: (el._screenY||0) + (el.offsetHeight||18)/2 }}; }}

            function chainWires(els, out, inp, dir) {{
                let d = '';
                const arrows = [];
                for (let i = 0; i < els.length - 1; i++) {{
                    const w = wireD(out(els[i]), inp(els[i+1]), dir);
                    d += w.d;
                    arrows.push(w.pts);
                }}
                return {{ d, arrows }};
            }}

            // Arrow: midpoint of path, chevron pointing in travel direction
            let _arrows = [];
            function queueArrow(pts, dim) {{
                let total = 0;
                const lens = [];
                for (let i = 0; i < pts.length - 1; i++) {{ const l = Math.hypot(pts[i+1].x-pts[i].x, pts[i+1].y-pts[i].y); lens.push(l); total += l; }}
                if (total < WIRE.arrowMinLen) return;
                let target = total / 2, acc = 0;
                for (let i = 0; i < lens.length; i++) {{
                    if (acc + lens[i] >= target) {{
                        const t = (target - acc) / lens[i];
                        _arrows.push({{
                            x: pts[i].x + (pts[i+1].x-pts[i].x)*t,
                            y: pts[i].y + (pts[i+1].y-pts[i].y)*t,
                            a: Math.atan2(pts[i+1].y-pts[i].y, pts[i+1].x-pts[i].x) * 180 / Math.PI,
                            dim
                        }});
                        return;
                    }}
                    acc += lens[i];
                }}
            }}
            function flushArrows() {{
                wireArrowG.innerHTML = '';
                const s = WIRE.arrowSize, ns = 'http://www.w3.org/2000/svg';
                for (const ar of _arrows) {{
                    const g = document.createElementNS(ns, 'g');
                    g.setAttribute('transform', `translate(${{ar.x}},${{ar.y}}) rotate(${{ar.a}})`);
                    g.classList.add('wire-arrow');
                    if (ar.dim) g.classList.add('wire-arrow-dim');
                    for (const dy of [-s, s]) {{
                        const l = document.createElementNS(ns, 'line');
                        l.setAttribute('x1', -s); l.setAttribute('y1', dy); l.setAttribute('x2', 0); l.setAttribute('y2', 0);
                        g.appendChild(l);
                    }}
                    wireArrowG.appendChild(g);
                }}
            }}

            if (activeGroupPath) {{
                _arrows = [];
                const ancestorEls = stackLabels.filter(sl => sl.isAncestor && sl.anchorGroup === activeGroupPath && sl.element.style.display === 'block')
                    .sort((a, b) => b.ancestorOffset - a.ancestorOffset).map(sl => sl.element);
                const gSl = stackLabels.find(sl => sl.isGroup && sl.folderName === activeGroupPath);
                const gEl = gSl && gSl.element.style.display === 'block' ? gSl.element : null;
                const childEls = [];
                if (activeLabelPath && activeLabelPath !== activeGroupPath) {{
                    let cur = activeGroupPath;
                    for (const p of activeLabelPath.slice(activeGroupPath.length + 1).split('/')) {{
                        cur += '/' + p;
                        const cl = stackLabels.find(sl => !sl.isGroup && !sl.isAncestor && sl.folderName === cur && sl.element.style.display === 'block');
                        if (cl) childEls.push(cl.element);
                    }}
                }}

                const anc = gEl ? [...ancestorEls, gEl] : ancestorEls;
                const main = gEl ? [gEl, ...childEls] : childEls;

                // Determine flow direction from group label vs average child X
                let flowRight = true;
                if (gEl) {{
                    const visibleChildren = stackLabels.filter(sl => !sl.isGroup && !sl.isAncestor && sl.navVisible && sl.element.style.display === 'block' && sl.element._screenVisible);
                    if (visibleChildren.length > 0) {{
                        const avgChildX = visibleChildren.reduce((s, sl) => s + (sl.element._screenX || 0), 0) / visibleChildren.length;
                        const groupX = gEl._screenX || 0;
                        flowRight = avgChildX >= groupX;
                    }}
                }}
                const wireDir = flowRight ? 1 : -1;
                const labelOut = flowRight ? labelR : labelL;
                const labelIn = flowRight ? labelL : labelR;

                const ancW = chainWires(anc, labelOut, labelIn, wireDir);
                wirePathAncestor.setAttribute('d', ancW.d);
                ancW.arrows.forEach(pts => queueArrow(pts, false));

                const mainW = chainWires(main, labelOut, labelIn, wireDir);
                let mainD = mainW.d;
                mainW.arrows.forEach(pts => queueArrow(pts, false));
                const isLeaf = activeLabelPath && !stackLabels.some(sl => !sl.isGroup && !sl.isAncestor && sl.navParentPath === activeLabelPath);
                if (!sceneData.isTreeDriven() && !isLeaf && main.length) {{
                    const mw = wireD(labelOut(main[main.length - 1]), {{ x: mouseX, y: mouseY }}, wireDir);
                    mainD += mw.d;
                    queueArrow(mw.pts, false);
                }}
                wirePath.setAttribute('d', mainD);

                const onPath = new Set([...(gEl ? [gEl.dataset.folderName] : []), ...childEls.map(el => el.dataset.folderName)]);
                let dimD = '';
                for (const cel of main) {{
                    const cp = cel.dataset.folderName;
                    for (const sl of stackLabels) {{
                        if (sl.isGroup || sl.isAncestor || !sl.navVisible || sl.navParentPath !== cp || sl.element.style.display !== 'block' || onPath.has(sl.folderName)) continue;
                        const w = wireD(labelOut(cel), labelIn(sl.element), wireDir);
                        dimD += w.d;
                        queueArrow(w.pts, true);
                    }}
                }}
                wirePathDim.setAttribute('d', dimD);
                flushArrows();
                wireSvg.style.display = '';
            }} else {{
                wirePath.setAttribute('d', '');
                wirePathAncestor.setAttribute('d', '');
                wirePathDim.setAttribute('d', '');
                wireArrowG.innerHTML = '';
                wireSvg.style.display = 'none';
            }}
        }}

        if (ssBg) ssBg.update(camera, now);

        frameCount++;
        renderer.render(scene, camera);
        stats.end();
    }}
    animate();

    activeScenes.push(sceneData);
    window.addEventListener('resize', resizeHandler);
            
            const _ro = new ResizeObserver(() => {{
        if (container.clientWidth > 0 && container.clientHeight > 0) resizeHandler();
    }});
    _ro.observe(container);

    await loadingPromise;
    return sceneData;
}}
 
async function loadText(path) {{
    const res = await fetch(path);
    return await res.text();
}}

function findNodeByPath(node, targetPath) {{
    if (node.path === targetPath || node.name === targetPath) return node;
    for (const child of node.children) {{
        const found = findNodeByPath(child, targetPath);
        if (found) return found;
    }}
    return null;
}}

let currentNode = null;
let isInitialLoad = true;

function updateURL(node) {{
    const path = node.path || '';
    history.pushState(null, '', '?path=' + encodeURIComponent(path)); 
}}

const NAV_HISTORY_MAX = 20;
const navHistory = [];
let navHistoryIndex = -1;
let navIsTraversing = false;

function recordHistory(node) {{
    const entry = {{ path: node.path, name: node.name || node.path }};
    if (navHistoryIndex >= 0 && navHistory[navHistoryIndex] && navHistory[navHistoryIndex].path === entry.path) return;
    if (navHistoryIndex < navHistory.length - 1) navHistory.splice(navHistoryIndex + 1);
    navHistory.push(entry);
    if (navHistory.length > NAV_HISTORY_MAX) navHistory.shift();
    navHistoryIndex = navHistory.length - 1;
    updateNavButtons();
}}

function navigateHistory(dir) {{
    const idx = navHistoryIndex + dir;
    if (idx < 0 || idx >= navHistory.length) return;
    navHistoryIndex = idx;
    updateNavButtons();
    const h = navHistory[navHistoryIndex];
    const node = findNodeByPath(dataTree, h.path);
    if (node) {{
        navIsTraversing = true;
        renderContent(node);
    }}
}}

function updateNavButtons() {{
    const back = document.getElementById('nav-back');
    const fwd = document.getElementById('nav-fwd');
    back.style.color = navHistoryIndex > 0 ? '#fff' : '#333';
    fwd.style.color = navHistoryIndex < navHistory.length - 1 ? '#fff' : '#333';
}}

async function renderContent(node, page) {{
    if (page === undefined) {{
        if (!isInitialLoad && (!currentNode || currentNode.path !== node.path)) {{
            if (navIsTraversing) {{ navIsTraversing = false; }}
            else {{
                if (navHistory.length === 0 && currentNode) recordHistory(currentNode);
                recordHistory(node);
            }}
        }}
        if (!isInitialLoad) updateURL(node);
        updateURL(node);
        currentNode = node;
        updateTreeState(window.innerWidth > 768);
        let ch = node.children.length > 0 ? node.children : [node];
        allFilteredChildren = ch.filter(child => (child.ai.length > 0 || child.at.length > 0) && !child.hid);
        currentPage = 0;
    }} else {{
        currentPage = page;
    }}

    const RANDOM_TEXTDIV_POSITION = spriteConfig.random_textdiv_position;
    const DIV_RATIO_HALF = spriteConfig.div_ratio_half;
    const SEED = spriteConfig.seed;

    for (const key in stackMaterialCache) {{
        if (stackMaterialCache[key]) stackMaterialCache[key].dispose();
        delete stackMaterialCache[key];
    }}
    
    activeScenes.forEach(disposeScene);
    activeScenes = [];
    const contentDiv = document.getElementById('content');
    contentDiv.innerHTML = '';

    const totalChildren = allFilteredChildren.length;
    const totalPages = Math.ceil(totalChildren / getPageSize());
    const start = currentPage * getPageSize();
    const children = allFilteredChildren.slice(start, start + getPageSize());
            
    const scenePromises = []; 
    
    const count = children.length;
    const isMobile = window.innerWidth <= 768;
    const t = children.filter(c => !c.ai.length && c.at.length), d = children.filter(c => c.ai.length && !c.at.length);
    const mvs = isMobile && count === 2 && (t.length === 2 || (t.length === 1 && d.length === 1));
    const asymmetric = !DIV_RATIO_HALF && !isMobile && count === 2 && t.length === 1 && d.length === 1;
    //let cols = mvs ? 1 : Math.ceil(Math.sqrt(count));
    let cols = mvs ? 1 : ((!isMobile && count <= 4) ? count : Math.ceil(Math.sqrt(count)));    //<----

    
    for (const child of children) {{
        const div = document.createElement('div');
        div.className = 'content-div';
        let childWidth;
        if (asymmetric) {{
            const isText = !child.ai.length && child.at.length;
            childWidth = isText ? 'calc(40% - 2px)' : 'calc(60% - 2px)';
        }} else {{
            childWidth = mvs ? '100%' : `calc(${{100/cols}}% - 2px)`;
        }}
        div.style.width = childWidth;
        div.style.height = mvs ? 'calc(50% - 2px)' : (count <= 2 ? '100%' : `calc(${{100/Math.ceil(count/cols)}}% - 2px)`); 
        const label = document.createElement('div');
        label.className = 'div-label';
        label.innerHTML = breadcrumbHtml(child.path, currentNode.path);
            
        label.style.pointerEvents = 'auto';
        label.querySelectorAll('span[data-path]').forEach(span => {{
            span.onclick = (e) => {{
                e.stopPropagation();
                const node = findNodeByPath(dataTree, span.dataset.path);
                if (node) renderContent(node);
            }};
        }});

        div.appendChild(label);

        if (child.ai.length > 0 && child.at.length > 0) {{
            div.style.display = 'flex';
            div.style.flexDirection = 'row';
            
            // 1. Create a Wrapper for Text + Nav (This stays fixed)
            const textWrapper = document.createElement('div');
            textWrapper.style.flex = '3';
            textWrapper.style.position = 'relative'; // Anchor for the Nav
            textWrapper.style.borderLeft = '1px solid #333';
            textWrapper.style.boxSizing = 'border-box';
            //textWrapper.style.border = '1px solid white';
            textWrapper.style.overflow = 'hidden';   // The wrapper does NOT scroll

            // 2. Create the Scrollable Text Area
            const textDiv = document.createElement('div');
            textDiv.className = 'text-content';
            textDiv.style.width = '100%';
            textDiv.style.height = '100%';
            textDiv.style.overflowY = 'auto'; // The text scrolls INSIDE here
            textDiv.style.position = 'absolute'; // Fill the wrapper
            textDiv.style.top = '0';
            textDiv.style.left = '0';

            let htmlParts = [];
            for (const path of child.at) {{
                htmlParts.push(await loadText(path));
            }}
            
textDiv.innerHTML = assembleHtmlParts(htmlParts, child.at);


const scriptTags = textDiv.querySelectorAll('script');
scriptTags.forEach(oldScript => {{
    const newScript = document.createElement('script');
    newScript.textContent = oldScript.textContent;
    oldScript.parentNode.replaceChild(newScript, oldScript);
}});

            // 4. Assemble
            textWrapper.appendChild(textDiv);

            // 5. Image Side
            const imgDiv = document.createElement('div');
            imgDiv.style.flex = '3';
            imgDiv.style.position = 'relative';
            imgDiv.style.borderLeft = '1px solid #333';
            imgDiv.style.boxSizing = 'border-box';
            imgDiv.style.overflow = 'hidden';

            if (RANDOM_TEXTDIV_POSITION) {{
                const seed = child.path.split('').reduce((a, c) => a + c.charCodeAt(0), 0) * SEED;
                const rand = seededRandom(seed);
                if (rand < 0.5) {{ div.appendChild(textWrapper); div.appendChild(imgDiv); }}
                else {{ div.appendChild(imgDiv); div.appendChild(textWrapper); }}
            }} else {{
                 div.appendChild(textWrapper); div.appendChild(imgDiv);
            }}
            
            div._sceneContainer = imgDiv;
            div._sceneChild = child;


        }} else if (child.ai.length > 0) {{
            div._sceneContainer = div;
            div._sceneChild = child;
       }} else if (child.at.length > 0) {{
            div.className = 'content-div';
            div.style.position = 'relative';
            div.style.overflow = 'hidden';
            
            const textDiv = document.createElement('div');
            textDiv.className = 'text-content';
            textDiv.style.width = '100%';
            textDiv.style.height = '100%';
            textDiv.style.overflowY = 'auto';
            textDiv.style.position = 'absolute';
            textDiv.style.top = '0';
            textDiv.style.left = '0';
            
            let htmlParts = [];
            for (const path of child.at) {{ htmlParts.push(await loadText(path)); }}
            textDiv.innerHTML = assembleHtmlParts(htmlParts, child.at);
            
            const scriptTags = textDiv.querySelectorAll('script');
            scriptTags.forEach(oldScript => {{
                const newScript = document.createElement('script');
                newScript.textContent = oldScript.textContent;
                oldScript.parentNode.replaceChild(newScript, oldScript);
            }});
            
            const labelHtml = breadcrumbHtml(child.path, currentNode.path);
            
            const labelDiv = document.createElement('div');
            labelDiv.className = 'div-label';
            labelDiv.innerHTML = labelHtml;
            labelDiv.querySelectorAll('span[data-path]').forEach(span => {{
                span.onclick = (e) => {{
                    e.stopPropagation();
                    const node = findNodeByPath(dataTree, span.dataset.path);
                    if (node) renderContent(node);
                }};
            }});
            
            
            div.appendChild(textDiv);
            div.appendChild(labelDiv);
        }}
        const parentPath = currentNode ? currentNode.path.split('/').slice(0, -1).join('/') : null;
        const hasParent = parentPath !== null && parentPath !== '';
        const canGoDeeper = child !== node;

        if (hasParent || canGoDeeper) {{
            const navBar = document.createElement('div');
            navBar.className = 'div-nav';
            if (hasParent) {{
                const backBtn = document.createElement('span');
                backBtn.className = 'div-nav-back';
                backBtn.textContent = '<';
                backBtn.onclick = (e) => {{
                    e.stopPropagation();
                    const parentNode = findNodeByPath(dataTree, parentPath);
                    if (parentNode) renderContent(parentNode);
                }};
                navBar.appendChild(backBtn);
            }}
            if (canGoDeeper) {{
                const fwdBtn = document.createElement('span');
                fwdBtn.className = 'div-nav-fwd';
                fwdBtn.textContent = '>';
                fwdBtn.onclick = (e) => {{
                    e.stopPropagation();
                    renderContent(child);
                }};
                navBar.appendChild(fwdBtn);
            }}
            div.appendChild(navBar);
        }}

        contentDiv.appendChild(div);
    }}

    if (count > 1 && count % cols !== 0) {{
            const remaining = cols - (count % cols) + 1;
            contentDiv.lastElementChild.style.width = `calc(${{100 * remaining / cols}}% - 2px)`;
        }}
            
    if (totalPages > 1) {{
        const pagBar = document.createElement('div');
        pagBar.style.cssText = 'position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);display:flex;gap:12px;align-items:center;z-index:9999;';
        for (let i = 0; i < totalPages; i++) {{
            const btn = document.createElement('span');
            btn.textContent = i + 1;
            btn.style.cssText = `color:${{i === currentPage ? '#ff3333' : '#ff333366'}};font-size:64px;font-family:monospace;font-weight:bold;cursor:pointer;padding:4px 12px;`;
            btn.onmouseenter = () => btn.style.background = 'black';
            btn.onmouseleave = () => btn.style.background = 'transparent';
            if (i !== currentPage) btn.onclick = () => renderContent(currentNode, i);
            pagBar.appendChild(btn);
        }}
        contentDiv.style.position = 'relative';
        contentDiv.appendChild(pagBar);
    }}

    await new Promise(resolve => requestAnimationFrame(resolve));
    const sceneDivs = Array.from(contentDiv.children).filter(d => d._sceneContainer);
    for (const div of sceneDivs) {{
        scenePromises.push(createThreeScene(div._sceneContainer, div._sceneChild.ai, div._sceneChild, currentNode.path));
    }}
    await Promise.all(scenePromises);
    return Promise.resolve();
}}
</script>
</body>
</html>''')









print("Generated spritesheets, data.json and index.html")
