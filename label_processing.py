from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def load_custom_font(size):
    requested_font = Path('fonts/UbuntuMono-Regular.ttf')
    if requested_font.exists():
        try:
            return ImageFont.truetype(str(requested_font), int(size))
        except IOError:
            print(f"Warning: Found {requested_font} but could not load it.")
            
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
            
    try:
        return ImageFont.load_default(size=int(size))
    except TypeError:
        return ImageFont.load_default()


def create_label_image(title, body_lines, color, output_path, img_resolution, font_size):
    try:
        img = Image.new('RGBA', (img_resolution, img_resolution), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        font_title = load_custom_font(int(font_size * 1.5))
        font_body = load_custom_font(int(font_size))
        
        def get_text_size(text, font):
            try:
                left, top, right, bottom = font.getbbox(text)
                return right - left, bottom - top
            except AttributeError:
                return font.getsize(text)

        scale_factor = img_resolution / 64.0
        #padding_left = int(10 * scale_factor)
        #padding_top = int(10 * scale_factor)
        padding_top = 1
        padding_left = 1
        
        line_spacing = 0 
        title_body_gap = int(2 * scale_factor) if (title and body_lines) else 0
        
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
