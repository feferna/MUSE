# toggle_renderer.py
from typing import Dict, Tuple, TYPE_CHECKING
import math


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _color_scheme(idx: int) -> Tuple[str, str, str]:
    schemes = [
        ("#3b82f6", "#e5e7eb", "#ffffff"),  # modern
        ("#10b981", "#d1fae5", "#ffffff"),  # nature
        ("#f59e0b", "#fef3c7", "#ffffff"),  # sunset
        ("#06b6d4", "#cffafe", "#ffffff"),  # ocean
        ("#1f2937", "#4b5563", "#ffffff"),  # dark
        ("#8b5cf6", "#c4b5fd", "#ffffff"),  # neon
        ("#374151", "#d1d5db", "#ffffff"),  # enterprise
    ]
    idx = max(0, min(idx, len(schemes)-1))
    return schemes[idx]


def _visual_style(idx: int) -> Tuple[str, str]:
    # returns (visual, track)
    styles = [
        ("flat",    "solid"),
        ("shadow",  "gradient"),
        ("glow",    "glass"),
        ("outline", "outlined"),
        ("elevated", "textured"),
    ]
    idx = max(0, min(idx, len(styles)-1))
    return styles[idx]


def _thumb_shape(idx: int) -> str:
    shapes = [
        "circle",    # 0
        "square",    # 1
        "diamond",   # 2
        "hexagon",   # 3
        "star",      # 4
        "triangle",  # 5
        "teardrop",  # 6
        "bean",      # 7
    ]
    idx = max(0, min(idx, len(shapes)-1))
    return shapes[idx]


def render_toggle_svg(params: Dict, *, is_on: bool = False, size: int = 140, margin: int = 16) -> str:

    def clamp01(x: float) -> float:
        return max(0.0, min(1.0, float(x)))

    def color_scheme(idx: int):
        schemes = [
            ("#3b82f6", "#e5e7eb", "#ffffff"),  # modern
            ("#10b981", "#d1fae5", "#ffffff"),  # nature
            ("#f59e0b", "#fef3c7", "#ffffff"),  # sunset
            ("#06b6d4", "#cffafe", "#ffffff"),  # ocean
            ("#1f2937", "#4b5563", "#ffffff"),  # dark
            ("#8b5cf6", "#c4b5fd", "#ffffff"),  # neon
            ("#374151", "#d1d5db", "#ffffff"),  # enterprise
        ]
        idx = max(0, min(idx, len(schemes)-1))
        return schemes[idx]

    def visual_style(idx: int):
        styles = [
            ("flat",    "solid"),
            ("shadow",  "gradient"),
            ("glow",    "glass"),
            ("outline", "outlined"),
            ("elevated", "textured"),
        ]
        idx = max(0, min(idx, len(styles)-1))
        return styles[idx]

    def get_thumb_shape_path(shape: str, x: float, y: float, size: float, rx: float, ry: float) -> str:
        """Generate SVG path for different thumb shapes"""
        cx, cy = x + size/2, y + size/2
        r = size/2

        if shape == "circle":
            return f'<circle cx="{cx}" cy="{cy}" r="{r}"/>'
        elif shape == "square":
            return f'<rect x="{x}" y="{y}" width="{size}" height="{size}" rx="{rx}" ry="{ry}"/>'
        elif shape == "diamond":
            return f'<polygon points="{cx},{y} {x+size},{cy} {cx},{y+size} {x},{cy}"/>'
        elif shape == "hexagon":
            points = []
            for i in range(6):
                angle = i * math.pi / 3
                px = cx + r * math.cos(angle)
                py = cy + r * math.sin(angle)
                points.append(f"{px:.2f},{py:.2f}")
            return f'<polygon points="{" ".join(points)}"/>'
        elif shape == "star":
            points = []
            for i in range(10):  # 5 points, 5 inner points
                angle = i * math.pi / 5
                radius = r if i % 2 == 0 else r * 0.4
                px = cx + radius * math.cos(angle - math.pi/2)
                py = cy + radius * math.sin(angle - math.pi/2)
                points.append(f"{px:.2f},{py:.2f}")
            return f'<polygon points="{" ".join(points)}"/>'
        elif shape == "triangle":
            p1 = f"{cx},{y}"
            p2 = f"{x+size},{y+size}"
            p3 = f"{x},{y+size}"
            return f'<polygon points="{p1} {p2} {p3}"/>'
        elif shape == "teardrop":
            # The frontend uses CSS border-radius: '50% 50% 50% 15%' for teardrop.
            # We reproduce that by drawing a rounded rectangle with per-corner
            # radii (top-left, top-right, bottom-right, bottom-left) expressed
            # as percentages of the thumb size.
            w = size
            h = size
            # percentages from frontend: TL=50%, TR=50%, BR=50%, BL=15%
            pct_tl, pct_tr, pct_br, pct_bl = 0.5, 0.5, 0.5, 0.15
            rtl = max(0.0, min(w/2.0, pct_tl * w))
            rtr = max(0.0, min(w/2.0, pct_tr * w))
            rbr = max(0.0, min(w/2.0, pct_br * w))
            rbl = max(0.0, min(w/2.0, pct_bl * w))

            # Ensure radii don't exceed half dimensions
            rtl = min(rtl, w/2, h/2)
            rtr = min(rtr, w/2, h/2)
            rbr = min(rbr, w/2, h/2)
            rbl = min(rbl, w/2, h/2)

            x0 = x
            y0 = y
            x1 = x + w
            y1 = y + h

            # Build SVG path using arcs for rounded corners
            # Start at top-left corner (after radius)
            path = []
            path.append(f'M {x0 + rtl:.2f},{y0:.2f}')
            # top edge -> to top-right corner
            path.append(f'L {x1 - rtr:.2f},{y0:.2f}')
            # top-right arc
            path.append(f'A {rtr:.2f},{rtr:.2f} 0 0 1 {x1:.2f},{y0 + rtr:.2f}')
            # right edge -> to bottom-right corner
            path.append(f'L {x1:.2f},{y1 - rbr:.2f}')
            # bottom-right arc
            path.append(f'A {rbr:.2f},{rbr:.2f} 0 0 1 {x1 - rbr:.2f},{y1:.2f}')
            # bottom edge -> to bottom-left corner
            path.append(f'L {x0 + rbl:.2f},{y1:.2f}')
            # bottom-left arc
            path.append(f'A {rbl:.2f},{rbl:.2f} 0 0 1 {x0:.2f},{y1 - rbl:.2f}')
            # left edge -> to top-left corner
            path.append(f'L {x0:.2f},{y0 + rtl:.2f}')
            # top-left arc
            path.append(f'A {rtl:.2f},{rtl:.2f} 0 0 1 {x0 + rtl:.2f},{y0:.2f}')
            path.append('Z')

            return '<path d="' + ' '.join(path) + '"/>'
        elif shape == "bean":
            # Bean shape using curves
            return f'<path d="M {x+r},{y} Q {x+size},{y} {x+size},{y+r} Q {x+size},{y+size} {x+r},{y+size} Q {x},{y+size} {x},{y+r} Q {x},{y} {x+r},{y} Z"/>'
        else:
            # Default to rectangle
            return f'<rect x="{x}" y="{y}" width="{size}" height="{size}" rx="{rx}" ry="{ry}"/>'

    # Thumb icon and text label rendering is handled in frontend; no-op here.
    # Extract parameters with new defaults
    color_idx = int(params.get("colorScheme", 0))
    v_idx = int(params.get("visualStyle", 0))
    thumb_ratio = float(params.get("thumbRatio", 0.8))
    border_radius = float(params.get("borderRadius", 16))
    thumb_shape_idx = int(params.get("thumbShape", 0))

    thumb_shape = _thumb_shape(thumb_shape_idx)
    active, inactive, thumb_color = color_scheme(color_idx)
    visual, track = visual_style(v_idx)

    width = size
    height = int(size * 0.5)
    thumb_ratio = clamp01(thumb_ratio)
    thumb_size = height * thumb_ratio
    padding = (height - thumb_size) / 2.0
    max_translate = width - thumb_size - 2*padding
    x_thumb = padding + (max_translate if is_on else 0.0)
    y_thumb = padding

    # Clamp radii to CSS-like behavior
    rx_track = min(border_radius, width/2, height/2)
    ry_track = rx_track
    rx_thumb = min(border_radius*0.8, thumb_size/2)
    ry_thumb = rx_thumb

    track_color = active if is_on else inactive

    # --- defs (no 8-digit hex; use explicit opacities) ---
    defs_parts = []
    defs_parts.append("""
      <filter id="shadow" x="-50%" y="-50%" width="200%" height="200%">
        <feDropShadow dx="0" dy="2" stdDeviation="2" flood-opacity="0.15"/>
      </filter>
    """)
    defs_parts.append("""
      <filter id="glow" x="-80%" y="-80%" width="260%" height="260%">
        <feGaussianBlur stdDeviation="6" result="blur"/>
        <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
      </filter>
    """)
    defs_parts.append(f"""
      <pattern id="dots" patternUnits="userSpaceOnUse" width="8" height="8">
        <rect width="8" height="8" fill="{track_color}"/>
        <circle cx="2" cy="2" r="1" fill="white" fill-opacity="0.2"/>
      </pattern>
    """)

    # Track fill by style (match React):
    # - 'glass': translucent fill + light border (no backdrop blur in static PNG)
    # - 'outlined': no fill, 2px stroke
    # - 'textured': pattern
    # - default: solid
    if track == "outlined":
        track_fill = 'none'
        track_attrs = f'stroke="{track_color}" stroke-width="2"'
    elif track == "textured":
        track_fill = 'url(#dots)'
        track_attrs = ''
    elif track == "glass":
        track_fill = track_color
        track_attrs = 'fill-opacity="0.25" stroke="white" stroke-opacity="0.2" stroke-width="1"'
    else:
        track_fill = track_color
        track_attrs = ''

    # Thumb extras by visual (match React behavior):
    # - 'glow': glow only when ON; otherwise subtle shadow
    # - 'shadow': subtle shadow
    # - 'outline': white fill + state-colored stroke
    # - 'elevated': shadow + tiny rise when ON
    thumb_attrs = []

    if visual == "glow":
        thumb_filter = 'url(#glow)' if is_on else 'url(#shadow)'
        thumb_attrs.append(f'filter="{thumb_filter}"')

    elif visual == "shadow":
        thumb_attrs.append('filter="url(#shadow)"')

    elif visual == "outline":
        stroke_col = active if is_on else "#d1d5db"
        thumb_attrs.append(f'stroke="{stroke_col}" stroke-width="2"')
        thumb_color = "white"  # Override thumb color for outline style

    elif visual == "elevated":
        thumb_attrs.append('filter="url(#shadow)"')
        # static PNG can't show the tiny translateY; ignore

    thumb_attr_str = " ".join(thumb_attrs)

    # Generate thumb shape path
    thumb_shape_svg = get_thumb_shape_path(
        thumb_shape,
        margin + x_thumb,
        margin + y_thumb,
        thumb_size,
        rx_thumb,
        ry_thumb
    )

    thumb_icon_svg = ""
    text_labels_svg = ""

    svg_w = width + margin*2
    svg_h = height + margin*2

    # Replace thumb_shape_svg with proper fill and attributes
    if thumb_shape_svg.startswith('<rect'):
        # For rect-based shapes, we can add fill and attributes directly
        thumb_shape_svg = thumb_shape_svg.replace(
            '/>', f' fill="{thumb_color}" {thumb_attr_str}/>')
    elif '<circle' in thumb_shape_svg or '<polygon' in thumb_shape_svg or '<path' in thumb_shape_svg:
        # For other shapes, add fill and attributes before the closing />
        thumb_shape_svg = thumb_shape_svg.replace(
            '/>', f' fill="{thumb_color}" {thumb_attr_str}/>')

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w}" height="{svg_h}" viewBox="0 0 {svg_w} {svg_h}">
    <defs>{''.join(defs_parts)}</defs>
    <rect x="0" y="0" width="{svg_w}" height="{svg_h}" fill="#ffffff"/>
    <rect x="{margin}" y="{margin}" rx="{rx_track}" ry="{ry_track}"
        width="{width}" height="{height}" fill="{track_fill}" {track_attrs}/>
    {text_labels_svg}
    {thumb_shape_svg}
    {thumb_icon_svg}
    </svg>"""
    return svg


def save_toggle_png(params: Dict, out_png: str, *, is_on: bool = False, size: int = 140) -> str:
    """
    Renders the toggle to SVG and writes a PNG to out_png. Returns the path.
    """
    # Help static type checkers (and the editor) resolve the cairosvg symbol
    # without forcing a runtime import during static analysis.
    if TYPE_CHECKING:  # pragma: no cover
        import cairosvg  # type: ignore

    svg = render_toggle_svg(params, is_on=is_on, size=size)

    # Runtime import with a clear error if the dependency is not installed.
    try:
        import cairosvg  # type: ignore[import]
    except Exception as e:
        raise ImportError(
            "cairosvg is required to convert SVG to PNG. "
            "Install it in your environment: pip install cairosvg"
        ) from e

    cairosvg.svg2png(bytestring=svg.encode("utf-8"),
                     write_to=out_png, background_color="white")
    return out_png
