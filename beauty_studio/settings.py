"""
Every knob the editor exposes, in one dataclass, plus the presets.

Two rules hold this file together:

1. Sliders are 0..100 "percent of effect" in the UI and 0..1 here. The UI never
   passes raw kernel sizes or gamma values around.
2. Nothing is uncapped. `Settings.normalised()` clamps each effect to the
   strongest setting that still survives being looked at on a big screen. The
   brief for this tool is a natural, professional result; a slider that can be
   pushed to obviously-retouched is a slider that will be.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, fields, replace

# Upper bound applied to each normalised amount (1.0 = the slider's 100 really
# means 100). These are the "you can't make it look fake" guard rails; the
# numbers come from where each effect starts to read as a filter on a 27" panel.
CAPS = {
    "skin_smooth": 0.85,
    "skin_even": 0.80,
    "blemish": 0.90,
    "glow": 0.55,
    "eye_brighten": 0.65,
    "teeth_whiten": 0.70,
    "lip_enhance": 0.55,
    "face_slim": 0.60,
    "face_round": 0.85,
    "chin_shape": 0.55,
    "nose_slim": 0.45,
    "eye_enlarge": 0.35,
    "body_slim": 0.75,
    "body_fuller": 0.70,
    "waist_shape": 0.85,
    "curve_shape": 0.80,
    # Widening is capped lower than narrowing on purpose: an outward push
    # reads as a distortion several percent sooner than an inward one, because
    # it has to invent silhouette where the background used to be.
    "bust_shape": 0.70,
    "hip_shape": 0.75,
    "hair_detail": 0.80,
    "hair_shine": 0.65,
    "hair_volume": 0.50,
    "hair_frizz": 0.85,
    "hdr_strength": 0.90,
    "clarity": 0.70,
    "vibrance": 0.70,
    "bloom": 0.45,
    "sharpen": 0.60,
}


@dataclass
class Settings:
    # ---- HDR grade -------------------------------------------------------
    hdr_strength: float = 0.55      # local tone mapping: shadow lift + highlight recovery
    shadows: float = 0.0            # -1..1 extra shadow lift / crush
    highlights: float = 0.0         # -1..1 extra highlight recovery / push
    clarity: float = 0.35           # mid-frequency local contrast
    vibrance: float = 0.35          # saturation weighted toward the dull colours
    saturation: float = 0.0         # -1..1 flat saturation on top of vibrance
    warmth: float = 0.0             # -1..1 white balance, blue <-> amber
    tint: float = 0.0               # -1..1 white balance, green <-> magenta
    contrast: float = 0.20          # gentle S-curve
    bloom: float = 0.15             # highlight glow
    sharpen: float = 0.25           # final output sharpening
    protect_skin_colour: bool = True  # keep vibrance off skin so faces stay real

    # ---- skin ------------------------------------------------------------
    skin_smooth: float = 0.45       # frequency-separated smoothing
    texture: float = 0.65           # how much real pore detail is put back (1 = all)
    skin_even: float = 0.40         # evens blotchy colour, not brightness
    blemish: float = 0.50           # spot suppression
    glow: float = 0.25              # soft luminosity on skin
    eye_brighten: float = 0.35
    teeth_whiten: float = 0.30
    lip_enhance: float = 0.25
    under_eye: float = 0.35         # dark-circle reduction

    # ---- face shape ------------------------------------------------------
    face_slim: float = 0.25         # jaw + cheek narrowing
    face_round: float = 0.0         # jaw + cheek widening (the opposite)
    chin_shape: float = 0.15        # chin taper
    nose_slim: float = 0.10
    eye_enlarge: float = 0.10

    # ---- body shape ------------------------------------------------------
    body_slim: float = 0.20         # whole-silhouette narrowing
    body_fuller: float = 0.0        # whole-silhouette widening (the opposite)
    waist_shape: float = 0.25       # waist pinch
    curve_shape: float = 0.20       # hourglass: waist in, hips/bust out, together
    bust_shape: float = 0.0         # bust/chest width on its own
    hip_shape: float = 0.0          # hip/thigh width on its own
    posture: float = 0.0            # shoulder lift/straighten, subtle

    # ---- hair ------------------------------------------------------------
    hair_detail: float = 0.40       # strand definition
    hair_shine: float = 0.30        # gloss highlights
    hair_volume: float = 0.20       # fuller silhouette
    hair_frizz: float = 0.35        # flyaway suppression
    hair_richness: float = 0.30     # colour depth

    # ---- global ----------------------------------------------------------
    naturalness: float = 0.85       # 1.0 = full effect, lower = blend back toward source
    process_scale: int = 1080       # long-edge working resolution for the effect stack
    stabilise: bool = True          # temporal smoothing of masks/landmarks (video only)

    # ---- output ----------------------------------------------------------
    out_long_edge: int = 0          # 0 = keep source size
    quality: int = 18               # x264 CRF (lower = better)
    hdr10: bool = False             # experimental PQ/BT.2020 export

    def normalised(self) -> "Settings":
        """Clamp to 0..1 (or -1..1 for the bipolar knobs) and apply CAPS."""
        out = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if f.type == "bool" or isinstance(v, bool):
                out[f.name] = bool(v)
            elif isinstance(v, int) and f.name in ("process_scale", "out_long_edge", "quality"):
                out[f.name] = int(v)
            else:
                v = float(v)
                if f.name in ("shadows", "highlights", "saturation", "warmth", "tint", "posture"):
                    v = max(-1.0, min(1.0, v))
                else:
                    v = max(0.0, min(1.0, v))
                    v = min(v, CAPS.get(f.name, 1.0))
                out[f.name] = v
        return Settings(**out)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Settings":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def with_(self, **kw) -> "Settings":
        return replace(self, **kw)

    def touches_face(self) -> bool:
        return any(getattr(self, k) > 0 for k in (
            "skin_smooth", "skin_even", "blemish", "glow", "eye_brighten",
            "teeth_whiten", "lip_enhance", "under_eye", "face_slim",
            "face_round", "chin_shape", "nose_slim", "eye_enlarge"))

    def touches_body(self) -> bool:
        return any(getattr(self, k) != 0 for k in
                   ("body_slim", "body_fuller", "waist_shape", "curve_shape",
                    "bust_shape", "hip_shape", "posture"))

    def touches_hair(self) -> bool:
        return any(getattr(self, k) > 0 for k in
                   ("hair_detail", "hair_shine", "hair_volume", "hair_frizz", "hair_richness"))


# --------------------------------------------------------------------- presets

PRESETS: dict[str, Settings] = {
    # The default. Someone should be able to run this on their own footage and
    # have a colleague say the lighting looks good, not that the video looks
    # edited.
    "Natural": Settings(),

    "Natural+ (subtle)": Settings(
        hdr_strength=0.35, clarity=0.22, vibrance=0.22, contrast=0.14, bloom=0.08,
        sharpen=0.18, skin_smooth=0.28, texture=0.80, skin_even=0.25, blemish=0.40,
        glow=0.15, eye_brighten=0.22, teeth_whiten=0.18, lip_enhance=0.12,
        under_eye=0.22, face_slim=0.12, chin_shape=0.08, nose_slim=0.0,
        eye_enlarge=0.0, body_slim=0.10, waist_shape=0.12, curve_shape=0.08,
        hair_detail=0.28, hair_shine=0.18, hair_volume=0.10, hair_frizz=0.25,
        hair_richness=0.18, naturalness=0.90),

    # Interview / talking-head / corporate footage: clean skin, no shape work.
    "Professional Portrait": Settings(
        hdr_strength=0.50, clarity=0.40, vibrance=0.28, contrast=0.22, bloom=0.10,
        sharpen=0.32, skin_smooth=0.45, texture=0.70, skin_even=0.45, blemish=0.60,
        glow=0.22, eye_brighten=0.40, teeth_whiten=0.35, lip_enhance=0.20,
        under_eye=0.45, face_slim=0.15, chin_shape=0.10, nose_slim=0.0,
        eye_enlarge=0.0, body_slim=0.10, waist_shape=0.10, curve_shape=0.0,
        hair_detail=0.45, hair_shine=0.30, hair_volume=0.15, hair_frizz=0.45,
        hair_richness=0.28, naturalness=0.90),

    "HDR Cinematic": Settings(
        hdr_strength=0.80, shadows=0.20, highlights=-0.15, clarity=0.55,
        vibrance=0.45, contrast=0.30, warmth=0.08, bloom=0.30, sharpen=0.35,
        skin_smooth=0.35, texture=0.75, skin_even=0.30, blemish=0.45, glow=0.20,
        eye_brighten=0.35, teeth_whiten=0.25, lip_enhance=0.20, under_eye=0.30,
        face_slim=0.15, chin_shape=0.10, body_slim=0.12, waist_shape=0.15,
        curve_shape=0.10, hair_detail=0.55, hair_shine=0.40, hair_volume=0.18,
        hair_frizz=0.40, hair_richness=0.40, naturalness=0.95),

    "Glam": Settings(
        hdr_strength=0.60, clarity=0.35, vibrance=0.50, contrast=0.26, bloom=0.32,
        sharpen=0.28, skin_smooth=0.65, texture=0.55, skin_even=0.60, blemish=0.75,
        glow=0.42, eye_brighten=0.55, teeth_whiten=0.50, lip_enhance=0.45,
        under_eye=0.55, face_slim=0.35, chin_shape=0.25, nose_slim=0.18,
        eye_enlarge=0.15, body_slim=0.28, waist_shape=0.40, curve_shape=0.35,
        hair_detail=0.55, hair_shine=0.50, hair_volume=0.35, hair_frizz=0.55,
        hair_richness=0.45, naturalness=0.85),

    # The hourglass, asked for by name: waist in, bust and hips out, with a
    # light grade and retouch so it does not look like a shape edit sitting on
    # ungraded footage.
    "Curvy": Settings(
        hdr_strength=0.50, clarity=0.35, vibrance=0.35, contrast=0.22, bloom=0.16,
        sharpen=0.28, skin_smooth=0.45, texture=0.65, skin_even=0.45, blemish=0.55,
        glow=0.28, eye_brighten=0.40, teeth_whiten=0.30, lip_enhance=0.28,
        under_eye=0.40, face_slim=0.22, chin_shape=0.15, nose_slim=0.05,
        eye_enlarge=0.05, body_slim=0.20, waist_shape=0.60, curve_shape=0.45,
        bust_shape=0.45, hip_shape=0.55, hair_detail=0.45, hair_shine=0.35,
        hair_volume=0.25, hair_frizz=0.45, hair_richness=0.35, naturalness=0.95),

    "Curvy (strong)": Settings(
        hdr_strength=0.55, clarity=0.38, vibrance=0.42, contrast=0.24, bloom=0.22,
        sharpen=0.30, skin_smooth=0.55, texture=0.60, skin_even=0.55, blemish=0.65,
        glow=0.35, eye_brighten=0.48, teeth_whiten=0.35, lip_enhance=0.35,
        under_eye=0.48, face_slim=0.30, chin_shape=0.20, nose_slim=0.10,
        eye_enlarge=0.10, body_slim=0.30, waist_shape=0.90, curve_shape=0.70,
        bust_shape=0.70, hip_shape=0.80, hair_detail=0.50, hair_shine=0.42,
        hair_volume=0.30, hair_frizz=0.50, hair_richness=0.40, naturalness=1.0),

    # Fuller figure, three strengths. The mirror image of the slimming
    # presets: the silhouette goes out instead of in, the face rounds rather
    # than tapers, and the waist is left alone - a fuller figure that keeps a
    # pinched waist reads as two edits arguing with each other.
    "Chubby (light)": Settings(
        hdr_strength=0.50, clarity=0.32, vibrance=0.32, contrast=0.20, bloom=0.14,
        sharpen=0.26, skin_smooth=0.42, texture=0.68, skin_even=0.40, blemish=0.50,
        glow=0.26, eye_brighten=0.35, teeth_whiten=0.28, lip_enhance=0.25,
        under_eye=0.38, face_slim=0.0, face_round=0.35, chin_shape=0.0,
        nose_slim=0.0, eye_enlarge=0.0, body_slim=0.0, body_fuller=0.30,
        waist_shape=0.0, curve_shape=0.0, bust_shape=0.10, hip_shape=0.15,
        hair_detail=0.42, hair_shine=0.32, hair_volume=0.22, hair_frizz=0.42,
        hair_richness=0.32, naturalness=0.95),

    "Chubby (medium)": Settings(
        hdr_strength=0.50, clarity=0.32, vibrance=0.32, contrast=0.20, bloom=0.14,
        sharpen=0.26, skin_smooth=0.45, texture=0.66, skin_even=0.42, blemish=0.52,
        glow=0.28, eye_brighten=0.36, teeth_whiten=0.28, lip_enhance=0.26,
        under_eye=0.40, face_slim=0.0, face_round=0.60, chin_shape=0.0,
        nose_slim=0.0, eye_enlarge=0.0, body_slim=0.0, body_fuller=0.55,
        waist_shape=0.0, curve_shape=0.0, bust_shape=0.20, hip_shape=0.28,
        hair_detail=0.42, hair_shine=0.32, hair_volume=0.22, hair_frizz=0.42,
        hair_richness=0.32, naturalness=0.95),

    "Chubby (heavy)": Settings(
        hdr_strength=0.50, clarity=0.30, vibrance=0.32, contrast=0.20, bloom=0.14,
        sharpen=0.24, skin_smooth=0.48, texture=0.64, skin_even=0.45, blemish=0.55,
        glow=0.30, eye_brighten=0.36, teeth_whiten=0.28, lip_enhance=0.26,
        under_eye=0.42, face_slim=0.0, face_round=0.90, chin_shape=0.0,
        nose_slim=0.0, eye_enlarge=0.0, body_slim=0.0, body_fuller=0.85,
        waist_shape=0.0, curve_shape=0.0, bust_shape=0.30, hip_shape=0.42,
        hair_detail=0.42, hair_shine=0.32, hair_volume=0.22, hair_frizz=0.42,
        hair_richness=0.32, naturalness=1.0),

    # Shape work only - for when the grade is already done elsewhere.
    "Shape Only": Settings(
        hdr_strength=0.0, clarity=0.0, vibrance=0.0, contrast=0.0, bloom=0.0,
        sharpen=0.0, skin_smooth=0.0, texture=1.0, skin_even=0.0, blemish=0.0,
        glow=0.0, eye_brighten=0.0, teeth_whiten=0.0, lip_enhance=0.0,
        under_eye=0.0, face_slim=0.30, chin_shape=0.20, nose_slim=0.12,
        eye_enlarge=0.10, body_slim=0.25, waist_shape=0.30, curve_shape=0.25,
        bust_shape=0.20, hip_shape=0.25,
        hair_detail=0.0, hair_shine=0.0, hair_volume=0.0, hair_frizz=0.0,
        hair_richness=0.0, naturalness=1.0),

    # Grade only - useful for landscape/product footage with no person in it.
    "HDR Only (no retouch)": Settings(
        hdr_strength=0.75, shadows=0.15, highlights=-0.10, clarity=0.50,
        vibrance=0.45, contrast=0.28, bloom=0.22, sharpen=0.35,
        skin_smooth=0.0, texture=1.0, skin_even=0.0, blemish=0.0, glow=0.0,
        eye_brighten=0.0, teeth_whiten=0.0, lip_enhance=0.0, under_eye=0.0,
        face_slim=0.0, chin_shape=0.0, nose_slim=0.0, eye_enlarge=0.0,
        body_slim=0.0, waist_shape=0.0, curve_shape=0.0,
        hair_detail=0.0, hair_shine=0.0, hair_volume=0.0, hair_frizz=0.0,
        hair_richness=0.0, naturalness=1.0),
}

DEFAULT_PRESET = "Natural"

# ----------------------------------------------------------- combining presets
#
# Presets can be stacked - "Chubby (medium) + HDR Cinematic" - and that only
# works if each one knows which part of the picture it is about. A preset
# contributes the fields of the domains it owns and leaves the rest alone, so
# stacking a grade onto a shape preset gives you both instead of whichever was
# applied last.

DOMAIN_FIELDS: dict[str, tuple[str, ...]] = {
    "grade": ("hdr_strength", "shadows", "highlights", "clarity", "vibrance",
              "saturation", "warmth", "tint", "contrast", "bloom", "sharpen",
              "protect_skin_colour"),
    "skin": ("skin_smooth", "texture", "skin_even", "blemish", "glow",
             "eye_brighten", "teeth_whiten", "lip_enhance", "under_eye"),
    "face": ("face_slim", "face_round", "chin_shape", "nose_slim", "eye_enlarge"),
    "body": ("body_slim", "body_fuller", "waist_shape", "curve_shape",
             "bust_shape", "hip_shape", "posture"),
    "hair": ("hair_detail", "hair_shine", "hair_volume", "hair_frizz",
             "hair_richness"),
    "finish": ("naturalness",),
}

ALL_DOMAINS = tuple(DOMAIN_FIELDS)

# What each preset is *about*. A full look owns everything; a specialist owns
# only its own part, so stacking it changes only that part.
PRESET_DOMAINS: dict[str, tuple[str, ...]] = {
    "Natural": ALL_DOMAINS,
    "Natural+ (subtle)": ALL_DOMAINS,
    "Professional Portrait": ALL_DOMAINS,
    "Glam": ALL_DOMAINS,
    "HDR Cinematic": ("grade",),
    "HDR Only (no retouch)": ("grade",),
    "Curvy": ("body",),
    "Curvy (strong)": ("body",),
    "Chubby (light)": ("body", "face"),
    "Chubby (medium)": ("body", "face"),
    "Chubby (heavy)": ("body", "face"),
    "Shape Only": ("face", "body"),
}

MAX_STACK = 3


def combine_presets(names) -> Settings:
    """
    Stack up to three presets into one Settings.

    The broadest preset is applied first and the most specific last, whatever
    order they were picked in, and each one only writes the domains it owns.
    So "Chubby (medium) + HDR Cinematic" keeps the chubby shaping and takes
    the cinematic grade either way round, and adding a full look like
    Professional Portrait on top brings its skin and hair without quietly
    undoing the shaping you asked for first.
    """
    if isinstance(names, str):
        names = [names]
    names = [n for n in (names or []) if n in PRESETS][:MAX_STACK]
    if not names:
        return PRESETS[DEFAULT_PRESET]

    # Broad to narrow. Without this, a full look picked last silently
    # overwrites the specialist picked first - the user asks for curvy plus a
    # portrait look and gets the portrait's default body back.
    ordered = sorted(names, key=lambda n: -len(PRESET_DOMAINS.get(n, ALL_DOMAINS)))

    out = PRESETS[ordered[0]]
    for name in ordered[1:]:
        preset = PRESETS[name]
        fields: set[str] = set()
        for domain in PRESET_DOMAINS.get(name, ALL_DOMAINS):
            fields.update(DOMAIN_FIELDS[domain])
        out = out.with_(**{f: getattr(preset, f) for f in fields})
    return out


def stack_label(names) -> str:
    if isinstance(names, str):
        names = [names]
    names = [n for n in (names or []) if n in PRESETS][:MAX_STACK]
    return " + ".join(names) if names else DEFAULT_PRESET


# Amounts that describe work done ON A PERSON, as opposed to the grade. These
# are the ones scaled by `naturalness`, and by a tracker's confidence when a
# face is being acquired or coasted through a dropout.
PERSON_AMOUNTS = (
    "skin_smooth", "skin_even", "blemish", "glow", "eye_brighten",
    "teeth_whiten", "lip_enhance", "under_eye", "face_slim", "chin_shape",
    "nose_slim", "eye_enlarge", "face_round", "body_slim", "body_fuller",
    "waist_shape", "curve_shape", "bust_shape", "hip_shape",
    "hair_detail", "hair_shine", "hair_volume", "hair_frizz", "hair_richness",
)


def scale_person_amounts(s: Settings, k: float, relax_texture: bool = False) -> Settings:
    """Scale every person-directed amount by `k` (0 = source, 1 = as set).

    Used for two things that want exactly the same behaviour: the naturalness
    control, and ramping an effect in and out as a face is found or lost. In
    both cases scaling the amounts is right and cross-fading the finished
    frame is wrong - a cross-fade of a reshaped frame against the original
    leaves a double edge on every contour that moved.
    """
    k = float(max(0.0, min(1.0, k)))
    if k >= 0.999:
        return s
    kw = {name: getattr(s, name) * k for name in PERSON_AMOUNTS}
    kw["posture"] = s.posture * k
    if relax_texture:
        # Less retouching should also mean more of the real texture survives.
        kw["texture"] = s.texture + (1.0 - s.texture) * (1.0 - k)
    return s.with_(**kw)
