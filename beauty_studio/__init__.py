"""
Swamitech Beauty Studio - HDR video editor with natural retouching.

    from beauty_studio import Settings, PRESETS, render_video, process_image

The submodules are importable on their own (`grade`, `retouch`, `reshape`,
`hair`) so the stack can be used a stage at a time from a notebook or another
pipeline, not only through the UI.
"""
from .settings import DEFAULT_PRESET, PRESETS, Settings  # noqa: F401

__version__ = "1.8.0"
__all__ = ["Settings", "PRESETS", "DEFAULT_PRESET", "render_video",
           "process_image", "FrameProcessor", "__version__"]


def __getattr__(name):
    # The pipeline pulls in OpenCV and MediaPipe, which is a second or two of
    # import time; keep `from beauty_studio import Settings` cheap for anyone
    # who only wants the presets.
    if name in ("render_video", "process_image", "FrameProcessor", "probe",
                "grab_frame", "capability_report"):
        from . import pipeline
        return getattr(pipeline, name)
    raise AttributeError(name)
