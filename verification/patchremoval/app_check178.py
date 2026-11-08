"""v11.2.178: build the Space's real app.py Blocks (Gradio 4.44.1) with the heavy model packages stubbed, list the API names, launch
it locally and call phoenix_patch_preview exactly as the app does (gradio_client). usage: app_check.py PACKDIR TESTDIR"""
import sys, types, importlib.abc, importlib.machinery, unittest.mock as um, os, json
PACK, TD = sys.argv[1], sys.argv[2]
HEAVY = ("insightface", "onnxruntime", "torch", "torchvision", "mediapipe", "basicsr", "gfpgan", "realesrgan", "facexlib", "spaces",
         "rembg", "skimage", "scipy", "timm", "kornia", "lama_cleaner", "simple_lama_inpainting")
class _Stub(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in HEAVY:
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
    def create_module(self, spec):
        m = um.MagicMock(name=spec.name); m.__path__ = []; m.__spec__ = spec; return m
    def exec_module(self, module):
        pass
sys.meta_path.insert(0, _Stub())
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
sys.path.insert(0, PACK); os.chdir(PACK)
import app
demo = next(v for v in vars(app).values() if v.__class__.__name__ == "Blocks")
names = sorted(d.api_name for d in demo.fns.values() if getattr(d, "api_name", None))
print("API names:", names)
assert "phoenix_patch_preview" in names
demo.queue().launch(server_name="127.0.0.1", server_port=7871, prevent_thread_lock=True, show_api=True)
from gradio_client import Client, handle_file
c = Client("http://127.0.0.1:7871/", verbose=False)
res = c.predict(handle_file(TD + "/frame.jpg"), handle_file(TD + "/mask.png"), api_name="/phoenix_patch_preview")
print("result:", res)
info = json.loads(res[1]); assert info.get("ok") and os.path.isfile(res[0]), res
print("PASS phoenix_patch_preview through the real Gradio app: a cleaned frame file and JSON ok")
demo.close()
