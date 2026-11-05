"""Assemble a compile harness from the REAL source text of the restyled composables (MainActivity.kt / PatchRemoval.kt),
with stubs only for Android-only types, and write it to OUTDIR (Harness.kt + Stubs.kt + HfSettings.kt)."""
import sys, re, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kt_edit as K
SRC, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
main = open(os.path.join(SRC, "MainActivity.kt"), encoding="utf-8").read()
patch = open(os.path.join(SRC, "PatchRemoval.kt"), encoding="utf-8").read()
thumbs = open(os.path.join(SRC, "ui/MediaThumbs.kt"), encoding="utf-8").read()

drop = ("android.content.Intent", "android.os.Bundle", "androidx.lifecycle.viewmodel.compose.viewModel",
        "androidx.compose.ui.platform.LocalContext")
imports = [l for l in main.splitlines() if l.startswith("import ") and not l.startswith("import androidx.activity")
           and not any(l.strip() == "import " + d for d in drop)]
palette = main[main.index("private val Navy = Color"):main.index("class MainActivity : ComponentActivity()")]
funcs = ["SpaceChipRow", "SplitPartsStrip", "ChromeBar", "MediaDropZone", "FaceSlotTile", "StepCard", "SubHeading", "EmptyNote",
         "DisclosureRow", "PresetButtons", "Dropdown", "ActionButton", "GhostButton"]
body = "\n\n".join(K.get_func(main, f) for f in funcs)
pr_vals = "\n".join(l for l in patch.splitlines() if re.match(r"private val Pr(Slate|Sub) = ", l))
belly = K.get_func(patch, "BellyOneTapCard")
root_check = '''
@Composable
fun RootCheck(content: @Composable () -> Unit) {
    MaterialTheme(colorScheme = PhoenixColors, typography = PhoenixType, shapes = PhoenixShapes) {
        Column(
            Modifier.fillMaxSize().background(Brush.verticalGradient(listOf(Color(0xFFFBFCFF), Page)))
        ) {
            Column(
                Modifier
                    .weight(1f)
                    .fillMaxWidth()
                    .verticalScroll(rememberScrollState())
                    .padding(horizontal = if (LocalConfiguration.current.orientation == Configuration.ORIENTATION_LANDSCAPE) 8.dp else 14.dp)
                    .padding(top = 6.dp, bottom = 16.dp),
                verticalArrangement = Arrangement.spacedBy(if (LocalConfiguration.current.orientation == Configuration.ORIENTATION_LANDSCAPE) 6.dp else 12.dp)
            ) { content() }
        }
    }
}
'''
header = "package com.swamitech.phoenix\n\n" + "\n".join(imports) + "\nimport androidx.compose.foundation.border\n\n"
open(os.path.join(OUT, "Harness.kt"), "w", encoding="utf-8").write(
    header + palette + "\n" + pr_vals + "\n\n" + body + "\n\n" + belly + "\n" + root_check)

# the real ThumbState (plain Kotlin) and HfSettings.kt (PhoenixOptions)
ts_start = thumbs.index("data class ThumbState(")
ts_end = thumbs.index("\n}\n", ts_start) + 3
stubs = '''package android.net
class Uri { val lastPathSegment: String? = null }
'''
open(os.path.join(OUT, "StubUri.kt"), "w").write(stubs)
open(os.path.join(OUT, "StubConfig.kt"), "w").write('''package android.content.res
class Configuration { var orientation: Int = 1; companion object { const val ORIENTATION_LANDSCAPE = 2 } }
''')
open(os.path.join(OUT, "StubPlatform.kt"), "w").write('''package androidx.compose.ui.platform
import androidx.compose.runtime.compositionLocalOf
val LocalConfiguration = compositionLocalOf { android.content.res.Configuration() }
''')
open(os.path.join(OUT, "StubThumbs.kt"), "w").write('''package com.swamitech.phoenix.ui
import androidx.compose.runtime.Composable
import androidx.compose.runtime.State
import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.graphics.ImageBitmap

''' + thumbs[ts_start:ts_end] + '''
@Composable
fun rememberThumb(uri: android.net.Uri?, isVideo: Boolean, framePercent: Int = 0, maxPx: Int = 640): State<ThumbState> = mutableStateOf(ThumbState())
''')
open(os.path.join(OUT, "StubVm.kt"), "w").write('''package com.swamitech.phoenix
class PhoenixViewModel {
    data class SpaceSlot(val label: String, val url: String)
    data class HistoryItem(val kind: String = "Video", val partCount: Int = 1, val groupId: String? = null, val status: String = "",
                           val partIndex: Int = 0, val progress: Float = 0f)
    var swapBlockReason: String? = null
    var activeTab: String = "Video"
    fun setTab(tab: String) { activeTab = tab }
    fun startProcessing() {}
    var farmPercent: Int = 0
    var progress: Float = 0f
    var running: Boolean = false
    var history: List<HistoryItem> = emptyList()
    var spaces: List<SpaceSlot> = emptyList()
    var selectedSpaceIndex: Int = 0
    fun isSpaceBusy(url: String): Boolean = false
    fun selectProcessingSpace(index: Int) {}
    var options: PhoenixOptions = PhoenixOptions()
    fun updateOptions(transform: (PhoenixOptions) -> PhoenixOptions) { options = transform(options) }
}
''')
open(os.path.join(OUT, "HfSettings.kt"), "w").write(open(os.path.join(SRC, "HfSettings.kt"), encoding="utf-8").read())
print("harness written:", len(funcs) + 1, "composables")
