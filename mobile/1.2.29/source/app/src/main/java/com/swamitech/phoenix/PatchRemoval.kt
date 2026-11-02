package com.swamitech.phoenix

import android.content.Context
import androidx.activity.compose.BackHandler
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.Build
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.gestures.calculateCentroid
import androidx.compose.foundation.gestures.calculatePan
import androidx.compose.foundation.gestures.calculateZoom
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.clipToBounds
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.clipPath
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream

/*
 * Patch Removal (app 1.2.25 / 1.2.26; Space v11.2.96 / v11.2.97).
 *
 * The user picks a frame of the target video ON THE PHONE (the exact frame, not the nearest key frame), paints over a
 * mark with a finger (tattoo, mole, any dark spot, any shape), asks the Space for a preview of the frame with the mark
 * removed, and freezes it. 1.2.26: the same mark can be painted at SEVERAL POSITIONS (other frames where the person
 * has moved or turned; the Space uses the nearest one on every frame), zoom buttons and a magnifier under the finger.
 * 1.2.27: the editor is a full screen inside the app's own window (not a dialog window, whose insets are unreliable),
 * works in portrait and landscape, keeps its state through rotation (ViewModel) and keeps every button on screen.
 */

/** One painted position of a mark: the frame, the painted mask (white on black, same size), its time. */
data class PatchView(
    val id: String,
    val frameFile: File,
    val maskFile: File,
    val previewFile: File?,
    val timeMs: Long,
    val width: Int,
    val height: Int
)

/** One mark, painted at one or more positions. */
data class PatchSpec(val id: String, val views: List<PatchView>) {
    val timeMs: Long get() = views.firstOrNull()?.timeMs ?: 0L
}

/** A finger stroke in FRAME pixels (so it survives zoom / pan) with its brush radius in frame pixels. */
data class PaintStroke(val points: List<Offset>, val radius: Float)

/** The editor's working state. Kept in the ViewModel, so a rotation or a short trip out of the app loses nothing. */
class PatchSession(val video: Uri) {
    val markId: String = "m" + System.currentTimeMillis().toString(36)
    val positions = mutableStateListOf<PatchView>()       // positions already added for this mark
    val strokes = mutableStateListOf<PaintStroke>()       // the position being painted
    var frame by mutableStateOf<Bitmap?>(null)
    var frameMs by mutableLongStateOf(-1L)               // the frame shown (and painted on)
    var posMs by mutableFloatStateOf(0f)                 // the slider
    var durationMs by mutableLongStateOf(0L)
    var stepMs by mutableLongStateOf(33L)                // one frame
    var preview by mutableStateOf<Bitmap?>(null)
    var previewView by mutableStateOf<PatchView?>(null)  // the saved position the preview belongs to
    var showPreview by mutableStateOf(false)
    var brush by mutableIntStateOf(1)
    var zoom by mutableFloatStateOf(1f)
    var pan by mutableStateOf(Offset.Zero)
    var canvas by mutableStateOf(IntSize.Zero)
    var loading by mutableStateOf(false)

    fun clearPainting() {
        strokes.clear(); preview = null; previewView = null; showPreview = false
    }

    fun fit(): Pair<Float, Offset> {
        val b = frame ?: return 1f to Offset.Zero
        val s = minOf(canvas.width / b.width.toFloat(), canvas.height / b.height.toFloat()).coerceAtLeast(1e-3f)
        return s to Offset((canvas.width - b.width * s) / 2f, (canvas.height - b.height * s) / 2f)
    }

    /** Zoom by k around the middle of the picture (the + / - buttons). */
    fun zoomBy(k: Float) {
        val nz = (zoom * k).coerceIn(1f, 12f)
        if (nz <= 1.0001f) { zoom = 1f; pan = Offset.Zero; return }
        val (_, o) = fit()
        val c = Offset(canvas.width / 2f, canvas.height / 2f)
        pan = (c - o) - (c - o - pan) * (nz / zoom)
        zoom = nz
    }

    fun resetZoom() { zoom = 1f; pan = Offset.Zero }

    fun brushPx(): Float {
        val b = frame ?: return 8f
        return maxOf(b.width, b.height) * when (brush) { 0 -> 0.006f; 1 -> 0.013f; else -> 0.026f }
    }
}

object PatchFrames {
    const val MAX_PX = 1280            // the frame sent to the Space (long side)
    const val MAX_PATCHES = 10         // marks per video (1.2.28: was 4; needs Space v11.2.100)
    const val MAX_VIEWS = 20           // painted positions in all (1.2.28: was 12)
    const val MAX_VIEWS_PER_MARK = 10  // painted positions of ONE mark (1.2.28: was 6)

    fun durationMs(context: Context, uri: Uri): Long {
        val r = MediaMetadataRetriever()
        return try {
            r.setDataSource(context, uri)
            r.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: 0L
        } catch (_: Throwable) {
            0L
        } finally {
            runCatching { r.release() }
        }
    }

    /** One frame's duration in ms (frame count / duration when the file says it, else 30 fps). */
    fun frameStepMs(context: Context, uri: Uri): Long {
        val r = MediaMetadataRetriever()
        return try {
            r.setDataSource(context, uri)
            val d = r.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)?.toLongOrNull() ?: 0L
            val n = if (Build.VERSION.SDK_INT >= 28)
                r.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT)?.toLongOrNull() ?: 0L else 0L
            if (d > 0 && n > 0) (d / n).coerceIn(8L, 100L) else 33L
        } catch (_: Throwable) {
            33L
        } finally {
            runCatching { r.release() }
        }
    }

    /** The exact frame at [timeMs] (OPTION_CLOSEST, not the nearest key frame), at most [maxPx] on its long side. */
    fun frameAt(context: Context, uri: Uri, timeMs: Long, maxPx: Int = MAX_PX): Bitmap? {
        val r = MediaMetadataRetriever()
        try {
            r.setDataSource(context, uri)
            val w = r.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)?.toIntOrNull() ?: 0
            val h = r.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)?.toIntOrNull() ?: 0
            val rot = r.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION)?.toIntOrNull() ?: 0
            val (dw, dh) = if (rot == 90 || rot == 270) h to w else w to h
            val us = (timeMs.coerceAtLeast(0L)) * 1000L
            val bmp = if (Build.VERSION.SDK_INT >= 27 && dw > 0 && dh > 0) {
                val s = maxOf(dw, dh).toFloat() / maxPx
                val tw = if (s > 1f) (dw / s).toInt() else dw
                val th = if (s > 1f) (dh / s).toInt() else dh
                r.getScaledFrameAtTime(us, MediaMetadataRetriever.OPTION_CLOSEST, tw.coerceAtLeast(1), th.coerceAtLeast(1))
                    ?: r.getFrameAtTime(us, MediaMetadataRetriever.OPTION_CLOSEST)
            } else {
                r.getFrameAtTime(us, MediaMetadataRetriever.OPTION_CLOSEST)
            } ?: return null
            val s = maxOf(bmp.width, bmp.height).toFloat() / maxPx
            return if (s > 1f) Bitmap.createScaledBitmap(bmp, (bmp.width / s).toInt(), (bmp.height / s).toInt(), true) else bmp
        } catch (_: Throwable) {
            return null
        } finally {
            runCatching { r.release() }
        }
    }

    /** The painted area as a mask the size of the frame: white where painted, black elsewhere. */
    fun maskBitmap(width: Int, height: Int, strokes: List<PaintStroke>): Bitmap {
        val bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val c = android.graphics.Canvas(bmp)
        c.drawColor(android.graphics.Color.BLACK)
        val p = android.graphics.Paint().apply {
            color = android.graphics.Color.WHITE
            isAntiAlias = false
            strokeCap = android.graphics.Paint.Cap.ROUND
            strokeJoin = android.graphics.Paint.Join.ROUND
        }
        for (s in strokes) {
            if (s.points.isEmpty()) continue
            p.style = android.graphics.Paint.Style.FILL
            for (pt in s.points) c.drawCircle(pt.x, pt.y, s.radius, p)
            if (s.points.size > 1) {
                p.style = android.graphics.Paint.Style.STROKE
                p.strokeWidth = s.radius * 2f
                val path = android.graphics.Path()
                path.moveTo(s.points[0].x, s.points[0].y)
                for (pt in s.points.drop(1)) path.lineTo(pt.x, pt.y)
                c.drawPath(path, p)
            }
        }
        return bmp
    }

    fun save(bmp: Bitmap, file: File, png: Boolean) {
        file.parentFile?.mkdirs()
        FileOutputStream(file).use {
            bmp.compress(if (png) Bitmap.CompressFormat.PNG else Bitmap.CompressFormat.JPEG, if (png) 100 else 92, it)
        }
    }

    fun decode(file: File?): Bitmap? = file?.takeIf { it.isFile }?.let { runCatching { BitmapFactory.decodeFile(it.absolutePath) }.getOrNull() }

    fun clock(ms: Long): String {
        val t = ms.coerceAtLeast(0L)
        return "%d:%02d.%03d".format(t / 60000L, (t / 1000L) % 60L, t % 1000L)
    }
}

private val PrInk = Color(0xFF0B1220)
private val PrPanel = Color(0xFF162033)
private val PrAccent = Color(0xFFC0265C)
private val PrBlue = Color(0xFF2563EB)
private val PrMuted = Color(0xFF94A3B8)
private val PrSlate = Color(0xFF1E293B)
private val PrSub = Color(0xFF64748B)

/** The optional "Patch Removal" button and the list of frozen marks, shown under the target video. */
@Composable
fun PatchRemovalSection(vm: PhoenixViewModel) {
    val video = vm.selectedVideo
    val views = vm.patches.sumOf { it.views.size }
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        OutlinedButton(
            onClick = { vm.openPatchEditor() },
            enabled = video != null && vm.patches.size < PatchFrames.MAX_PATCHES && views < PatchFrames.MAX_VIEWS,
            modifier = Modifier.fillMaxWidth().height(44.dp),
            shape = RoundedCornerShape(14.dp)
        ) {
            Text(
                if (vm.patches.isEmpty()) "Patch Removal (optional)" else "Patch Removal — add another mark",
                fontWeight = FontWeight.Bold, fontSize = 13.sp
            )
        }
        if (vm.patches.size >= PatchFrames.MAX_PATCHES || views >= PatchFrames.MAX_VIEWS) {
            Text("Limit reached (${PatchFrames.MAX_PATCHES} marks, ${PatchFrames.MAX_VIEWS} painted positions) - remove a mark to add another.",
                color = PrAccent, fontSize = 11.sp, lineHeight = 15.sp)
        }
        if (vm.patches.isEmpty()) {
            Text(
                "Erase a tattoo, mole or dark mark: paint over it with your finger, preview, freeze. If the person moves " +
                    "or turns, paint the same mark again at the new position (+ Position) - it is then removed on every frame.",
                color = PrSub, fontSize = 11.sp, lineHeight = 15.sp
            )
        } else {
            Text("${vm.patches.size} mark(s), $views painted position(s), will be removed across the video",
                color = PrSlate, fontSize = 12.sp, fontWeight = FontWeight.SemiBold)
            Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                vm.patches.forEach { p -> FrozenPatchChip(p, onRemove = { vm.removePatch(p.id) }) }
            }
        }
    }
}

@Composable
private fun FrozenPatchChip(p: PatchSpec, onRemove: () -> Unit) {
    val first = p.views.firstOrNull()
    val thumb by produceState<ImageBitmap?>(null, p.id) {
        value = withContext(Dispatchers.IO) {
            PatchFrames.decode(first?.previewFile ?: first?.frameFile)?.let { b ->
                val s = 160f / maxOf(b.width, b.height)
                Bitmap.createScaledBitmap(b, (b.width * s).toInt().coerceAtLeast(1), (b.height * s).toInt().coerceAtLeast(1), true).asImageBitmap()
            }
        }
    }
    Column(
        Modifier.width(104.dp).clip(RoundedCornerShape(12.dp)).background(Color(0xFFF3F5F8)).padding(6.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Box(Modifier.fillMaxWidth().height(56.dp).clip(RoundedCornerShape(8.dp)).background(Color(0xFFE2E8F0))) {
            thumb?.let { Image(it, contentDescription = null, contentScale = ContentScale.Crop, modifier = Modifier.fillMaxSize()) }
        }
        Text(PatchFrames.clock(p.timeMs) + if (p.views.size > 1) " +${p.views.size - 1}" else "", fontSize = 10.5.sp, color = PrSlate)
        Text("Remove", fontSize = 11.sp, color = PrAccent, fontWeight = FontWeight.Bold, modifier = Modifier.clickable { onRemove() }.padding(2.dp))
    }
}

/** Full-screen editor drawn inside the app's own window (1.2.27): portrait - picture on top, controls below; landscape -
 *  picture on the left, controls on the right. The action buttons are never inside a scroll area and the whole screen
 *  sits inside the system bars (safe drawing area) with a margin, so every button is always on the screen. */
@Composable
fun PatchEditorScreen(vm: PhoenixViewModel, s: PatchSession) {
    val context = LocalContext.current
    BackHandler { vm.closePatchEditor() }
    LaunchedEffect(s) {
        if (s.durationMs <= 0L) {
            s.durationMs = withContext(Dispatchers.IO) { PatchFrames.durationMs(context, s.video) }
            s.stepMs = withContext(Dispatchers.IO) { PatchFrames.frameStepMs(context, s.video) }
            if (s.frameMs < 0) {
                s.posMs = (s.durationMs * vm.framePercent / 100L).toFloat()
                s.frameMs = s.posMs.toLong()
            }
        }
    }
    var loadedMs by remember(s) { mutableLongStateOf(if (s.frame != null) s.frameMs else -1L) }
    LaunchedEffect(s.frameMs) {
        if (s.frameMs < 0 || s.frameMs == loadedMs) return@LaunchedEffect
        s.loading = true
        val b = withContext(Dispatchers.IO) { PatchFrames.frameAt(context, s.video, s.frameMs) }
        s.frame = b
        loadedMs = s.frameMs
        s.clearPainting()
        s.resetZoom()
        s.loading = false
    }
    val frameImg = remember(s.frame) { s.frame?.asImageBitmap() }
    val previewImg = remember(s.preview) { s.preview?.asImageBitmap() }
    val showingPreview = s.showPreview && previewImg != null

    fun go(ms: Long) {
        val d = maxOf(0L, s.durationMs)
        val t = ms.coerceIn(0L, if (d > 0) d else ms)
        s.posMs = t.toFloat()
        s.frameMs = t
    }

    @Composable
    fun Picture(modifier: Modifier) {
        Box(modifier.clip(RoundedCornerShape(12.dp)).background(PrPanel), contentAlignment = Alignment.Center) {
            val img = if (showingPreview) previewImg else frameImg
            if (img != null) {
                PatchCanvas(s, img, drawStrokes = !showingPreview, enabled = !vm.patchBusy && !showingPreview,
                    onStroke = { s.preview = null; s.previewView = null; s.showPreview = false })
            }
            if (s.loading) CircularProgressIndicator(color = Color.White)
            if (s.zoom > 1.01f) {
                Text("${"%.1f".format(s.zoom)}×", color = Color.White, fontSize = 11.sp,
                    modifier = Modifier.align(Alignment.TopEnd).padding(6.dp).clip(RoundedCornerShape(6.dp))
                        .background(Color(0x99000000)).padding(horizontal = 6.dp, vertical = 2.dp))
            }
            if (showingPreview) {
                Text("AFTER (erased)", color = Color.White, fontSize = 11.sp, fontWeight = FontWeight.Bold,
                    modifier = Modifier.align(Alignment.TopStart).padding(6.dp).clip(RoundedCornerShape(6.dp))
                        .background(Color(0xCC2563EB)).padding(horizontal = 6.dp, vertical = 2.dp))
            }
        }
    }

    @Composable
    fun Controls() {
        Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text("Frame ${PatchFrames.clock(s.posMs.toLong())}" + if (s.durationMs > 0) " / ${PatchFrames.clock(s.durationMs)}" else "",
                color = PrMuted, fontSize = 11.sp)
            Slider(
                value = s.posMs,
                onValueChange = { s.posMs = it },
                onValueChangeFinished = { go(s.posMs.toLong()) },
                valueRange = 0f..maxOf(1f, s.durationMs.toFloat()),
                enabled = s.durationMs > 0 && !vm.patchBusy,
                modifier = Modifier.height(32.dp)
            )
            ChipRow {
                SmallChip("◀ 1s", !vm.patchBusy) { go(s.frameMs - 1000L) }
                SmallChip("◀ frame", !vm.patchBusy) { go(s.frameMs - s.stepMs) }
                SmallChip("frame ▶", !vm.patchBusy) { go(s.frameMs + s.stepMs) }
                SmallChip("1s ▶", !vm.patchBusy) { go(s.frameMs + 1000L) }
            }
            Text(
                if (s.positions.isEmpty()) "Position 1 of this mark" else
                    "Position ${s.positions.size + 1} of this mark — added: " +
                        s.positions.joinToString("  ") { PatchFrames.clock(it.timeMs) },
                color = Color.White, fontSize = 11.sp, lineHeight = 14.sp
            )
            if (s.positions.isNotEmpty()) {
                ChipRow { SmallChip("Remove the last added position", !vm.patchBusy) { vm.removeLastPosition() } }
            }
            ChipRow {
                Text("Brush", color = PrMuted, fontSize = 11.sp, modifier = Modifier.align(Alignment.CenterVertically))
                listOf("S", "M", "L").forEachIndexed { i, l ->
                    Box(
                        Modifier.size(30.dp).clip(CircleShape).background(if (s.brush == i) PrAccent else PrPanel).clickable { s.brush = i },
                        contentAlignment = Alignment.Center
                    ) { Text(l, color = Color.White, fontWeight = FontWeight.Bold, fontSize = 12.sp) }
                }
                SmallChip("Undo", s.strokes.isNotEmpty() && !vm.patchBusy) {
                    if (s.strokes.isNotEmpty()) s.strokes.removeAt(s.strokes.size - 1)
                    s.preview = null; s.previewView = null; s.showPreview = false
                }
                SmallChip("Clear", s.strokes.isNotEmpty() && !vm.patchBusy) { s.clearPainting() }
            }
            ChipRow {
                Text("Zoom", color = PrMuted, fontSize = 11.sp, modifier = Modifier.align(Alignment.CenterVertically))
                SmallChip("−", s.zoom > 1.01f) { s.zoomBy(1f / 1.6f) }
                SmallChip("+", s.zoom < 11.9f) { s.zoomBy(1.6f) }
                SmallChip("Fit", s.zoom > 1.01f) { s.resetZoom() }
                if (previewImg != null) {
                    SmallChip(if (s.showPreview) "Show before" else "Show after", true) { s.showPreview = !s.showPreview }
                }
            }
            Text(vm.patchMessage ?: when {
                s.frame == null -> "Choose a frame where the mark is clearly visible."
                s.strokes.isEmpty() -> "Paint over the mark (a little past its edge) with one finger. Pinch or + to zoom; a magnifier shows what is under your finger."
                else -> "Preview to check the erase. If the person moves or turns later, + Position and paint the mark there too."
            }, color = PrMuted, fontSize = 11.sp, lineHeight = 14.sp)
        }
    }

    @Composable
    fun Actions(modifier: Modifier) {
        Row(modifier, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            Button(
                onClick = { s.frame?.let { b -> vm.previewPatch(s, b) } },
                enabled = s.frame != null && s.strokes.isNotEmpty() && !vm.patchBusy,
                modifier = Modifier.weight(1f).height(42.dp),
                shape = RoundedCornerShape(12.dp),
                colors = ButtonDefaults.buttonColors(containerColor = PrBlue),
                contentPadding = PaddingValues(horizontal = 4.dp)
            ) {
                if (vm.patchBusy) { CircularProgressIndicator(Modifier.size(14.dp), color = Color.White, strokeWidth = 2.dp); Spacer(Modifier.width(4.dp)) }
                Text(if (vm.patchBusy) "Erasing…" else "Preview", fontWeight = FontWeight.Black, fontSize = 12.sp, maxLines = 1)
            }
            OutlinedButton(
                onClick = { s.frame?.let { b -> vm.addPatchPosition(s, b) } },
                enabled = s.frame != null && s.strokes.isNotEmpty() && !vm.patchBusy &&
                    s.positions.size + 1 < PatchFrames.MAX_VIEWS_PER_MARK,
                modifier = Modifier.weight(1f).height(42.dp),
                shape = RoundedCornerShape(12.dp),
                contentPadding = PaddingValues(horizontal = 4.dp)
            ) { Text("+ Position", color = Color.White, fontWeight = FontWeight.Bold, fontSize = 12.sp, maxLines = 1) }
            Button(
                onClick = { vm.freezePatchMark(s, s.frame) },
                enabled = (s.positions.isNotEmpty() || (s.frame != null && s.strokes.isNotEmpty())) && !vm.patchBusy,
                modifier = Modifier.weight(1f).height(42.dp),
                shape = RoundedCornerShape(12.dp),
                colors = ButtonDefaults.buttonColors(containerColor = PrAccent),
                contentPadding = PaddingValues(horizontal = 4.dp)
            ) {
                val n = s.positions.size + if (s.strokes.isNotEmpty()) 1 else 0
                Text(if (n > 1) "Freeze ($n)" else "Freeze", fontWeight = FontWeight.Black, fontSize = 12.sp, maxLines = 1)
            }
        }
    }

    Surface(Modifier.fillMaxSize(), color = PrInk) {
        BoxWithConstraints(Modifier.fillMaxSize().safeDrawingPadding().padding(horizontal = 12.dp, vertical = 10.dp)) {
            val mw = maxWidth
            val mh = maxHeight
            val landscape = mw > mh
            if (landscape) {
                Row(Modifier.fillMaxSize(), horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    Picture(Modifier.weight(1f).fillMaxHeight())
                    Column(Modifier.width(minOf(300.dp, mw * 0.42f)).fillMaxHeight()) {
                        Header(vm)
                        Column(Modifier.weight(1f).verticalScroll(rememberScrollState())) { Controls() }
                        Spacer(Modifier.height(6.dp))
                        Actions(Modifier.fillMaxWidth())
                    }
                }
            } else {
                Column(Modifier.fillMaxSize(), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    Header(vm)
                    Picture(Modifier.fillMaxWidth().weight(1f))
                    Column(Modifier.fillMaxWidth().heightIn(max = mh * 0.42f).verticalScroll(rememberScrollState())) { Controls() }
                    Actions(Modifier.fillMaxWidth())
                }
            }
        }
    }
}

@Composable
private fun Header(vm: PhoenixViewModel) {
    Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
        Text("Patch Removal", color = Color.White, fontWeight = FontWeight.Black, fontSize = 15.sp, modifier = Modifier.weight(1f))
        TextButton(onClick = { vm.closePatchEditor() }, contentPadding = PaddingValues(horizontal = 8.dp)) {
            Text("Close", color = PrMuted, fontSize = 12.sp)
        }
    }
}

/** A row of chips that scrolls sideways if the screen is narrow (nothing is ever cut off). */
@Composable
private fun ChipRow(content: @Composable RowScope.() -> Unit) {
    Row(Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(5.dp),
        verticalAlignment = Alignment.CenterVertically, content = content)
}

@Composable
private fun SmallChip(text: String, enabled: Boolean, onClick: () -> Unit) {
    Box(
        Modifier.height(30.dp).clip(RoundedCornerShape(8.dp))
            .background(if (enabled) PrPanel else Color(0xFF0F1828))
            .clickable(enabled = enabled) { onClick() }
            .padding(horizontal = 8.dp),
        contentAlignment = Alignment.Center
    ) { Text(text, color = if (enabled) Color.White else Color(0xFF475569), fontSize = 11.sp, fontWeight = FontWeight.SemiBold) }
}

/** The frame with the painted strokes; one finger paints (a magnifier shows what is under it), two fingers zoom and pan. */
@Composable
private fun PatchCanvas(s: PatchSession, image: ImageBitmap, drawStrokes: Boolean, enabled: Boolean, onStroke: () -> Unit) {
    val live = remember { mutableStateListOf<Offset>() }
    val enabledNow by rememberUpdatedState(enabled)
    val onStrokeNow by rememberUpdatedState(onStroke)

    fun toImage(p: Offset): Offset {
        val (f, o) = s.fit()
        return (p - o - s.pan) / (f * s.zoom)
    }

    Canvas(
        Modifier
            .fillMaxSize()
            .clipToBounds()
            .onSizeChanged { s.canvas = it }
            .pointerInput(image) {
                awaitEachGesture {
                    val down = awaitFirstDown(requireUnconsumed = false)
                    var painting = enabledNow
                    live.clear()
                    if (painting) live.add(toImage(down.position))
                    while (true) {
                        val ev = awaitPointerEvent()
                        val pressed = ev.changes.filter { it.pressed }
                        if (pressed.isEmpty()) break
                        if (pressed.size >= 2) {
                            painting = false
                            live.clear()
                            val (_, o) = s.fit()
                            val z = ev.calculateZoom()
                            val c = ev.calculateCentroid(useCurrent = true)
                            val nz = (s.zoom * z).coerceIn(1f, 12f)
                            val k = nz / s.zoom
                            s.pan = if (nz <= 1.0001f) Offset.Zero else (c - o) - (c - o - s.pan) * k + ev.calculatePan()
                            s.zoom = nz
                            ev.changes.forEach { it.consume() }
                        } else if (painting) {
                            live.add(toImage(pressed[0].position))
                            pressed[0].consume()
                        }
                    }
                    if (painting && live.isNotEmpty()) {
                        s.strokes.add(PaintStroke(live.toList(), s.brushPx()))
                        onStrokeNow()
                    }
                    live.clear()
                }
            }
    ) {
        if (s.canvas.width == 0) return@Canvas
        val (f, o) = s.fit()
        val sc = f * s.zoom
        val origin = o + s.pan
        drawImageAt(image, origin, sc)
        val all = if (drawStrokes) s.strokes.toList() + if (live.isNotEmpty()) listOf(PaintStroke(live.toList(), s.brushPx())) else emptyList()
            else emptyList()
        drawPaint(all, origin, sc)
        // magnifier: while painting, what is under the finger, 3x larger, in the corner away from the finger
        if (live.isNotEmpty()) {
            val p = live.last()
            val finger = origin + p * sc
            val r = 62.dp.toPx()
            val left = finger.x > size.width / 2f
            val ctr = Offset(if (left) r + 10f else size.width - r - 10f, r + 10f)
            val mag = sc * 3f
            val o2 = ctr - p * mag
            val circle = Path().apply { addOval(Rect(ctr, r)) }
            clipPath(circle) {
                drawRect(Color.Black, topLeft = ctr - Offset(r, r), size = Size(2 * r, 2 * r))
                drawImageAt(image, o2, mag)
                drawPaint(all, o2, mag)
                drawCircle(Color.White, radius = s.brushPx() * mag, center = ctr, style = Stroke(width = 1.5f))
            }
            drawCircle(Color.White, radius = r, center = ctr, style = Stroke(width = 2.5f))
        }
    }
}

private fun DrawScope.drawImageAt(image: ImageBitmap, origin: Offset, scale: Float) {
    drawImage(
        image = image,
        srcOffset = IntOffset.Zero,
        srcSize = IntSize(image.width, image.height),
        dstOffset = IntOffset(origin.x.toInt(), origin.y.toInt()),
        dstSize = IntSize((image.width * scale).toInt().coerceAtLeast(1), (image.height * scale).toInt().coerceAtLeast(1))
    )
}

private fun DrawScope.drawPaint(strokes: List<PaintStroke>, origin: Offset, scale: Float) {
    val paint = Color(0x99FF2D6A)
    for (st in strokes) {
        val pts = st.points.map { origin + it * scale }
        val r = st.radius * scale
        if (pts.size == 1) {
            drawCircle(paint, radius = r, center = pts[0])
        } else {
            val path = Path().apply {
                moveTo(pts[0].x, pts[0].y)
                pts.drop(1).forEach { lineTo(it.x, it.y) }
            }
            drawPath(path, paint, style = Stroke(width = r * 2f, cap = StrokeCap.Round, join = StrokeJoin.Round))
        }
    }
}
