package com.swamitech.phoenix

import android.content.Context
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
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.clipToBounds
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream

/*
 * Patch Removal (app 1.2.25, Space v11.2.96 / SDOS-096).
 *
 * The user picks a frame of the target video ON THE PHONE (the exact frame, not the nearest key frame), paints over a
 * mark with a finger (tattoo, mole, any dark spot, any shape; pinch to zoom for small ones), asks the Space for a
 * preview of the frame with the mark removed, and freezes it. Frozen marks travel with the job; the Space anchors each
 * on its frame, follows the skin through the whole video and removes the mark on every frame before the face swap.
 */

/** One frozen mark: the frame it was painted on, the painted mask (white on black, same size), its time. */
data class PatchSpec(
    val id: String,
    val frameFile: File,
    val maskFile: File,
    val previewFile: File?,
    val timeMs: Long,
    val width: Int,
    val height: Int
)

/** A finger stroke in FRAME pixels (so it survives zoom / pan) with its brush radius in frame pixels. */
data class PaintStroke(val points: List<Offset>, val radius: Float)

object PatchFrames {
    const val MAX_PX = 1280            // the frame sent to the Space (long side)
    const val MAX_PATCHES = 4

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
private val PrMuted = Color(0xFF94A3B8)
private val PrSlate = Color(0xFF1E293B)
private val PrSub = Color(0xFF64748B)

/** The optional "Patch Removal" button and the list of frozen marks, shown under the target video. */
@Composable
fun PatchRemovalSection(vm: PhoenixViewModel) {
    var open by remember { mutableStateOf(false) }
    val video = vm.selectedVideo
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        OutlinedButton(
            onClick = { open = true },
            enabled = video != null && vm.patches.size < PatchFrames.MAX_PATCHES,
            modifier = Modifier.fillMaxWidth().height(44.dp),
            shape = RoundedCornerShape(14.dp)
        ) {
            Text(
                if (vm.patches.isEmpty()) "Patch Removal (optional)" else "Patch Removal — add another mark",
                fontWeight = FontWeight.Bold, fontSize = 13.sp
            )
        }
        if (vm.patches.isEmpty()) {
            Text(
                "Erase a tattoo, mole or dark mark: paint over it with your finger on one frame, preview, freeze. " +
                    "It is then removed on every frame of the video, even as it moves.",
                color = PrSub, fontSize = 11.sp, lineHeight = 15.sp
            )
        } else {
            Text("${vm.patches.size} mark(s) will be removed across the video", color = PrSlate, fontSize = 12.sp, fontWeight = FontWeight.SemiBold)
            Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                vm.patches.forEach { p -> FrozenPatchChip(p, onRemove = { vm.removePatch(p.id) }) }
            }
        }
    }
    if (open && video != null) {
        PatchEditorDialog(vm, video, onClose = { open = false })
    }
}

@Composable
private fun FrozenPatchChip(p: PatchSpec, onRemove: () -> Unit) {
    val thumb by produceState<ImageBitmap?>(null, p.id) {
        value = withContext(Dispatchers.IO) {
            PatchFrames.decode(p.previewFile ?: p.frameFile)?.let { b ->
                val s = 160f / maxOf(b.width, b.height)
                Bitmap.createScaledBitmap(b, (b.width * s).toInt().coerceAtLeast(1), (b.height * s).toInt().coerceAtLeast(1), true).asImageBitmap()
            }
        }
    }
    Column(
        Modifier.width(96.dp).clip(RoundedCornerShape(12.dp)).background(Color(0xFFF3F5F8)).padding(6.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Box(Modifier.fillMaxWidth().height(56.dp).clip(RoundedCornerShape(8.dp)).background(Color(0xFFE2E8F0))) {
            thumb?.let { Image(it, contentDescription = null, contentScale = ContentScale.Crop, modifier = Modifier.fillMaxSize()) }
        }
        Text(PatchFrames.clock(p.timeMs), fontSize = 10.5.sp, color = PrSlate)
        Text("Remove", fontSize = 11.sp, color = PrAccent, fontWeight = FontWeight.Bold, modifier = Modifier.clickable { onRemove() }.padding(2.dp))
    }
}

/** Full-screen editor: choose the frame, paint over the mark, preview the erase, freeze. */
@Composable
fun PatchEditorDialog(vm: PhoenixViewModel, video: Uri, onClose: () -> Unit) {
    val context = LocalContext.current
    var durationMs by remember { mutableLongStateOf(0L) }
    var posMs by remember { mutableFloatStateOf(0f) }
    var shownMs by remember { mutableLongStateOf(-1L) }
    var frame by remember { mutableStateOf<Bitmap?>(null) }
    var frameImg by remember { mutableStateOf<ImageBitmap?>(null) }
    var loading by remember { mutableStateOf(false) }
    val strokes = remember { mutableStateListOf<PaintStroke>() }
    var brush by remember { mutableIntStateOf(1) }                  // 0 small, 1 medium, 2 large
    var preview by remember { mutableStateOf<ImageBitmap?>(null) }
    var previewFile by remember { mutableStateOf<File?>(null) }
    var showPreview by remember { mutableStateOf(false) }
    var draft by remember { mutableStateOf<PatchDraft?>(null) }
    var message by remember { mutableStateOf("Choose the frame where the mark is clearly visible, then paint over it.") }
    var zoomReset by remember { mutableIntStateOf(0) }

    LaunchedEffect(video) {
        durationMs = withContext(Dispatchers.IO) { PatchFrames.durationMs(context, video) }
        posMs = (durationMs * vm.framePercent / 100L).toFloat()
        shownMs = posMs.toLong()
    }
    LaunchedEffect(shownMs) {
        if (shownMs < 0) return@LaunchedEffect
        loading = true
        val b = withContext(Dispatchers.IO) { PatchFrames.frameAt(context, video, shownMs) }
        frame = b
        frameImg = b?.asImageBitmap()
        strokes.clear(); preview = null; previewFile = null; showPreview = false; draft = null
        zoomReset++
        loading = false
        message = if (b == null) "This frame could not be read. Try another position." else
            "Paint over the mark (one finger). Pinch with two fingers to zoom. Then tap Preview."
    }
    val fw = frame?.width ?: 1
    val brushPx = maxOf(fw, frame?.height ?: 1) * when (brush) { 0 -> 0.008f; 1 -> 0.016f; else -> 0.03f }

    Dialog(onDismissRequest = onClose, properties = DialogProperties(usePlatformDefaultWidth = false)) {
        Surface(Modifier.fillMaxSize(), color = PrInk) {
            Column(Modifier.fillMaxSize().padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                    Text("Patch Removal", color = Color.White, fontWeight = FontWeight.Black, fontSize = 18.sp, modifier = Modifier.weight(1f))
                    TextButton(onClick = onClose) { Text("Close", color = PrMuted) }
                }
                Text("Frame ${PatchFrames.clock(posMs.toLong())}" + if (durationMs > 0) " of ${PatchFrames.clock(durationMs)}" else "",
                    color = PrMuted, fontSize = 12.sp)
                Slider(
                    value = posMs,
                    onValueChange = { posMs = it },
                    onValueChangeFinished = { shownMs = posMs.toLong() },
                    valueRange = 0f..maxOf(1f, durationMs.toFloat()),
                    enabled = durationMs > 0 && !vm.patchBusy
                )
                Box(
                    Modifier.fillMaxWidth().weight(1f).clip(RoundedCornerShape(12.dp)).background(PrPanel),
                    contentAlignment = Alignment.Center
                ) {
                    val img = if (showPreview && preview != null) preview else frameImg
                    if (img != null) {
                        PatchCanvas(
                            image = img,
                            strokes = strokes,
                            drawStrokes = !(showPreview && preview != null),
                            brushPx = brushPx,
                            resetKey = zoomReset,
                            enabled = !vm.patchBusy,
                            onStroke = { preview = null; previewFile = null; showPreview = false; draft = null }
                        )
                    }
                    if (loading) CircularProgressIndicator(color = Color.White)
                }
                Row(horizontalArrangement = Arrangement.spacedBy(6.dp), verticalAlignment = Alignment.CenterVertically) {
                    listOf("S", "M", "L").forEachIndexed { i, l ->
                        val on = brush == i
                        Box(
                            Modifier.size(36.dp).clip(CircleShape).background(if (on) PrAccent else PrPanel).clickable { brush = i },
                            contentAlignment = Alignment.Center
                        ) { Text(l, color = Color.White, fontWeight = FontWeight.Bold, fontSize = 13.sp) }
                    }
                    Spacer(Modifier.width(4.dp))
                    TextButton(onClick = { if (strokes.isNotEmpty()) strokes.removeAt(strokes.size - 1); preview = null; showPreview = false; draft = null },
                        enabled = strokes.isNotEmpty() && !vm.patchBusy) { Text("Undo", color = Color.White) }
                    TextButton(onClick = { strokes.clear(); preview = null; showPreview = false; draft = null },
                        enabled = strokes.isNotEmpty() && !vm.patchBusy) { Text("Clear", color = Color.White) }
                    TextButton(onClick = { zoomReset++ }) { Text("Fit", color = Color.White) }
                    if (preview != null) {
                        Spacer(Modifier.weight(1f))
                        TextButton(onClick = { showPreview = !showPreview }) {
                            Text(if (showPreview) "Show before" else "Show after", color = Color.White, fontWeight = FontWeight.Bold)
                        }
                    }
                }
                Text(vm.patchMessage ?: message, color = PrMuted, fontSize = 12.sp, lineHeight = 16.sp)
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(
                        onClick = {
                            val b = frame ?: return@Button
                            val d = vm.makePatchDraft(b, strokes.toList(), shownMs)
                            draft = d
                            vm.previewPatch(d) { file ->
                                previewFile = file
                                preview = PatchFrames.decode(file)?.asImageBitmap()
                                showPreview = preview != null
                            }
                        },
                        enabled = frame != null && strokes.isNotEmpty() && !vm.patchBusy,
                        modifier = Modifier.weight(1f).height(48.dp),
                        shape = RoundedCornerShape(14.dp),
                        colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF2563EB))
                    ) {
                        if (vm.patchBusy) { CircularProgressIndicator(Modifier.size(16.dp), color = Color.White, strokeWidth = 2.dp); Spacer(Modifier.width(8.dp)) }
                        Text(if (vm.patchBusy) "Erasing…" else "Preview", fontWeight = FontWeight.Black)
                    }
                    Button(
                        onClick = {
                            val b = frame ?: return@Button
                            val d = draft ?: vm.makePatchDraft(b, strokes.toList(), shownMs)
                            vm.freezePatch(d, previewFile)
                            onClose()
                        },
                        enabled = frame != null && strokes.isNotEmpty() && !vm.patchBusy,
                        modifier = Modifier.weight(1f).height(48.dp),
                        shape = RoundedCornerShape(14.dp),
                        colors = ButtonDefaults.buttonColors(containerColor = PrAccent)
                    ) { Text(if (preview != null) "Freeze" else "Freeze (no preview)", fontWeight = FontWeight.Black) }
                }
            }
        }
    }
}

/** The frame with the painted strokes; one finger paints, two fingers zoom and pan. */
@Composable
private fun PatchCanvas(
    image: ImageBitmap,
    strokes: MutableList<PaintStroke>,
    drawStrokes: Boolean,
    brushPx: Float,
    resetKey: Int,
    enabled: Boolean,
    onStroke: () -> Unit
) {
    var size by remember { mutableStateOf(IntSize.Zero) }
    var zoom by remember(resetKey) { mutableFloatStateOf(1f) }
    var pan by remember(resetKey) { mutableStateOf(Offset.Zero) }
    val live = remember { mutableStateListOf<Offset>() }
    val brushNow by rememberUpdatedState(brushPx)
    val enabledNow by rememberUpdatedState(enabled)
    val onStrokeNow by rememberUpdatedState(onStroke)

    fun fit(): Pair<Float, Offset> {
        val s = minOf(size.width / image.width.toFloat(), size.height / image.height.toFloat()).coerceAtLeast(1e-3f)
        val o = Offset((size.width - image.width * s) / 2f, (size.height - image.height * s) / 2f)
        return s to o
    }
    fun toImage(p: Offset): Offset {
        val (s, o) = fit()
        return (p - o - pan) / (s * zoom)
    }

    Canvas(
        Modifier
            .fillMaxSize()
            .clipToBounds()
            .onSizeChanged { size = it }
            .pointerInput(image, resetKey) {
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
                            val (_, o) = fit()
                            val z = ev.calculateZoom()
                            val c = ev.calculateCentroid(useCurrent = true)
                            val nz = (zoom * z).coerceIn(1f, 10f)
                            val k = nz / zoom
                            pan = (c - o) - (c - o - pan) * k + ev.calculatePan()
                            zoom = nz
                            ev.changes.forEach { it.consume() }
                        } else if (painting) {
                            live.add(toImage(pressed[0].position))
                            pressed[0].consume()
                        }
                    }
                    if (painting && live.isNotEmpty()) {
                        strokes.add(PaintStroke(live.toList(), brushNow))
                        onStrokeNow()
                    }
                    live.clear()
                }
            }
    ) {
        if (size.width == 0) return@Canvas
        val (s, o) = fit()
        val sc = s * zoom
        val origin = o + pan
        drawImage(
            image = image,
            srcOffset = IntOffset.Zero,
            srcSize = IntSize(image.width, image.height),
            dstOffset = IntOffset(origin.x.toInt(), origin.y.toInt()),
            dstSize = IntSize((image.width * sc).toInt().coerceAtLeast(1), (image.height * sc).toInt().coerceAtLeast(1))
        )
        if (drawStrokes) {
            val paint = Color(0x99FF2D6A)
            val all = strokes.toList() + if (live.isNotEmpty()) listOf(PaintStroke(live.toList(), brushNow)) else emptyList()
            for (st in all) {
                val pts = st.points.map { origin + it * sc }
                val r = st.radius * sc
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
    }
}

/** A painted mark saved for the Space: frame JPEG + mask PNG of the same size, and the frame's time. */
data class PatchDraft(val id: String, val frameFile: File, val maskFile: File, val timeMs: Long, val width: Int, val height: Int)
