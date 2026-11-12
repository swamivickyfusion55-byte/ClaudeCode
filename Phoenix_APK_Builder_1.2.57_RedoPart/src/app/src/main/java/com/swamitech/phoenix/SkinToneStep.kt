package com.swamitech.phoenix

// 1.2.51: the Skin tone step (Space v11.2.205+). One slider per replacement slot - that person's whole body (face, neck,
// arms, legs) a little darker or brighter; everyone else stays as filmed. Preview tones the frame picked in step 01 on the
// Space, exactly as the video job will, and shows Before | After with a line to drag.

import android.graphics.BitmapFactory
import android.net.Uri
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectHorizontalDragGestures
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawWithContent
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.clipRect
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.swamitech.phoenix.ui.rememberThumb
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import kotlin.math.abs
import kotlin.math.roundToInt

private val StInk = Color(0xFF0F172A)
private val StMuted = Color(0xFF64748B)
private val StLine = Color(0xFFE6EAF2)
private val StBurgundy = Color(0xFF9B1B4A)
private val StSoft = Color(0xFFFDF2F6)
// the slider's track: deep to light skin, so the direction reads at a glance
private val SkinDeep = Color(0xFF5B3A26)
private val SkinMid = Color(0xFFB98260)
private val SkinLight = Color(0xFFF6DCC6)

/** "Natural", "+25 % brighter", "15 % darker". */
fun skinToneLabel(v: Float): String {
    val p = (abs(v) * 100f).roundToInt()
    return when {
        p == 0 -> "Natural"
        v > 0 -> "+$p % brighter"
        else -> "$p % darker"
    }
}

@Composable
fun SkinToneContent(
    vm: PhoenixViewModel,
    previewButton: @Composable (text: String, enabled: Boolean, busy: Boolean, onClick: () -> Unit) -> Unit,
    resetButton: @Composable (text: String, onClick: () -> Unit) -> Unit,
    empty: @Composable (text: String) -> Unit
) {
    Text(
        "Brighten or darken a person's skin — face, neck, arms and legs. Only the people you move a slider for " +
            "change; everyone else stays as filmed.",
        color = StMuted, fontSize = 12.sp, lineHeight = 16.sp
    )
    val faces = listOf(vm.face1, vm.face2, vm.face3, vm.face4)
    val slots = (1..4).filter { faces[it - 1] != null }
    if (slots.isEmpty()) {
        empty("Add the replacement faces in step 03 first. Each person then gets their own skin tone.")
        return
    }
    slots.forEach { slot -> SkinToneRow(slot, vm.skinThumb(slot) ?: faces[slot - 1], vm.skinToneAt(slot)) { vm.setSkinTone(slot, it) } }
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
        Box(Modifier.weight(1f)) {
            previewButton(
                if (vm.skinBusy) "Toning…" else "Preview at ${vm.framePercent} %",
                vm.selectedVideo != null && vm.skinToneOn && !vm.skinBusy,
                vm.skinBusy
            ) { vm.previewSkin() }
        }
        if (vm.skinToneOn) {
            Box(Modifier.width(118.dp)) { resetButton("All natural") { vm.resetSkinTone() } }
        }
    }
    vm.skinMessage?.let { Text(it, color = StMuted, fontSize = 11.5.sp, lineHeight = 15.sp) }
    val after = vm.skinAfter
    if (after != null) {
        SkinCompare(vm.skinBefore, after, vm.skinPeople)
        if (vm.skinPreviewPct >= 0 && vm.skinPreviewPct != vm.framePercent) {
            Text("This preview is of the frame at ${vm.skinPreviewPct} %. Preview again to see the frame now picked.",
                color = StMuted, fontSize = 11.sp)
        }
    }
    if (vm.skinToneOn) {
        Text("The video uses these settings on every frame of the chosen people.", color = StMuted, fontSize = 11.sp)
    }
}

@Composable
private fun SkinToneRow(slot: Int, face: Uri?, value: Float, onChange: (Float) -> Unit) {
    val thumb by rememberThumb(face, isVideo = false, maxPx = 160)
    val on = abs(value) >= 0.005f
    Row(
        Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(16.dp))
            .background(if (on) StSoft else Color(0xFFF8FAFC))
            .border(1.dp, if (on) StBurgundy.copy(alpha = 0.28f) else StLine, RoundedCornerShape(16.dp))
            .padding(horizontal = 12.dp, vertical = 10.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Box(
            Modifier.size(46.dp).clip(CircleShape).background(Color(0xFFE2E8F0))
                .border(2.dp, if (on) StBurgundy else Color.White, CircleShape),
            contentAlignment = Alignment.Center
        ) {
            val bmp = thumb.bitmap
            if (bmp != null) Image(bmp, "Person #$slot", Modifier.fillMaxSize(), contentScale = ContentScale.Crop)
            else Text("#$slot", color = StMuted, fontWeight = FontWeight.Bold, fontSize = 13.sp)
        }
        Spacer(Modifier.width(12.dp))
        Column(Modifier.weight(1f)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("Person #$slot", color = StInk, fontWeight = FontWeight.Bold, fontSize = 13.5.sp)
                Spacer(Modifier.weight(1f))
                Box(
                    Modifier
                        .clip(RoundedCornerShape(10.dp))
                        .background(if (on) StBurgundy else Color(0xFFEFF2F7))
                        .clickable(enabled = on) { onChange(0f) }
                        .padding(horizontal = 9.dp, vertical = 3.dp)
                ) {
                    Text(skinToneLabel(value), color = if (on) Color.White else StMuted, fontSize = 11.sp, fontWeight = FontWeight.Bold)
                }
            }
            Box(Modifier.fillMaxWidth().height(34.dp), contentAlignment = Alignment.Center) {
                // the track: deep -> light skin, with a mark at Natural
                Box(
                    Modifier.fillMaxWidth().padding(horizontal = 10.dp).height(8.dp).clip(RoundedCornerShape(4.dp))
                        .background(Brush.horizontalGradient(listOf(SkinDeep, SkinMid, SkinLight)))
                )
                Box(Modifier.width(2.dp).height(16.dp).background(Color.White.copy(alpha = 0.9f)))
                Slider(
                    value = value,
                    onValueChange = { onChange(it) },
                    valueRange = -1f..1f,
                    steps = 39,
                    modifier = Modifier.fillMaxWidth(),
                    colors = SliderDefaults.colors(
                        thumbColor = Color.White,
                        activeTrackColor = Color.Transparent,
                        inactiveTrackColor = Color.Transparent,
                        activeTickColor = Color.Transparent,
                        inactiveTickColor = Color.Transparent
                    )
                )
            }
            Row {
                Text("Darker", color = StMuted, fontSize = 10.5.sp)
                Spacer(Modifier.weight(1f))
                Text("Brighter", color = StMuted, fontSize = 10.5.sp)
            }
        }
    }
}

private fun decodeScaled(f: File, maxPx: Int = 1400): ImageBitmap? = runCatching {
    val o = BitmapFactory.Options().apply { inJustDecodeBounds = true }
    BitmapFactory.decodeFile(f.absolutePath, o)
    var s = 1
    while (maxOf(o.outWidth, o.outHeight) / s > maxPx) s *= 2
    BitmapFactory.decodeFile(f.absolutePath, BitmapFactory.Options().apply { inSampleSize = s })?.asImageBitmap()
}.getOrNull()

/** Before | After of the same frame, split by a line to drag (or tap). Each toned person is tagged at their face. */
@Composable
private fun SkinCompare(beforeFile: File?, afterFile: File, people: List<PhoenixViewModel.SkinPerson>) {
    val before by produceState<ImageBitmap?>(null, beforeFile) { value = withContext(Dispatchers.IO) { beforeFile?.let { decodeScaled(it) } } }
    val after by produceState<ImageBitmap?>(null, afterFile) { value = withContext(Dispatchers.IO) { decodeScaled(afterFile) } }
    var split by remember(afterFile) { mutableFloatStateOf(0.5f) }
    val a = after
    if (a == null) {
        Box(Modifier.fillMaxWidth().height(160.dp).clip(RoundedCornerShape(14.dp)).background(Color(0xFFF1F5F9)), contentAlignment = Alignment.Center) {
            CircularProgressIndicator(Modifier.size(22.dp), color = StBurgundy, strokeWidth = 2.dp)
        }
        return
    }
    val shape = RoundedCornerShape(14.dp)
    BoxWithConstraints(
        Modifier
            .fillMaxWidth()
            .aspectRatio(a.width.toFloat() / a.height.toFloat())
            .clip(shape)
            .border(1.dp, StLine, shape)
            .pointerInput(afterFile) { detectTapGestures { p -> split = (p.x / size.width).coerceIn(0f, 1f) } }
            .pointerInput(afterFile) {
                detectHorizontalDragGestures { change, _ -> split = (change.position.x / size.width).coerceIn(0f, 1f) }
            }
    ) {
        Image(a, "After", Modifier.fillMaxSize(), contentScale = ContentScale.FillBounds)
        before?.let { b ->
            Image(
                b, "Before",
                Modifier.fillMaxSize().drawWithContent { clipRect(right = size.width * split) { this@drawWithContent.drawContent() } },
                contentScale = ContentScale.FillBounds
            )
        }
        Canvas(Modifier.fillMaxSize()) {
            val x = size.width * split
            drawLine(Color.White, Offset(x, 0f), Offset(x, size.height), strokeWidth = 2.dp.toPx())
            val c = Offset(x, size.height / 2f)
            drawCircle(Color.White, 13.dp.toPx(), c)
            drawCircle(StBurgundy, 13.dp.toPx(), c, style = androidx.compose.ui.graphics.drawscope.Stroke(2.dp.toPx()))
            val h = 4.dp.toPx()
            drawLine(StBurgundy, Offset(c.x - 3.dp.toPx(), c.y), Offset(c.x - 3.dp.toPx() - h, c.y), strokeWidth = 2.dp.toPx())
            drawLine(StBurgundy, Offset(c.x + 3.dp.toPx(), c.y), Offset(c.x + 3.dp.toPx() + h, c.y), strokeWidth = 2.dp.toPx())
        }
        val w = maxWidth
        val hh = maxHeight
        people.filter { it.box.size == 4 && abs(it.amount) >= 0.005f }.forEach { p ->
            Box(
                Modifier
                    .offset(x = w * p.box[0], y = (hh * p.box[1] - 22.dp).coerceAtLeast(0.dp))
                    .clip(RoundedCornerShape(8.dp))
                    .background(Color(0xCC9B1B4A))
                    .padding(horizontal = 6.dp, vertical = 2.dp)
            ) {
                Text("#${p.slot} ${skinToneLabel(p.amount)}", color = Color.White, fontSize = 10.sp, fontWeight = FontWeight.Bold)
            }
        }
        SkinTag("Before", Modifier.align(Alignment.BottomStart))
        SkinTag("After", Modifier.align(Alignment.BottomEnd))
    }
}

@Composable
private fun SkinTag(text: String, modifier: Modifier) {
    Box(
        modifier.padding(8.dp).clip(RoundedCornerShape(8.dp)).background(Color(0x99000000))
            .padding(horizontal = 8.dp, vertical = 3.dp)
    ) { Text(text, color = Color.White, fontSize = 11.sp, fontWeight = FontWeight.Bold) }
}
