package com.swamitech.phoenix

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.tween
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.platform.LocalConfiguration
import android.content.res.Configuration
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.swamitech.phoenix.ui.ThumbState
import com.swamitech.phoenix.ui.rememberThumb
import java.text.DateFormat
import java.util.Date

/* ----------------------------------------------------------------------- */
/* Modern brand palette — soft, high-contrast, 2026-trendy                  */
/* ----------------------------------------------------------------------- */

private val Navy = Color(0xFF0B1220)
private val NavyMid = Color(0xFF162033)
private val Burgundy = Color(0xFF9B1B4A)          // richer primary
private val BurgundyLite = Color(0xFFC0265C)
private val BurgundySoft = Color(0xFFFDF2F6)      // very light tint surfaces
private val Gold = Color(0xFFD4A017)
private val Blue = Color(0xFF2563EB)
private val Page = Color(0xFFF7F8FC)              // cooler, cleaner background
private val Ink = Color(0xFF0F172A)
private val Muted = Color(0xFF64748B)
private val Line = Color(0xFFE2E8F0)
private val SoftBlue = Color(0xFFEFF6FF)
private val SoftGold = Color(0xFFFFFBEB)
private val SoftGreen = Color(0xFFECFDF5)
private val SoftRed = Color(0xFFFEF2F2)
private val Slate = Color(0xFF1E293B)
private val CardShadow = Color(0x0A0F172A)

private val PhoenixColors = lightColorScheme(
    primary = Burgundy,
    onPrimary = Color.White,
    secondary = Blue,
    tertiary = Gold,
    background = Page,
    surface = Color.White,
    surfaceVariant = BurgundySoft,
    onSurface = Ink,
    onSurfaceVariant = Muted,
    outline = Line,
    outlineVariant = Color(0xFFF1F5F9)
)

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        runCatching { stopService(Intent(this, PhoenixKeepAliveService::class.java)) }
        // targetSdk 35 forces edge-to-edge on Android 15. Without this (and the
        // inset padding below) the bottom action bar sits under the system
        // navigation bar and appears to be missing.
        enableEdgeToEdge()
        setContent { MaterialTheme(colorScheme = PhoenixColors) { PhoenixApp() } }
    }
}

/* ----------------------------------------------------------------------- */
/* Root                                                                     */
/* ----------------------------------------------------------------------- */

@Composable
private fun PhoenixApp(vm: PhoenixViewModel = viewModel()) {
    val context = LocalContext.current

    fun persist(uri: Uri?) {
        if (uri == null) return
        runCatching {
            context.contentResolver.takePersistableUriPermission(
                uri, Intent.FLAG_GRANT_READ_URI_PERMISSION
            )
        }
    }

    val videoPicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) {
        persist(it); vm.selectVideo(it)
    }
    val imagePicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) {
        persist(it); vm.selectImage(it)
    }
    val face1 = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) {
        persist(it); vm.selectFace(1, it)
    }
    val face2 = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) {
        persist(it); vm.selectFace(2, it)
    }
    val face3 = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) {
        persist(it); vm.selectFace(3, it)
    }
    val face4 = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) {
        persist(it); vm.selectFace(4, it)
    }
    val facePickers = listOf(face1, face2, face3, face4)

    // Write permission, unlike persist() above which only ever grants read
    // access for source files the app is consuming, not creating.
    val outputFolderPicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocumentTree()) { uri ->
        if (uri != null) {
            runCatching {
                context.contentResolver.takePersistableUriPermission(
                    uri,
                    Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_WRITE_URI_PERMISSION
                )
            }
            vm.setOutputTree(uri)
        }
    }

    // Update Space code: a shipped package .zip, or its loose files. Read
    // once, straight away, so no persistable permission is needed.
    val packagePicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenMultipleDocuments()) { uris ->
        vm.loadDeployPackage(uris)
    }

    var showConnection by remember { mutableStateOf(false) }
    var showDeploy by remember { mutableStateOf(false) }
    var showAdvanced by remember { mutableStateOf(false) }
    var token by remember { mutableStateOf("") }
    var showPrivacy by remember { mutableStateOf(false) }

    Column(
        Modifier
            .fillMaxSize()
            .background(Page)
    ) {
        TopBar(vm)
        TabBar(vm)
        // Space routing is a first-step decision, so keep it visible without
        // requiring the user to scroll down into Connection settings.
        if (vm.activeTab != "History") {
            TopSpaceSelectorBar(vm)
        }

        // FIX: previously this Column used fillMaxSize(), which consumed every
        // remaining pixel and pushed the bottom action bar (the Swap button)
        // completely off screen. weight(1f) leaves room for it.
        Column(
            Modifier
                .weight(1f)
                .fillMaxWidth()
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 16.dp)
                .padding(top = 8.dp, bottom = 20.dp),
            verticalArrangement = Arrangement.spacedBy(9.dp)
        ) {
            when (vm.activeTab) {
                "Image" -> ImageWorkspace(vm, imagePicker, facePickers)
                "History" -> HistoryWorkspace(vm)
                else -> VideoWorkspace(vm, videoPicker, facePickers, outputFolderPicker, showAdvanced) {
                    showAdvanced = !showAdvanced
                }
            }
            if (vm.activeTab != "History") {
                ConnectionCard(
                    vm = vm,
                    open = showConnection,
                    toggle = { showConnection = !showConnection },
                    token = token,
                    setToken = { token = it },
                    clearToken = { token = "" }
                )
                DeployCard(
                    vm = vm,
                    open = showDeploy,
                    toggle = { showDeploy = !showDeploy },
                    pickPackage = { packagePicker.launch(arrayOf("*/*")) }
                )
                PrivacyCard { showPrivacy = true }
            }
        }

        if (vm.activeTab != "History") BottomActionBar(vm) else Spacer(Modifier.navigationBarsPadding())
    }

    if (showPrivacy) {
        AlertDialog(
            onDismissRequest = { showPrivacy = false },
            title = { Text("Privacy & security", fontWeight = FontWeight.Bold) },
            text = {
                Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                    Text(
                        "Cloud mode sends the selected media and replacement faces to the private Phoenix " +
                            "Space you configured. Your Hugging Face token is encrypted with the Android " +
                            "Keystore and never leaves the device except as a bearer header to your own Space."
                    )
                    Text(
                        "If Android's DNS resolver fails, the app falls back to DNS-over-HTTPS (Cloudflare) " +
                            "to look up your Space's address. That's a hostname lookup only - none of your " +
                            "media or token is included in it."
                    )
                    Text(
                        "Finished results are saved to Movies/Phoenix or Pictures/Phoenix - a public, " +
                            "shared album visible to your Gallery and any app with media access, not private " +
                            "to Phoenix. Uploaded source files and intermediate previews are deleted from the " +
                            "app's own cache as soon as each step no longer needs them; \"Clear cache now\" " +
                            "below removes anything left over from before this cleanup existed."
                    )
                    GhostButton("Clear cache now") { vm.clearLocalCache() }
                }
            },
            confirmButton = { Button({ showPrivacy = false }) { Text("Got it") } }
        )
    }
}

/* ----------------------------------------------------------------------- */
/* Chrome                                                                   */
/* ----------------------------------------------------------------------- */

@Composable
private fun TopBar(vm: PhoenixViewModel) {
    val isLandscape = LocalConfiguration.current.orientation == Configuration.ORIENTATION_LANDSCAPE
    val live = vm.activeJobCount > 0
    Column(
        Modifier
            .fillMaxWidth()
            .background(
                Brush.linearGradient(
                    listOf(Color(0xFF08111F), Color(0xFF15152A), Color(0xFF251326))
                )
            )
            .statusBarsPadding()
            .padding(horizontal = 18.dp, vertical = if (isLandscape) 4.dp else 9.dp)
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Box(
                Modifier
                    .size(38.dp)
                    .shadow(9.dp, RoundedCornerShape(12.dp), clip = false)
                    .clip(RoundedCornerShape(12.dp))
                    .background(Brush.linearGradient(listOf(BurgundyLite, Color(0xFFE64B7A)))),
                contentAlignment = Alignment.Center
            ) {
                Text("✦", color = Color.White, fontSize = 18.sp, fontWeight = FontWeight.Black)
            }
            Spacer(Modifier.width(12.dp))
            Column(Modifier.weight(1f)) {
                Text("PHOENIX", color = Color.White, fontWeight = FontWeight.Black, fontSize = 17.sp, letterSpacing = 1.1.sp)
                Text("AI face studio", color = Color(0xFF9BA8BA), fontSize = 9.5.sp, fontWeight = FontWeight.Medium)
            }

            Box(
                Modifier
                    .clip(RoundedCornerShape(50.dp))
                    .background(if (live) Color(0x1A34D399) else Color(0x14FFFFFF))
                    .border(1.dp, if (live) Color(0x3834D399) else Color(0x20FFFFFF), RoundedCornerShape(50.dp))
                    .padding(horizontal = 10.dp, vertical = 7.dp)
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Box(Modifier.size(7.dp).clip(CircleShape).background(if (live) Color(0xFF34D399) else Color(0xFF94A3B8)))
                    Spacer(Modifier.width(6.dp))
                    Text(if (live) "${vm.activeJobCount} LIVE" else "READY", color = if (live) Color(0xFF6EE7B7) else Color(0xFFCBD5E1), fontSize = 10.sp, fontWeight = FontWeight.Black, letterSpacing = .5.sp)
                }
            }
        }

        // Idle, this card only repeated the space name (already shown in
        // PROCESSING SPACE below) and a static "Ready" message - no
        // information, just height. It earns its place once a job is
        // actually running, when vm.status carries real progress/ETA text.
        if (live || vm.farmPhase.isNotBlank()) {
            Spacer(Modifier.height(7.dp))
            Column(
                Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(15.dp))
                    .background(Color(0x12FFFFFF))
                    .border(1.dp, Color(0x18FFFFFF), RoundedCornerShape(15.dp))
                    .padding(horizontal = 12.dp, vertical = 8.dp)
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Box(Modifier.size(8.dp).clip(CircleShape).background(if (vm.hfTokenSet) Color(0xFF34D399) else Gold))
                    Spacer(Modifier.width(9.dp))
                    Column(Modifier.weight(1f)) {
                        Text(
                            vm.farmPhase.ifBlank { vm.spaceLabel },
                            color = Color(0xFFF6A8C5), fontSize = 10.5.sp, fontWeight = FontWeight.Bold,
                            maxLines = 1, overflow = TextOverflow.Ellipsis
                        )
                        Text(vm.status, color = Color(0xFFD8E0EA), fontSize = 11.5.sp, maxLines = 1, overflow = TextOverflow.Ellipsis)
                    }
                    Text(
                        if (vm.farmPercent > 0) "${vm.farmPercent}%" else if (vm.hfTokenSet) "SECURE" else "SETUP",
                        color = if (vm.farmPercent > 0) Gold else if (vm.hfTokenSet) Color(0xFF6EE7B7) else Color(0xFFFCD34D),
                        fontSize = 12.sp, fontWeight = FontWeight.Black
                    )
                }
                if (vm.farmPercent > 0) {
                    Spacer(Modifier.height(6.dp))
                    LinearProgressIndicator(
                        progress = { (vm.farmPercent / 100f).coerceIn(0f, 1f) },
                        modifier = Modifier.fillMaxWidth().height(4.dp).clip(CircleShape),
                        color = Gold,
                        trackColor = Color(0x20FFFFFF)
                    )
                }
            }
        }

        // The per-job progress rows are the single largest pinned consumer
        // (up to 84dp) and they duplicate what the History tab already shows.
        // In landscape the pinned header was taking over half the viewport,
        // so drop them there and keep the one-line status pill above.
        if (!isLandscape) {
            ActiveJobsStrip(vm)
        }
    }
}

/**
 * The status pill above shows one global string, which whichever job's
 * coroutine happens to update last simply overwrites - with real concurrency
 * across up to MAX_SPACES Spaces, that string flickers between unrelated
 * jobs and never gives a stable picture of what's actually running where.
 * This shows every currently-active job's own Space, progress, and ETA at
 * once, visible on every tab (not just History), so multi-Space runs are
 * actually legible instead of a guess.
 */
@Composable
private fun ActiveJobsStrip(vm: PhoenixViewModel) {
    val active = vm.history.filter { it.status !in setOf("Done", "Failed", "Monitoring stopped") }
    if (active.isEmpty()) return
    Column(
        Modifier
            .fillMaxWidth()
            .padding(top = 6.dp)
            .heightIn(max = 84.dp)
            .verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(5.dp)
    ) {
        active.take(PhoenixViewModel.MAX_SPACES).forEach { item ->
            Row(
                Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(10.dp))
                    .background(Color(0x0EFFFFFF))
                    .padding(horizontal = 9.dp, vertical = 4.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Box(Modifier.size(6.dp).clip(CircleShape).background(Color(0xFF34D399)))
                Spacer(Modifier.width(7.dp))
                Text(item.spaceLabel, color = Color(0xFFE8B4C8), fontSize = 10.sp, fontWeight = FontWeight.Bold, maxLines = 1, overflow = TextOverflow.Ellipsis, modifier = Modifier.weight(.85f))
                LinearProgressIndicator(progress = { item.progress.coerceIn(0f,1f) }, modifier = Modifier.weight(1.5f).height(4.dp).clip(CircleShape), color = Gold, trackColor = Color(0x20FFFFFF))
                Spacer(Modifier.width(8.dp))
                Text("${(item.progress * 100).toInt()}%", color = Color(0xFFCBD5E1), fontSize = 9.5.sp, fontWeight = FontWeight.Bold)
            }
        }
    }
}

@Composable
private fun TabBar(vm: PhoenixViewModel) {
    Row(
        Modifier.fillMaxWidth().background(Page).padding(horizontal = 16.dp, vertical = 6.dp),
        horizontalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        listOf("Image" to "▣", "Video" to "▶", "History" to "◷").forEach { (tab, icon) ->
            val selected = vm.activeTab == tab
            val bg by animateColorAsState(if (selected) Ink else Color.White, animationSpec = tween(180), label = "tabBg")
            val fg by animateColorAsState(if (selected) Color.White else Muted, animationSpec = tween(180), label = "tabFg")
            Row(
                Modifier
                    .weight(1f)
                    .height(45.dp)
                    .clip(RoundedCornerShape(15.dp))
                    .background(bg)
                    .border(1.dp, if (selected) Color.Transparent else Line, RoundedCornerShape(15.dp))
                    .clickable { vm.setTab(tab) },
                horizontalArrangement = Arrangement.Center,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(icon, color = if (selected) Color(0xFFFFB6CE) else Muted, fontSize = 14.sp, fontWeight = FontWeight.Bold)
                Spacer(Modifier.width(7.dp))
                Text(tab, color = fg, fontSize = 12.5.sp, fontWeight = if (selected) FontWeight.Bold else FontWeight.SemiBold)
            }
        }
    }
}

/* ----------------------------------------------------------------------- */
/* Workspaces                                                               */
/* ----------------------------------------------------------------------- */

@Composable
private fun VideoWorkspace(
    vm: PhoenixViewModel,
    picker: ActivityResultLauncher<Array<String>>,
    facePickers: List<ActivityResultLauncher<Array<String>>>,
    outputFolderPicker: ActivityResultLauncher<Uri?>,
    advancedOpen: Boolean,
    toggleAdvanced: () -> Unit
) {
    StepCard("01", "Target video", "Pick the clip, then slide to a frame where the faces are clearly visible.") {
        // The frame picker sits directly under the preview, which follows the
        // slider ("FRAME n%"), so the frame being picked is always in view -
        // it used to be in the next card, below the file details, and needed
        // scrolling back up after every move.
        MediaDropZone(
            uri = vm.selectedVideo,
            isVideo = true,
            framePercent = vm.framePercent,
            emptyTitle = "Select a video",
            emptyHint = "MP4 / MOV / MKV and other Android-supported formats",
            onPick = { picker.launch(arrayOf("video/*")) },
            onClear = { vm.selectVideo(null) },
            belowPreview = {
                Text(
                    "Frame position \u2014 ${vm.framePercent}%",
                    color = Slate,
                    fontWeight = FontWeight.SemiBold,
                    fontSize = 12.5.sp
                )
                Slider(
                    value = vm.framePercent.toFloat(),
                    onValueChange = { vm.updateFrame(it.toInt()) },
                    valueRange = 0f..100f,
                    steps = 99,
                    enabled = vm.selectedVideo != null
                )
                ActionButton(
                    text = if (vm.detectingFrame) "Detecting\u2026" else "Show frame & detect faces",
                    enabled = vm.selectedVideo != null && !vm.detectingFrame,
                    busy = vm.detectingFrame,
                    onClick = { vm.detectFrame() }
                )
            }
        )
    }

    StepCard("02", "Detected faces", "The frame the Space analysed and the faces it found.") {
        vm.detectedFrameUri?.let { PreviewPanel(it, "Annotated frame returned by the Space") }
        ErrorPanel(vm)
        DetectedFacesStrip(vm)
    }

    StepCard("03", "Replacement faces", "Up to four source faces. Tap a tile to upload, or capture a detected face.") {
        FaceSlotGrid(vm, facePickers)
    }

    StepCard("04", "Output & quality", "Presets cover most cases; open advanced controls for fine tuning.") {
        PresetButtons(vm)
        Spacer(Modifier.height(4.dp))
        Dropdown("Download quality", listOf("640p (Fast)", "680p", "720p (HD)", "900p (HD+)", "1080p (Full HD)"), vm.options.resolution) { v ->
            vm.updateOptions { it.copy(resolution = v) }
        }
        Dropdown("Quality", listOf("Fast", "Balanced", "Optimized", "Best", "Ultra"), vm.options.quality) { v ->
            vm.updateOptions { it.copy(quality = v) }
        }
        Dropdown("Face restoration", listOf("None", "Cinematic (clarity + smooth)", "GFPGAN", "CodeFormer"), vm.options.enhancer) { v ->
            vm.updateOptions { it.copy(enhancer = v) }
        }
        Text(
            if (vm.options.enhancer == "None")
                "Smart Boost is ON — lightweight detail, colour and temporal-safe finishing without heavy AI restoration."
            else
                "Smart Boost remains active; ${vm.options.enhancer} adds heavier face restoration on top.",
            color = Muted, fontSize = 11.5.sp, lineHeight = 16.sp
        )
        Dropdown("Face mode", listOf("1 face (fastest)", "2 faces", "All detected faces"), vm.options.faceMode) { v ->
            vm.updateOptions { it.copy(faceMode = v) }
        }
        Text("Farm across Spaces", fontWeight = FontWeight.SemiBold, color = Slate, fontSize = 12.5.sp)
        Dropdown(
            "Split",
            listOf("Off") + (2..10).map { "$it parts" },
            vm.options.splitParts
        ) { v -> vm.updateOptions { it.copy(splitParts = v) } }
        if (vm.options.splitParts != "Off") {
            if (vm.farmPercent > 0) {
                LinearProgressIndicator(
                    progress = { (vm.farmPercent / 100f).coerceIn(0f, 1f) },
                    modifier = Modifier.fillMaxWidth().height(6.dp).clip(CircleShape),
                    color = Burgundy,
                    trackColor = Color(0xFFE9EDF3)
                )
                Text("${vm.farmPhase}  ${vm.farmPercent}%", color = Muted, fontSize = 11.5.sp)
            }
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                listOf(true to "Same settings", false to "Per-part faces").forEach { (shared, label) ->
                    val on = vm.options.splitSharedSettings == shared
                    Box(
                        Modifier
                            .clip(RoundedCornerShape(20.dp))
                            .background(if (on) Color(0xFF15152A) else Color(0xFFF3F5F8))
                            .clickable { vm.updateOptions { it.copy(splitSharedSettings = shared) } }
                            .padding(horizontal = 12.dp, vertical = 7.dp)
                    ) {
                        Text(label, color = if (on) Color.White else Slate, fontSize = 11.5.sp, fontWeight = FontWeight.SemiBold)
                    }
                }
            }
            if (!vm.options.splitSharedSettings) {
                val n = vm.options.splitParts.substringBefore(" ").toIntOrNull()?.coerceIn(2, 10) ?: 0
                val modes = listOf("1 face (fastest)", "2 faces", "All detected faces")
                for (i in 0 until n) {
                    Dropdown("Part ${i + 1}", modes, vm.splitFaceModeAt(i)) { v -> vm.setSplitFaceMode(i, v) }
                }
            }
            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text("Auto-merge", color = Slate, fontSize = 12.5.sp, fontWeight = FontWeight.SemiBold)
                Switch(
                    checked = vm.options.autoMergeSplits,
                    onCheckedChange = { on -> vm.updateOptions { it.copy(autoMergeSplits = on) } }
                )
            }
            Button(
                onClick = { vm.mergeSplitGroup() },
                modifier = Modifier.fillMaxWidth(),
                colors = ButtonDefaults.buttonColors(containerColor = Burgundy),
                shape = RoundedCornerShape(14.dp)
            ) {
                Text("Merge parts now", fontWeight = FontWeight.Bold)
            }
        }

        DisclosureRow(
            label = if (advancedOpen) "Hide advanced controls" else "Advanced controls",
            open = advancedOpen,
            onClick = toggleAdvanced
        )

        if (advancedOpen) {
            Dropdown("Duration", listOf(10, 20, 30, 60, 90, 120, 150, 180, 240, 300, 360).map { "$it sec" }, "${vm.options.durationSec} sec") { v ->
                vm.updateOptions { it.copy(durationSec = v.removeSuffix(" sec").toInt()) }
            }
            Dropdown("FPS", listOf(10, 12, 15, 20, 24, 25, 30, 40, 50, 60).map(Int::toString), vm.options.fps.toString()) { v ->
                vm.updateOptions { it.copy(fps = v.toInt()) }
            }
            Dropdown("Swap every N frames", listOf("Auto", "1", "2", "3", "4", "6", "10", "15"), vm.options.swapEvery) { v ->
                vm.updateOptions { it.copy(swapEvery = v) }
            }
            Dropdown("Re-detect every N swaps", listOf("Auto", "1", "2", "3", "4", "8", "12", "16"), vm.options.detectEvery) { v ->
                vm.updateOptions { it.copy(detectEvery = v) }
            }
            Dropdown("Enhancement scope", listOf("Primary face only (faster)", "All faces"), vm.options.enhanceScope) { v ->
                vm.updateOptions { it.copy(enhanceScope = v) }
            }
            Dropdown("Primary face", listOf("Auto", "Slot 1", "Slot 2", "Slot 3", "Slot 4"), vm.options.primarySlot) { v ->
                vm.updateOptions { it.copy(primarySlot = v) }
            }
            Dropdown("Motion smoothing", listOf("Off", "Fast (blend)", "Quality (motion-compensated)"), vm.options.smoothMotion) { v ->
                vm.updateOptions { it.copy(smoothMotion = v) }
            }
            Dropdown("Compute", listOf("CPU only", "Auto / dedicated GPU"), vm.options.deviceMode) { v ->
                vm.updateOptions { it.copy(deviceMode = v) }
            }

            Text("Trim range", fontWeight = FontWeight.SemiBold, color = Slate, fontSize = 12.5.sp)
            Text("Start ${vm.options.trimStart}%", color = Muted, fontSize = 11.5.sp)
            Slider(
                value = vm.options.trimStart.toFloat(),
                onValueChange = { value -> vm.updateOptions { o -> o.copy(trimStart = value.toInt().coerceIn(0, 99)) } },
                valueRange = 0f..99f
            )
            Text("End ${vm.options.trimEnd}%", color = Muted, fontSize = 11.5.sp)
            Slider(
                value = vm.options.trimEnd.toFloat(),
                onValueChange = { value -> vm.updateOptions { o -> o.copy(trimEnd = value.toInt().coerceIn(1, 100)) } },
                valueRange = 1f..100f
            )

            OutlinedTextField(
                value = vm.options.password,
                onValueChange = { v -> vm.updateOptions { it.copy(password = v) } },
                modifier = Modifier.fillMaxWidth(),
                label = { Text("Optional encrypted ZIP password") },
                singleLine = true,
                shape = RoundedCornerShape(12.dp)
            )
            Text(
                "With a password set, the Space returns an encrypted ZIP instead of a plain MP4. The password is never stored.",
                color = Muted,
                fontSize = 11.5.sp
            )
            GhostButton("Reset to recommended") { vm.resetOptions() }

            SubHeading("Save location")
            Row(
                Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(11.dp))
                    .background(SoftBlue)
                    .padding(horizontal = 12.dp, vertical = 10.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    vm.outputFolderLabel(),
                    Modifier.weight(1f),
                    fontSize = 12.sp,
                    fontWeight = FontWeight.Medium,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )
            }
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
                GhostButton("Choose folder…", Modifier.weight(1f)) { outputFolderPicker.launch(null) }
                if (vm.outputTreeUri != null) {
                    GhostButton("Use default", Modifier.weight(1f), danger = true) { vm.setOutputTree(null) }
                }
            }
            Text(
                "Finished videos and images save here instead of Movies/Phoenix and Pictures/Phoenix. " +
                    "If the chosen folder ever becomes unavailable, results fall back to the default " +
                    "location automatically rather than failing the job.",
                color = Muted,
                fontSize = 11.5.sp
            )
        }
    }
}

@Composable
private fun ImageWorkspace(
    vm: PhoenixViewModel,
    picker: ActivityResultLauncher<Array<String>>,
    facePickers: List<ActivityResultLauncher<Array<String>>>
) {
    StepCard("01", "Target image", "The photo whose faces will be replaced.") {
        MediaDropZone(
            uri = vm.selectedImage,
            isVideo = false,
            emptyTitle = "Select an image",
            emptyHint = "JPG / PNG / WEBP / HEIC",
            onPick = { picker.launch(arrayOf("image/*")) },
            onClear = { vm.selectImage(null) }
        )
        ActionButton(
            text = if (vm.detectingImage) "Detecting\u2026" else "Detect faces",
            enabled = vm.selectedImage != null && !vm.detectingImage,
            busy = vm.detectingImage,
            onClick = { vm.detectImage() }
        )
        ErrorPanel(vm)
        DetectedFacesStrip(vm)
    }

    StepCard("02", "Replacement faces", "Tap a tile to upload a source face, or capture one that was detected.") {
        FaceSlotGrid(vm, facePickers)
    }

    StepCard("03", "Image output", "Quality preset and the last saved result.") {
        Dropdown("Quality", listOf("Fast", "Balanced", "Best", "Ultra"), vm.imageQuality) { vm.updateImageQuality(it) }
        val output = vm.imageResultUri
        if (output != null) PreviewPanel(output, "Saved to Pictures/Phoenix")
        else EmptyNote("No output yet \u2014 run a swap to produce one.")
    }
}

/* ----------------------------------------------------------------------- */
/* Media tiles                                                              */
/* ----------------------------------------------------------------------- */

@Composable
private fun MediaDropZone(
    uri: Uri?,
    isVideo: Boolean,
    framePercent: Int = 0,
    emptyTitle: String,
    emptyHint: String,
    onPick: () -> Unit,
    onClear: () -> Unit,
    belowPreview: (@Composable ColumnScope.() -> Unit)? = null
) {
    val thumb by rememberThumb(uri, isVideo, framePercent)

    if (uri == null) {
        Column(
            Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(16.dp))
                .background(SoftBlue)
                .border(1.dp, Color(0xFFBFD6EC), RoundedCornerShape(16.dp))
                .clickable { onPick() }
                .padding(vertical = 26.dp, horizontal = 16.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            Text(if (isVideo) "\uD83C\uDFAC" else "\uD83D\uDDBC", fontSize = 30.sp)
            Text(emptyTitle, fontWeight = FontWeight.Bold, color = Slate, fontSize = 14.sp)
            Text(emptyHint, color = Muted, fontSize = 11.5.sp, textAlign = TextAlign.Center)
        }
        return
    }

    Column(
        Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(16.dp))
            .background(Color(0xFFF7F9FC))
            .border(1.dp, Line, RoundedCornerShape(16.dp))
    ) {
        Box(
            Modifier
                .fillMaxWidth()
                .height(190.dp)
                .background(Slate),
            contentAlignment = Alignment.Center
        ) {
            val bmp = thumb.bitmap
            when {
                bmp != null -> Image(
                    bitmap = bmp,
                    contentDescription = emptyTitle,
                    modifier = Modifier.fillMaxSize(),
                    contentScale = ContentScale.Crop
                )
                thumb.loading -> CircularProgressIndicator(color = Gold, strokeWidth = 2.dp)
                else -> Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text(if (isVideo) "\uD83C\uDFAC" else "\uD83D\uDDBC", fontSize = 26.sp)
                    Text(
                        thumb.error ?: "Preview unavailable",
                        color = Color(0xFFB9C7D8),
                        fontSize = 11.sp,
                        textAlign = TextAlign.Center,
                        modifier = Modifier.padding(horizontal = 20.dp, vertical = 4.dp)
                    )
                }
            }

            if (isVideo && thumb.hasImage) {
                Box(
                    Modifier
                        .align(Alignment.BottomStart)
                        .padding(9.dp)
                        .clip(RoundedCornerShape(7.dp))
                        .background(Color(0xCC071A35))
                        .padding(horizontal = 8.dp, vertical = 4.dp)
                ) {
                    Text(
                        "FRAME ${framePercent}%",
                        color = Gold,
                        fontSize = 9.5.sp,
                        fontWeight = FontWeight.Black
                    )
                }
            }
        }

        if (belowPreview != null) {
            Column(
                Modifier
                    .fillMaxWidth()
                    .background(Color.White)
                    .padding(horizontal = 12.dp, vertical = 8.dp),
                verticalArrangement = Arrangement.spacedBy(6.dp),
                content = belowPreview
            )
        }

        Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(7.dp)) {
            Text(
                thumb.displayName ?: uri.lastPathSegment ?: "Selected file",
                fontWeight = FontWeight.Bold,
                color = Ink,
                fontSize = 13.sp,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
            MetaChips(thumb)
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
                GhostButton("Replace", Modifier.weight(1f)) { onPick() }
                GhostButton("Clear", Modifier.weight(1f), danger = true) { onClear() }
            }
        }
    }
}

@Composable
private fun MetaChips(thumb: ThumbState) {
    val chips = listOfNotNull(thumb.durationLabel(), thumb.dimensionLabel(), thumb.sizeLabel())
    if (chips.isEmpty()) return
    Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
        chips.forEach { chip ->
            Box(
                Modifier
                    .clip(RoundedCornerShape(7.dp))
                    .background(SoftGold)
                    .padding(horizontal = 8.dp, vertical = 4.dp)
            ) {
                Text(chip, color = Color(0xFF7A6412), fontSize = 10.5.sp, fontWeight = FontWeight.Bold)
            }
        }
    }
}

@Composable
private fun FaceSlotGrid(
    vm: PhoenixViewModel,
    pickers: List<ActivityResultLauncher<Array<String>>>
) {
    val faces = listOf(vm.face1, vm.face2, vm.face3, vm.face4)
    Column(verticalArrangement = Arrangement.spacedBy(9.dp)) {
        for (row in 0..1) {
            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(9.dp)
            ) {
                for (col in 0..1) {
                    val index = row * 2 + col
                    FaceSlotTile(
                        slot = index + 1,
                        uri = faces[index],
                        modifier = Modifier.weight(1f),
                        onPick = { pickers[index].launch(arrayOf("image/*")) },
                        onClear = { vm.clearFace(index + 1) }
                    )
                }
            }
        }
        Text(
            "${faces.count { it != null }} of 4 face slots filled",
            color = Muted,
            fontSize = 11.5.sp
        )
    }
}

@Composable
private fun FaceSlotTile(
    slot: Int,
    uri: Uri?,
    modifier: Modifier = Modifier,
    onPick: () -> Unit,
    onClear: () -> Unit
) {
    val thumb by rememberThumb(uri, isVideo = false, maxPx = 320)
    val filled = uri != null

    Column(
        modifier
            .clip(RoundedCornerShape(14.dp))
            .background(if (filled) Color.White else Color(0xFFF5F8FC))
            .border(
                1.dp,
                if (filled) Burgundy.copy(alpha = 0.45f) else Line,
                RoundedCornerShape(14.dp)
            )
    ) {
        Box(
            Modifier
                .fillMaxWidth()
                .height(112.dp)
                .background(if (filled) Slate else Color(0xFFEBF1F8))
                .clickable { onPick() },
            contentAlignment = Alignment.Center
        ) {
            val bmp = thumb.bitmap
            when {
                bmp != null -> Image(
                    bitmap = bmp,
                    contentDescription = "Face $slot",
                    modifier = Modifier.fillMaxSize(),
                    contentScale = ContentScale.Crop
                )
                thumb.loading -> CircularProgressIndicator(
                    color = Gold,
                    strokeWidth = 2.dp,
                    modifier = Modifier.size(22.dp)
                )
                filled -> Text("\u26A0", fontSize = 20.sp)
                else -> Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text("\uFF0B", color = Muted, fontSize = 24.sp)
                    Text("Upload", color = Muted, fontSize = 10.5.sp)
                }
            }

            Box(
                Modifier
                    .align(Alignment.TopStart)
                    .padding(6.dp)
                    .clip(CircleShape)
                    .background(if (filled) Burgundy else Color(0xFFCBD8E6))
                    .padding(horizontal = 7.dp, vertical = 2.dp)
            ) {
                Text(
                    "#$slot",
                    color = if (filled) Color.White else Slate,
                    fontSize = 10.sp,
                    fontWeight = FontWeight.Black
                )
            }
        }

        Row(
            Modifier
                .fillMaxWidth()
                .padding(horizontal = 8.dp, vertical = 6.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(
                if (filled) (thumb.displayName ?: "Face selected") else "Empty",
                color = if (filled) Ink else Muted,
                fontSize = 10.5.sp,
                fontWeight = if (filled) FontWeight.SemiBold else FontWeight.Normal,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
                modifier = Modifier.weight(1f)
            )
            if (filled) {
                Text(
                    "\u2715",
                    color = Burgundy,
                    fontSize = 13.sp,
                    fontWeight = FontWeight.Black,
                    modifier = Modifier
                        .clip(CircleShape)
                        .clickable { onClear() }
                        .padding(horizontal = 6.dp, vertical = 1.dp)
                )
            }
        }
    }
}

@Composable
private fun DetectedFacesStrip(vm: PhoenixViewModel) {
    val detected = vm.detectedFaces
    val found = detected.count { it != null }
    var selected by remember { mutableStateOf<Int?>(null) }

    // Clear a stale selection if re-detection emptied that position.
    if (selected != null && detected.getOrNull(selected!!) == null) selected = null

    // These four tiles are always on screen, so it is obvious WHERE detected
    // faces land even before detection has been run (or when it fails).
    Column(verticalArrangement = Arrangement.spacedBy(7.dp)) {
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
            Text(
                "Detected faces",
                color = Slate,
                fontWeight = FontWeight.Bold,
                fontSize = 12.5.sp,
                modifier = Modifier.weight(1f)
            )
            Text("$found / 4 found", color = Muted, fontSize = 11.sp)
        }
        Text(
            when {
                found == 0 -> "Faces found in the target appear here. Tap one, then choose which replacement slot it should fill."
                selected == null -> "Tap a face to select it, then choose a slot. Scrub to another frame and detect again to fill the remaining slots \u2014 both people never have to appear in the same frame."
                else -> "Now choose the replacement slot for detected face #${selected!! + 1}."
            },
            color = Muted,
            fontSize = 11.sp
        )
        Row(
            Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            (0..3).forEach { index ->
                DetectedFaceChip(
                    uri = detected.getOrNull(index),
                    index = index,
                    selected = selected == index,
                    modifier = Modifier.weight(1f)
                ) { selected = if (selected == index) null else index }
            }
        }

        // Destination picker — only shown once a detected face is selected, so
        // the strip stays uncluttered until there is actually a choice to make.
        if (selected != null) {
            Row(
                Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                (1..4).forEach { slot ->
                    Box(
                        Modifier
                            .weight(1f)
                            .height(38.dp)
                            .clip(RoundedCornerShape(10.dp))
                            .background(Burgundy)
                            .clickable {
                                vm.captureDetectedFaceTo(selected!!, slot)
                                selected = null
                            },
                        contentAlignment = Alignment.Center
                    ) {
                        Text(
                            "\u2192 slot $slot",
                            color = Color.White,
                            fontSize = 11.sp,
                            fontWeight = FontWeight.Bold
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun DetectedFaceChip(
    uri: Uri?,
    index: Int,
    selected: Boolean,
    modifier: Modifier = Modifier,
    onSelect: () -> Unit
) {
    val thumb by rememberThumb(uri, isVideo = false, maxPx = 240)
    val filled = uri != null

    Column(
        modifier = modifier,
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(4.dp)
    ) {
        Box(
            Modifier
                .fillMaxWidth()
                .height(72.dp)
                .clip(RoundedCornerShape(11.dp))
                .background(if (filled) Slate else Color(0xFFF0F4F9))
                .border(
                    if (selected) 2.dp else 1.dp,
                    when {
                        selected -> Gold
                        filled -> Burgundy.copy(alpha = 0.45f)
                        else -> Line
                    },
                    RoundedCornerShape(11.dp)
                )
                .clickable(enabled = filled) { onSelect() },
            contentAlignment = Alignment.Center
        ) {
            val bmp = thumb.bitmap
            when {
                bmp != null -> Image(
                    bitmap = bmp,
                    contentDescription = "Detected face ${index + 1}",
                    modifier = Modifier.fillMaxSize(),
                    contentScale = ContentScale.Crop
                )
                thumb.loading -> CircularProgressIndicator(
                    color = Gold,
                    strokeWidth = 2.dp,
                    modifier = Modifier.size(18.dp)
                )
                else -> Text("\uD83D\uDC64", fontSize = 19.sp, color = Muted)
            }
        }
        Text(
            when {
                selected -> "selected"
                filled -> "face #${index + 1}"
                else -> "empty"
            },
            color = if (selected) Gold else if (filled) Burgundy else Muted,
            fontSize = 9.5.sp,
            fontWeight = if (filled || selected) FontWeight.Bold else FontWeight.Normal
        )
    }
}

@Composable
private fun ErrorPanel(vm: PhoenixViewModel) {
    val error = vm.lastError ?: return
    Column(
        Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(13.dp))
            .background(Color(0xFFFBF0F3))
            .border(1.dp, Burgundy.copy(alpha = 0.3f), RoundedCornerShape(13.dp))
            .padding(13.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
            Text("\u26A0  Last error", color = Burgundy, fontWeight = FontWeight.Black, fontSize = 11.sp, modifier = Modifier.weight(1f))
            Text(
                "\u2715",
                color = Muted,
                fontSize = 13.sp,
                modifier = Modifier.clip(CircleShape).clickable { vm.clearError() }.padding(4.dp)
            )
        }
        Text(error, color = Ink, fontSize = 11.5.sp)

        Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
            GhostButton(
                if (vm.diagnosing) "Checking\u2026" else "Diagnose Space",
                Modifier.weight(1f)
            ) { vm.diagnoseSpace() }
        }

        if (vm.spaceEndpoints.isNotEmpty()) {
            Text(
                "Routes published by ${vm.spaceLabel}:",
                color = Slate,
                fontWeight = FontWeight.SemiBold,
                fontSize = 11.sp
            )
            vm.requiredEndpoints.forEach { route ->
                val present = vm.spaceEndpoints.contains(route)
                val spaceSig = vm.spaceSignatures[route]
                val clientSig = vm.clientSignatures[route]
                Column(Modifier.padding(bottom = 5.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text(
                            if (present) "\u2713" else "\u2715",
                            color = if (present) Color(0xFF15803D) else Burgundy,
                            fontWeight = FontWeight.Black,
                            fontSize = 11.sp
                        )
                        Spacer(Modifier.width(7.dp))
                        Text(
                            route,
                            color = if (present) Slate else Burgundy,
                            fontSize = 11.sp,
                            fontWeight = FontWeight.SemiBold
                        )
                    }
                    if (present && spaceSig != null) {
                        Text(
                            "Space expects: $spaceSig",
                            color = Muted,
                            fontSize = 10.sp,
                            modifier = Modifier.padding(start = 18.dp)
                        )
                        if (clientSig != null) {
                            Text(
                                "App sends: $clientSig",
                                color = Muted,
                                fontSize = 10.sp,
                                modifier = Modifier.padding(start = 18.dp)
                            )
                        }
                    }
                }
            }
            val extras = vm.spaceEndpoints.filterNot { vm.requiredEndpoints.contains(it) }
            if (extras.isNotEmpty()) {
                Text(
                    "Also available: ${extras.joinToString(", ")}",
                    color = Muted,
                    fontSize = 10.5.sp
                )
            }
            if (vm.missingEndpoints.isNotEmpty()) {
                Text(
                    "The missing routes must be added to the Space's Gradio app with a matching api_name= before this feature can work.",
                    color = Burgundy,
                    fontSize = 11.sp,
                    fontWeight = FontWeight.Medium
                )
            }
        }
    }
}

@Composable
private fun PreviewPanel(uri: Uri, caption: String) {
    val thumb by rememberThumb(uri, isVideo = false, maxPx = 800)
    Column(verticalArrangement = Arrangement.spacedBy(5.dp)) {
        Box(
            Modifier
                .fillMaxWidth()
                .height(180.dp)
                .clip(RoundedCornerShape(13.dp))
                .background(Slate),
            contentAlignment = Alignment.Center
        ) {
            val bmp = thumb.bitmap
            when {
                bmp != null -> Image(
                    bitmap = bmp,
                    contentDescription = caption,
                    modifier = Modifier.fillMaxSize(),
                    contentScale = ContentScale.Fit
                )
                thumb.loading -> CircularProgressIndicator(color = Gold, strokeWidth = 2.dp)
                else -> Text(
                    thumb.error ?: "Preview unavailable",
                    color = Color(0xFFB9C7D8),
                    fontSize = 11.5.sp
                )
            }
        }
        Text(caption, color = Muted, fontSize = 11.sp)
    }
}

/* ----------------------------------------------------------------------- */
/* History                                                                  */
/* ----------------------------------------------------------------------- */

@Composable
private fun HistoryVideoThumb(uriString: String) {
    // Do not decode History videos with MediaMetadataRetriever. After a few
    // saved jobs that native decoder OOMs / SIGSEGVs and Android shows
    // "keeps stopping". A placeholder is enough; Download still works.
    Box(
        Modifier
            .fillMaxWidth()
            .height(72.dp)
            .clip(RoundedCornerShape(11.dp))
            .background(Slate),
        contentAlignment = Alignment.Center
    ) {
        Text("Video ready · tap Download", color = Color.White, fontSize = 12.sp, fontWeight = FontWeight.Bold)
    }
}

@Composable
private fun HistoryWorkspace(vm: PhoenixViewModel) {
    StepCard("\u2261", "History", "Every Phoenix job is tracked, including jobs running on multiple Spaces at once.") {
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
            GhostButton("Refresh", Modifier.weight(1f)) { vm.refreshHistory() }
            GhostButton("Merge parts", Modifier.weight(1f)) { vm.mergeSplitGroup() }
            GhostButton("Clear all", Modifier.weight(1f), danger = true) { vm.clearHistory() }
        }
        if (vm.activeJobCount > 0) {
            Text(
                "${vm.activeJobCount}/${PhoenixViewModel.MAX_SPACES} Phoenix jobs currently monitored",
                color = Burgundy,
                fontWeight = FontWeight.Bold,
                fontSize = 12.sp
            )
        }
        if (vm.history.isEmpty()) EmptyNote("No Phoenix jobs yet.")

        vm.history.forEach { item ->
            val done = item.status == "Done"
            val failed = item.status == "Failed"
            val onServer = item.status in setOf("Waiting for network", "Ready on server", "On server", "Reconnecting")
            Card(
                shape = RoundedCornerShape(14.dp),
                colors = CardDefaults.cardColors(
                    containerColor = when {
                        failed -> Color(0xFFFBF0F3)
                        onServer -> Color(0xFFFFF6E8)
                        done -> Color.White
                        else -> SoftBlue
                    }
                ),
                border = BorderStroke(1.dp, Line)
            ) {
                Column(
                    Modifier.fillMaxWidth().padding(12.dp),
                    verticalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    if (done && !item.uri.isNullOrBlank()) {
                        HistoryVideoThumb(item.uri)
                    }
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Column(Modifier.weight(1f)) {
                            Text(
                                item.name,
                                fontWeight = FontWeight.Bold,
                                fontSize = 13.sp,
                                maxLines = 2,
                                overflow = TextOverflow.Ellipsis
                            )
                            Text(
                                "${item.kind} \u00B7 ${item.spaceLabel}",
                                color = Muted,
                                fontSize = 11.sp
                            )
                        }
                        Box(
                            Modifier
                                .clip(RoundedCornerShape(7.dp))
                                .background(
                                    when {
                                        failed -> Color(0xFFF3D9E1)
                                        onServer -> Color(0xFFF8E7C9)
                                        done -> Color(0xFFDCF3E4)
                                        else -> Color(0xFFD9E8F7)
                                    }
                                )
                                .padding(horizontal = 8.dp, vertical = 4.dp)
                        ) {
                            Text(
                                item.status.uppercase(),
                                color = if (failed) Burgundy else Slate,
                                fontSize = 9.5.sp,
                                fontWeight = FontWeight.Black
                            )
                        }
                    }

                    if (item.status !in setOf("Done", "Failed", "Monitoring stopped")) {
                        LinearProgressIndicator(
                            progress = { item.progress.coerceIn(0f, 1f) },
                            modifier = Modifier.fillMaxWidth().height(6.dp).clip(CircleShape),
                            color = Burgundy,
                            trackColor = Color(0xFFDFE8F2)
                        )
                        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                            Text("${(item.progress * 100).toInt()}%", color = Muted, fontSize = 10.5.sp)
                            Text(item.eta, color = Muted, fontSize = 10.5.sp)
                        }
                    }

                    Row(
                        Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Text(
                            DateFormat.getDateTimeInstance(DateFormat.SHORT, DateFormat.SHORT)
                                .format(Date(item.createdAt)),
                            color = Muted,
                            fontSize = 10.5.sp
                        )
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            val showResume = !item.remoteJobId.isNullOrBlank() && !done && item.uri.isNullOrBlank() && item.status != "Replaced"
                            val showRedo = item.partCount > 1 && !item.groupId.isNullOrBlank() && item.status !in setOf("Done", "Replaced")
                            when {
                                showResume || showRedo -> {
                                    if (showResume) {
                                        TextButton({ vm.retryJob(item.id) }) {
                                            Text("Resume", color = Burgundy, fontWeight = FontWeight.Bold, fontSize = 12.sp)
                                        }
                                    }
                                    if (showRedo) {
                                        TextButton({ vm.redoSplitPart(item.id) }) {
                                            Text("Redo part", color = Burgundy, fontWeight = FontWeight.Bold, fontSize = 12.sp)
                                        }
                                    }
                                }
                                item.status !in setOf("Done", "Failed", "Monitoring stopped", "Replaced") && item.remoteJobId != null ->
                                    TextButton({ vm.cancelJob(item.id) }) {
                                        Text("Stop", color = Burgundy, fontSize = 12.sp)
                                    }
                                done && !item.uri.isNullOrBlank() -> {
                                    TextButton({ vm.downloadToDownloads(item) }) {
                                        Text("Download", color = Burgundy, fontWeight = FontWeight.Bold, fontSize = 12.sp)
                                    }
                                    TextButton({ vm.removeHistory(item.id) }) {
                                        Text("Delete", color = Muted, fontSize = 12.sp)
                                    }
                                }
                                else ->
                                    TextButton({ vm.removeHistory(item.id) }) {
                                        Text("Delete", color = Muted, fontSize = 12.sp)
                                    }
                            }
                        }
                    }
                }
            }
        }
    }
}

/* ----------------------------------------------------------------------- */
/* Always-visible processing Space selector                                 */
/* ----------------------------------------------------------------------- */

@Composable
private fun TopSpaceSelectorBar(vm: PhoenixViewModel) {
    val isLandscape = LocalConfiguration.current.orientation == Configuration.ORIENTATION_LANDSCAPE
    Surface(modifier = Modifier.fillMaxWidth(), color = Page) {
        Column(Modifier.padding(horizontal = 16.dp, vertical = 2.dp)) {
            // In landscape the pinned header competes directly with the work
            // area, so collapse to a single row: label + chips inline. The
            // "PROCESSING SPACE" caption and "Choose before Start" hint are
            // both dropped - they are guidance, not state.
            if (isLandscape) {
                Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                    val current = vm.spaces.getOrNull(vm.selectedSpaceIndex)
                    Text(
                        current?.label?.ifBlank { "Space ${vm.selectedSpaceIndex + 1}" } ?: "Space ${vm.selectedSpaceIndex + 1}",
                        color = Ink, fontSize = 12.sp, fontWeight = FontWeight.Bold,
                        maxLines = 1, overflow = TextOverflow.Ellipsis,
                        modifier = Modifier.widthIn(max = 120.dp)
                    )
                    Spacer(Modifier.width(10.dp))
                    SpaceChipRow(vm)
                }
            } else {
                Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                    Column(Modifier.weight(1f)) {
                        Text("PROCESSING SPACE", color = Burgundy, fontSize = 9.5.sp, fontWeight = FontWeight.Black, letterSpacing = 1.sp)
                        val current = vm.spaces.getOrNull(vm.selectedSpaceIndex)
                        Text(current?.label?.ifBlank { "Space ${vm.selectedSpaceIndex + 1}" } ?: "Space ${vm.selectedSpaceIndex + 1}", color = Ink, fontSize = 14.sp, fontWeight = FontWeight.Bold, maxLines = 1, overflow = TextOverflow.Ellipsis)
                    }
                    Text("Choose before Start", color = Muted, fontSize = 9.5.sp, fontWeight = FontWeight.SemiBold)
                }
                Spacer(Modifier.height(6.dp))
                SpaceChipRow(vm)
            }
        }
    }
}

/* ----------------------------------------------------------------------- */
/* Connection (up to MAX_SPACES Spaces)                                     */
/* ----------------------------------------------------------------------- */

@Composable
private fun ConnectionCard(
    vm: PhoenixViewModel,
    open: Boolean,
    toggle: () -> Unit,
    token: String,
    setToken: (String) -> Unit,
    clearToken: () -> Unit
) {
    StepCard(
        "\u2699",
        "Connection",
        "Up to ${PhoenixViewModel.MAX_SPACES} independent Hugging Face Spaces. Tap a number to route the next job to it \u2014 running jobs continue in parallel."
    ) {
        val current = vm.spaces.getOrNull(vm.selectedSpaceIndex)
        Row(
            Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(11.dp))
                .background(SoftBlue)
                .padding(horizontal = 12.dp, vertical = 10.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(Modifier.weight(1f)) {
                Text(
                    current?.label?.ifBlank { "Unnamed Space" } ?: "Space ${vm.selectedSpaceIndex + 1}",
                    fontWeight = FontWeight.Bold, fontSize = 13.sp
                )
                Text(
                    current?.url?.removePrefix("https://")?.ifBlank { "Not configured" } ?: "Not configured",
                    color = Muted, fontSize = 10.5.sp, maxLines = 1, overflow = TextOverflow.Ellipsis
                )
            }
            Text("Targeted next", color = Burgundy, fontWeight = FontWeight.Bold, fontSize = 10.sp)
        }

        Row(
            Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(10.dp))
                .background(if (vm.hfTokenSet) Color(0xFFE9F7EE) else SoftGold)
                .padding(horizontal = 11.dp, vertical = 8.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(if (vm.hfTokenSet) "\uD83D\uDD12" else "\u26A0", fontSize = 13.sp)
            Spacer(Modifier.width(8.dp))
            Text(
                if (vm.hfTokenSet) "Token stored in Android Keystore"
                else "No token saved \u2014 private Spaces will reject requests",
                color = Slate,
                fontSize = 11.5.sp,
                fontWeight = FontWeight.Medium
            )
        }

        DisclosureRow(
            label = if (open) "Hide connection settings" else "Edit Space settings & token",
            open = open,
            onClick = toggle
        )

        if (open) {
            vm.spaces.forEachIndexed { index, space ->
                SubHeading("Space ${index + 1}")
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
                    OutlinedTextField(
                        value = space.label,
                        onValueChange = { vm.updateSpaceLabel(index, it) },
                        modifier = Modifier.weight(1f),
                        label = { Text("Name") },
                        singleLine = true,
                        shape = RoundedCornerShape(12.dp)
                    )
                }
                OutlinedTextField(
                    value = space.url,
                    onValueChange = { vm.updateSpaceUrl(index, it) },
                    modifier = Modifier.fillMaxWidth(),
                    label = { Text("Space ${index + 1} URL") },
                    placeholder = { Text("https://your-space.hf.space") },
                    singleLine = true,
                    shape = RoundedCornerShape(12.dp)
                )
            }

            SubHeading("Shared API settings")
            OutlinedTextField(
                value = vm.endpoint,
                onValueChange = { vm.endpoint = it },
                modifier = Modifier.fillMaxWidth(),
                label = { Text("Submit endpoint") },
                singleLine = true,
                shape = RoundedCornerShape(12.dp)
            )
            OutlinedTextField(
                value = token,
                onValueChange = setToken,
                modifier = Modifier.fillMaxWidth(),
                label = { Text("Hugging Face token") },
                placeholder = { Text("Paste, then Save") },
                singleLine = true,
                shape = RoundedCornerShape(12.dp)
            )
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
                ActionButton("Save token", Modifier.weight(1f)) { vm.saveToken(token); clearToken() }
                GhostButton("Test Space", Modifier.weight(1f)) { vm.discoverSpace() }
            }
            GhostButton(if (vm.diagnosing) "Checking routes\u2026" else "Diagnose API routes") { vm.diagnoseSpace() }
            if (vm.hfTokenSet) {
                GhostButton("Remove stored token", danger = true) { vm.saveToken(""); clearToken() }
            }
            Text(
                "One HF token is shared across every configured Space here. To run jobs in parallel: tap a " +
                    "number above, start a job, tap a different number, start another \u2014 up to ${PhoenixViewModel.MAX_SPACES} " +
                    "at once. History records which Space each job went to.",
                color = Muted,
                fontSize = 11.5.sp
            )
        }
    }
}

/* ----------------------------------------------------------------------- */
/* Update Space code — push one Phoenix package to many Spaces              */
/* ----------------------------------------------------------------------- */

@Composable
private fun DeployCard(
    vm: PhoenixViewModel,
    open: Boolean,
    toggle: () -> Unit,
    pickPackage: () -> Unit
) {
    var deployToken by remember { mutableStateOf("") }
    var confirm by remember { mutableStateOf(false) }

    StepCard(
        "\u21EA",
        "Update Space code",
        "Push a new Phoenix Space package (e.g. v11.2.80) to many Spaces at once, instead of uploading the files into each Space by hand."
    ) {
        DisclosureRow(
            label = if (open) "Hide Space update" else "Update Spaces from a package",
            open = open,
            onClick = toggle
        )
        if (!open) return@StepCard

        GhostButton("Choose package (.zip, or its 8 files)\u2026") { pickPackage() }
        val pkg = vm.deployPackage
        Row(
            Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(11.dp))
                .background(if (pkg != null) Color(0xFFE9F7EE) else SoftGold)
                .padding(horizontal = 12.dp, vertical = 10.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(if (pkg != null) "\u2713" else "\u2139", fontSize = 13.sp)
            Spacer(Modifier.width(8.dp))
            Text(vm.deployNote, color = Slate, fontSize = 11.5.sp, fontWeight = FontWeight.Medium)
        }

        Row(
            Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(Modifier.weight(1f)) {
                Text("Also update README.md", color = Slate, fontSize = 12.5.sp, fontWeight = FontWeight.SemiBold)
                Text("The Space settings header. Each Space keeps its own title.", color = Muted, fontSize = 11.sp)
            }
            Switch(
                checked = vm.deployIncludeReadme,
                onCheckedChange = { vm.deployIncludeReadme = it },
                enabled = !vm.deploying
            )
        }

        SubHeading("Spaces to update")
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
            GhostButton("All free Spaces", Modifier.weight(1f)) { vm.setAllDeploySpaces(true) }
            GhostButton("None", Modifier.weight(1f)) { vm.setAllDeploySpaces(false) }
        }
        vm.spaces.forEachIndexed { index, space ->
            if (space.url.isBlank()) return@forEachIndexed
            val busy = vm.isSpaceBusy(space.url)
            Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                Checkbox(
                    checked = index in vm.deploySelected,
                    onCheckedChange = { vm.toggleDeploySpace(index) },
                    enabled = !vm.deploying
                )
                Column(Modifier.weight(1f)) {
                    Text(
                        "${index + 1}. ${space.label.ifBlank { "Space ${index + 1}" }}" + if (busy) "  \u00B7 job running" else "",
                        color = Ink, fontSize = 12.5.sp, fontWeight = FontWeight.SemiBold,
                        maxLines = 1, overflow = TextOverflow.Ellipsis
                    )
                    vm.deployResults[index]?.let { r ->
                        Text(r, color = if (r.startsWith("\u2717")) Burgundy else Muted, fontSize = 11.sp)
                    }
                    vm.deployStage[index]?.let { st ->
                        Text("Now: $st", color = Slate, fontSize = 11.sp)
                    }
                }
            }
        }

        SubHeading("Token for updating")
        Text(
            if (vm.deployTokenSet) "\uD83D\uDD12 A separate update token is saved and used for this."
            else "Using the main token. It needs WRITE access to change a Space; a read-only token is refused before anything is sent.",
            color = Muted, fontSize = 11.5.sp
        )
        OutlinedTextField(
            value = deployToken,
            onValueChange = { deployToken = it },
            modifier = Modifier.fillMaxWidth(),
            label = { Text("Update token with write access (optional)") },
            placeholder = { Text("Leave empty to use the main token") },
            singleLine = true,
            visualTransformation = PasswordVisualTransformation(),
            shape = RoundedCornerShape(12.dp)
        )
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
            GhostButton("Save update token", Modifier.weight(1f)) { vm.saveDeployToken(deployToken); deployToken = "" }
            if (vm.deployTokenSet) {
                GhostButton("Remove", Modifier.weight(1f), danger = true) { vm.saveDeployToken(""); deployToken = "" }
            }
        }

        if (vm.deploying) {
            GhostButton("Stop after the Spaces already updating", danger = true) { vm.stopDeploy() }
        } else {
            ActionButton(
                text = "Update ${vm.deploySelected.size} Space${if (vm.deploySelected.size == 1) "" else "s"}",
                enabled = pkg != null && vm.deploySelected.isNotEmpty(),
                onClick = { confirm = true }
            )
        }
        GhostButton("Check Space status") { vm.refreshDeployStatus() }
        Text(
            "Each updated Space restarts and rebuilds (a few minutes). Spaces whose files already match are " +
                "left alone. A Space with a job from this phone is skipped; a job started from the website " +
                "on it would be lost. On a free Hugging Face account only a limited number of Spaces can run " +
                "at once - the rest show a quota error until others are paused.",
            color = Muted,
            fontSize = 11.5.sp
        )
    }

    if (confirm) {
        val pkg = vm.deployPackage
        AlertDialog(
            onDismissRequest = { confirm = false },
            title = { Text("Update ${vm.deploySelected.size} Space(s)?", fontWeight = FontWeight.Bold) },
            text = {
                Text(
                    "Push ${pkg?.version ?: "this package"} (${pkg?.files?.size ?: 0} files) to the ticked Spaces. " +
                        "Each one restarts and rebuilds. Anything running on them from the website stops."
                )
            },
            confirmButton = { Button({ confirm = false; vm.startDeploy() }) { Text("Update") } },
            dismissButton = { TextButton({ confirm = false }) { Text("Cancel") } }
        )
    }
}

/**
 * Modern Space selector — compact numbered pills with live busy indicator.
 * Selected Space gets a filled primary treatment; busy Spaces show a green pulse.
 */
@Composable
private fun SpaceChipRow(vm: PhoenixViewModel) {
    Row(Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        vm.spaces.forEachIndexed { index, space ->
            val selected = vm.selectedSpaceIndex == index
            val configured = space.url.isNotBlank()
            val busy = vm.isSpaceBusy(space.url)
            val bg by animateColorAsState(if (selected) Burgundy else Color.White, animationSpec = tween(180), label = "spaceBg")
            val fg by animateColorAsState(if (selected) Color.White else Ink, animationSpec = tween(180), label = "spaceFg")
            val border = if (selected) Color.Transparent else if (configured) Line else Color(0xFFE8EDF3)
            val statusColor = when { busy -> Color(0xFF34D399); configured -> Color(0xFF94A3B8); else -> Color(0xFFCBD5E1) }
            // Compact pill. The old 72x72 tile repeated "SPACE n" AND the number
            // AND a status word - three ways of saying the same thing, costing
            // ~72dp of permanently-pinned height. The dot carries status, the
            // number carries identity; the selected Space's full name is shown
            // once in the header row above.
            Row(
                Modifier
                    .height(38.dp)
                    .widthIn(min = 54.dp)
                    .clip(RoundedCornerShape(12.dp))
                    .background(bg)
                    .border(1.dp, border, RoundedCornerShape(12.dp))
                    .clickable { vm.selectProcessingSpace(index) }
                    .padding(horizontal = 11.dp),
                horizontalArrangement = Arrangement.Center,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Box(Modifier.size(6.dp).clip(CircleShape).background(statusColor))
                Spacer(Modifier.width(6.dp))
                Text("${index + 1}", color = fg, fontSize = 15.sp, fontWeight = FontWeight.Black)
            }
        }
    }
}

@Composable
private fun PrivacyCard(onClick: () -> Unit) {
    Row(
        Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(18.dp))
            .background(Color.White)
            .border(1.dp, Line, RoundedCornerShape(18.dp))
            .clickable { onClick() }
            .padding(horizontal = 16.dp, vertical = 15.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Box(
            Modifier
                .size(36.dp)
                .clip(RoundedCornerShape(11.dp))
                .background(SoftBlue),
            contentAlignment = Alignment.Center
        ) {
            Text("⛨", fontSize = 16.sp)
        }
        Spacer(Modifier.width(12.dp))
        Column(Modifier.weight(1f)) {
            Text("Privacy & security", fontWeight = FontWeight.Bold, fontSize = 14.sp, color = Ink)
            Text("Where your media and token go", color = Muted, fontSize = 12.sp)
        }
        Text("›", color = Muted, fontSize = 22.sp, fontWeight = FontWeight.Light)
    }
}

/* ----------------------------------------------------------------------- */
/* Bottom action bar - the primary Swap command                             */
/* ----------------------------------------------------------------------- */

@Composable
private fun BottomActionBar(vm: PhoenixViewModel) {
    val blocked = vm.swapBlockReason
    val enabled = blocked == null
    val label = when {
        vm.activeJobCount > 0 && vm.activeTab != "Image" -> "Start another swap"
        vm.activeTab == "Image" -> "Swap faces"
        else -> "Start swap"
    }
    Surface(color = Color.White, shadowElevation = 10.dp, tonalElevation = 2.dp, shape = RoundedCornerShape(topStart = 24.dp, topEnd = 24.dp)) {
        Column(Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 11.dp).navigationBarsPadding(), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            if (vm.running || vm.progress > 0f) {
                LinearProgressIndicator(progress = { vm.progress.coerceIn(0f,1f) }, modifier = Modifier.fillMaxWidth().height(4.dp).clip(CircleShape), color = Burgundy, trackColor = Color(0xFFE9EDF3))
            }
            Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text(blocked ?: if (vm.activeJobCount > 0) "${vm.activeJobCount} job${if (vm.activeJobCount == 1) "" else "s"} running" else "Ready to process", color = if (enabled) Ink else Burgundy, fontSize = 11.5.sp, fontWeight = FontWeight.Bold, maxLines = 1, overflow = TextOverflow.Ellipsis)
                    Text(if (enabled) "Using ${vm.spaceLabel}" else "Complete the required steps above", color = Muted, fontSize = 9.5.sp, maxLines = 1, overflow = TextOverflow.Ellipsis)
                }
                Spacer(Modifier.width(12.dp))
                Box(
                    Modifier
                        .widthIn(min = 156.dp)
                        .height(50.dp)
                        .clip(RoundedCornerShape(16.dp))
                        .background(if (enabled) Brush.horizontalGradient(listOf(Color(0xFFE34B7B), Burgundy, Color(0xFF7B1739))) else Brush.horizontalGradient(listOf(Color(0xFFE4E9EF), Color(0xFFD5DCE5))))
                        .clickable(enabled = enabled) { vm.startProcessing() },
                    contentAlignment = Alignment.Center
                ) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text(if (vm.running) "Processing…" else "✦", color = if (enabled) Color.White else Color(0xFF94A3B8), fontWeight = FontWeight.Black, fontSize = if (vm.running) 14.sp else 18.sp)
                        Spacer(Modifier.width(7.dp))
                        Text(label, color = if (enabled) Color.White else Color(0xFF94A3B8), fontWeight = FontWeight.Black, fontSize = 13.sp)
                    }
                }
            }
        }
    }
}

/* ----------------------------------------------------------------------- */
/* Shared building blocks                                                   */
/* ----------------------------------------------------------------------- */

@Composable
private fun StepCard(
    badge: String,
    title: String,
    subtitle: String,
    content: @Composable ColumnScope.() -> Unit
) {
    Card(
        shape = RoundedCornerShape(20.dp),
        colors = CardDefaults.cardColors(containerColor = Color.White),
        elevation = CardDefaults.cardElevation(defaultElevation = 6.dp),
        border = BorderStroke(1.dp, Color(0xFFEDF1F6))
    ) {
        Column(Modifier.fillMaxWidth().padding(13.dp), verticalArrangement = Arrangement.spacedBy(9.dp)) {
            Row(verticalAlignment = Alignment.Top) {
                Box(Modifier.size(32.dp).shadow(5.dp, RoundedCornerShape(11.dp), clip = false).clip(RoundedCornerShape(11.dp)).background(Brush.linearGradient(listOf(BurgundyLite, Burgundy))), contentAlignment = Alignment.Center) {
                    Text(badge, color = Color.White, fontWeight = FontWeight.Black, fontSize = 10.sp)
                }
                Spacer(Modifier.width(11.dp))
                Column(Modifier.weight(1f)) {
                    Text(title, fontWeight = FontWeight.Black, fontSize = 15.5.sp, color = Ink, letterSpacing = (-.25).sp)
                    Spacer(Modifier.height(2.dp))
                    Text(subtitle, color = Muted, fontSize = 11.5.sp, lineHeight = 16.sp)
                }
            }
            content()
        }
    }
}

@Composable
private fun SubHeading(text: String) {
    Text(
        text.uppercase(),
        color = Burgundy,
        fontWeight = FontWeight.Black,
        fontSize = 10.5.sp,
        letterSpacing = 0.6.sp,
        modifier = Modifier.padding(top = 2.dp, bottom = 2.dp)
    )
}

@Composable
private fun EmptyNote(text: String) {
    Box(
        Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(16.dp))
            .background(Color(0xFFF8FAFC))
            .border(1.dp, Line, RoundedCornerShape(16.dp))
            .padding(vertical = 22.dp, horizontal = 16.dp),
        contentAlignment = Alignment.Center
    ) {
        Text(text, color = Muted, fontSize = 13.sp, textAlign = TextAlign.Center)
    }
}

@Composable
private fun DisclosureRow(label: String, open: Boolean, onClick: () -> Unit) {
    Row(
        Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(14.dp))
            .background(if (open) BurgundySoft else Color(0xFFF8FAFC))
            .border(1.dp, if (open) Color(0xFFFBCFE8) else Line, RoundedCornerShape(14.dp))
            .clickable { onClick() }
            .padding(horizontal = 14.dp, vertical = 13.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            label,
            color = if (open) Burgundy else Slate,
            fontWeight = FontWeight.SemiBold,
            fontSize = 13.sp,
            modifier = Modifier.weight(1f)
        )
        Text(
            if (open) "▴" else "▾",
            color = if (open) Burgundy else Muted,
            fontSize = 12.sp
        )
    }
}

@Composable
private fun PresetButtons(vm: PhoenixViewModel) {
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(5.dp)) {
        listOf("Speed", "Optimized", "Balanced", "Mobile HQ", "Quality", "HQ").forEach { name ->
            Box(
                Modifier
                    .weight(1f)
                    .clip(RoundedCornerShape(9.dp))
                    .background(Color(0xFFF1F5FA))
                    .border(1.dp, Line, RoundedCornerShape(9.dp))
                    .clickable {
                        when (name) {
                            "Speed" -> vm.updateOptions { it.copy(resolution = "640p (Fast)", quality = "Balanced", swapEvery = "3", detectEvery = "3", detectInterval = 6) }
                            // "Auto" hands frame-skip to the quality tier. Pinning
                            // these to "1" (as they were) forced every-frame
                            // processing and made the tier choice meaningless.
                            "Optimized" -> vm.updateOptions { it.copy(resolution = "720p (HD)", quality = "Optimized", enhancer = "None", swapEvery = "Auto", detectEvery = "Auto", detectInterval = 4) }
                            "Balanced" -> vm.updateOptions { it.copy(resolution = "720p (HD)", quality = "Balanced", enhancer = "None", swapEvery = "Auto", detectEvery = "Auto", detectInterval = 4) }
                            "Mobile HQ" -> vm.updateOptions { it.copy(resolution = "680p", quality = "Ultra", swapEvery = "Auto", detectEvery = "Auto", detectInterval = 4) }
                            // Quality/HQ deliberately stay at 1/1 - these presets
                            // exist to trade speed for maximum stability.
                            "Quality" -> vm.updateOptions { it.copy(resolution = "720p (HD)", quality = "Best", enhancer = "None", swapEvery = "1", detectEvery = "1", detectInterval = 4) }
                            "HQ" -> vm.updateOptions { it.copy(resolution = "900p (HD+)", quality = "Ultra", swapEvery = "1", detectEvery = "1", detectInterval = 4) }
                        }
                    }
                    .padding(vertical = 9.dp),
                contentAlignment = Alignment.Center
            ) {
                Text(name, color = Slate, fontSize = 10.sp, fontWeight = FontWeight.SemiBold, maxLines = 1)
            }
        }
    }
}

@Composable
private fun Dropdown(
    label: String,
    values: List<String>,
    selected: String,
    onSelect: (String) -> Unit
) {
    var open by remember { mutableStateOf(false) }
    Box {
        Row(
            Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(12.dp))
                .background(Color(0xFFF7F9FC))
                .border(1.dp, Line, RoundedCornerShape(12.dp))
                .clickable { open = true }
                .padding(horizontal = 13.dp, vertical = 12.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text(label, color = Slate, fontSize = 12.5.sp, modifier = Modifier.weight(1f))
            Text(selected, color = Burgundy, fontWeight = FontWeight.Bold, fontSize = 12.5.sp)
            Spacer(Modifier.width(6.dp))
            Text("\u25BE", color = Muted, fontSize = 11.sp)
        }
        DropdownMenu(expanded = open, onDismissRequest = { open = false }) {
            values.forEach { v ->
                DropdownMenuItem(
                    text = { Text(v, fontSize = 13.sp) },
                    onClick = { onSelect(v); open = false }
                )
            }
        }
    }
}

@Composable
private fun ActionButton(
    text: String,
    modifier: Modifier = Modifier.fillMaxWidth(),
    enabled: Boolean = true,
    busy: Boolean = false,
    onClick: () -> Unit
) {
    Button(
        onClick = onClick,
        modifier = modifier.height(48.dp),
        enabled = enabled,
        shape = RoundedCornerShape(15.dp),
        colors = ButtonDefaults.buttonColors(containerColor = Burgundy, contentColor = Color.White, disabledContainerColor = Color(0xFFE5EAF0), disabledContentColor = Color(0xFF94A3B8)),
        elevation = ButtonDefaults.buttonElevation(defaultElevation = 0.dp, pressedElevation = 1.dp)
    ) {
        if (busy) { CircularProgressIndicator(Modifier.size(16.dp), color = Color.White, strokeWidth = 2.dp); Spacer(Modifier.width(9.dp)) }
        Text(text, fontWeight = FontWeight.Black, fontSize = 13.sp)
    }
}

@Composable
private fun GhostButton(
    text: String,
    modifier: Modifier = Modifier.fillMaxWidth(),
    danger: Boolean = false,
    onClick: () -> Unit
) {
    OutlinedButton(
        onClick = onClick,
        modifier = modifier.height(44.dp),
        shape = RoundedCornerShape(14.dp),
        border = BorderStroke(1.dp, if (danger) Burgundy.copy(alpha = .35f) else Line),
        colors = ButtonDefaults.outlinedButtonColors(contentColor = if (danger) Burgundy else Slate)
    ) { Text(text, fontWeight = FontWeight.Bold, fontSize = 12.5.sp) }
}
