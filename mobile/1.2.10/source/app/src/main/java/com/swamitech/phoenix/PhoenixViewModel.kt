package com.swamitech.phoenix

import android.app.Application
import android.content.ContentValues
import android.content.Context
import android.net.ConnectivityManager
import android.net.Network
import android.net.NetworkCapabilities
import android.net.NetworkRequest
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import android.provider.DocumentsContract
import android.content.Intent
import androidx.core.content.ContextCompat
import android.widget.Toast
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.swamitech.phoenix.local.LocalRuntime
import com.swamitech.phoenix.local.ModelPackManager
import com.swamitech.phoenix.net.HfApi
import com.swamitech.phoenix.security.SecureStore
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.Deferred
import kotlinx.coroutines.delay
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.sync.Semaphore
import kotlinx.coroutines.sync.withPermit
import kotlinx.coroutines.withContext
import kotlinx.coroutines.withTimeoutOrNull
import kotlin.coroutines.resume
import org.json.JSONArray
import org.json.JSONObject
import android.provider.OpenableColumns
import java.io.File
import java.util.UUID

class PhoenixViewModel(app: Application) : AndroidViewModel(app) {
    companion object {
        const val DEFAULT_SPACE = "https://swamivicky-swamitechgradio2.hf.space"
        const val DEFAULT_ENDPOINT = "phoenix_submit_video"
        const val MAX_SPACES = 21
        private const val PREFS = "phoenix_history"
        private const val HISTORY_KEY = "items"
        private const val SPACES_JSON_KEY = "spaces_json"
        // Old two-slot keys, kept only so an existing install's Space 1/Space 2
        // configuration migrates into the new 5-slot list instead of being lost.
        private const val SPACE1_LABEL_KEY = "space1_label"
        private const val SPACE1_URL_KEY = "space1_url"
        private const val SPACE2_LABEL_KEY = "space2_label"
        private const val SPACE2_URL_KEY = "space2_url"
        private const val SELECTED_SPACE_KEY = "selected_space"
        private const val OUTPUT_TREE_KEY = "output_tree_uri"
        private const val SAVED_JOBS_KEY = "saved_remote_job_ids"
        private const val SERVER_KEEP_MS = 3L * 60L * 60L * 1000L
        private val DEAD_STATUSES = setOf("Done", "Failed", "Monitoring stopped")
        private val NETWORK_HOLD_STATUSES = setOf(
            "Waiting for network", "Ready on server", "On server", "Reconnecting"
        )
    }

    data class SpaceSlot(val label: String, val url: String)

    data class HistoryItem(
        val id: String,
        val kind: String,
        val name: String,
        val uri: String?,
        val status: String,
        val createdAt: Long,
        val spaceLabel: String = "Space 1",
        val spaceUrl: String = "",
        val remoteJobId: String? = null,
        val progress: Float = 0f,
        val eta: String = "ETA —",
        val groupId: String? = null,
        val partIndex: Int = 0,
        val partCount: Int = 1,
        val sourcePath: String? = null
    )

    private val secure = SecureStore(app)
    private val models = ModelPackManager(app)
    private val saveLock = Any()
    private val savingJobIds = mutableSetOf<String>()

    // Keyed by space+video so two concurrent jobs on DIFFERENT Spaces never
    // collide on a single shared slot - the old 3-field design let a
    // still-in-flight write on one Space be read by a job targeting a
    // different Space, handing back a path that was never uploaded there at
    // all. A ConcurrentHashMap of Deferred (not plain values) also makes
    // concurrent calls for the SAME key single-flight: a second caller joins
    // the one already-running upload instead of starting a wasteful
    // duplicate transfer of the same file.
    private val videoUploadCache = java.util.concurrent.ConcurrentHashMap<String, Deferred<UploadedVideo>>()

    /**
     * A video already sitting on a Space, plus whatever trim the server
     * still has to apply to it. When the clip was cut locally before upload,
     * the uploaded file IS the selected range, so the original percentages
     * must not be sent again - that would trim an already-trimmed clip.
     */
    private data class UploadedVideo(
        val path: String,
        val trimStartPct: Int,
        val trimEndPct: Int
    )

    /**
     * Upload concurrency is isolated PER Space. A global semaphore made jobs
     * on different Hugging Face Spaces wait behind each other even though the
     * remote renderers are independent. Keep one upload in flight per Space
     * to protect the phone/network while allowing Space 1 and Space 2 (etc.)
     * to start independently.
     */
    private val uploadGates = java.util.concurrent.ConcurrentHashMap<String, Semaphore>()

    private fun uploadGate(settings: HfSettings): Semaphore {
        val key = settings.spaceUrl.trim().trimEnd('/').lowercase()
        return uploadGates.computeIfAbsent(key) { Semaphore(permits = 3) }
    }

    // The trim range is part of the key: the uploaded file is now the cut
    // segment, not the whole source, so two different trims are two
    // genuinely different uploads. Without this, changing the trim would
    // silently reuse the previously-cut file.
    private fun videoCacheKey(settings: HfSettings, uri: Uri, trimStart: Int, trimEnd: Int): String =
        "${settings.spaceUrl.trim().trimEnd('/')}|$uri|$trimStart-$trimEnd"

    private suspend fun uploadVideoOnce(
        settings: HfSettings,
        uri: Uri,
        trimStart: Int = options.trimStart,
        trimEnd: Int = options.trimEnd,
        onProgress: ((written: Long, total: Long) -> Unit)? = null
    ): UploadedVideo {
        val key = videoCacheKey(settings, uri, trimStart, trimEnd)
        val wantStart = trimStart
        val wantEnd = trimEnd
        val deferred = videoUploadCache.computeIfAbsent(key) {
            viewModelScope.async(Dispatchers.IO) {
                val app = getApplication<Application>()
                val mime = app.contentResolver.getType(uri) ?: "video/mp4"

                // Cut the clip locally so only the selected range crosses the
                // network. Trimming used to happen entirely server-side, so a
                // 10-second selection out of a 4-minute clip still uploaded
                // all 4 minutes - the slowest part of the job, and the exact
                // transfer that has been dropping when several Spaces start
                // together. Any failure here falls straight through to the
                // old behaviour (upload everything, let the server trim): a
                // trim that cannot be performed must never fail the job.
                var trimmedFile: File? = null
                var sendStart = wantStart
                var sendEnd = wantEnd
                if (wantStart > 0 || wantEnd < 100) {
                    val totalUs = VideoTrimmer.durationUs(app, uri)
                    if (totalUs != null && totalUs > 0L) {
                        val startUs = totalUs * wantStart / 100L
                        val endUs = totalUs * wantEnd / 100L
                        if (endUs > startUs) {
                            status = "Trimming video locally…"
                            val dest = File(app.cacheDir, "phoenix_trim_${System.currentTimeMillis()}.mp4")
                            val cut = VideoTrimmer.trim(app, uri, startUs, endUs, dest)
                            if (cut != null && cut.durationUs > 0L) {
                                trimmedFile = cut.file
                                // Only the sync-frame residue is left for the
                                // server, expressed against the NEW duration.
                                // Truncating (not rounding) is deliberate: the
                                // server's trim is an integer percentage, so
                                // the residue can't always be expressed
                                // exactly, and truncating errs toward keeping
                                // a few extra milliseconds at the head rather
                                // than cutting into footage the user actually
                                // selected. The residual error is far finer
                                // than the trim slider itself, which is 1% of
                                // the SOURCE (seconds on a long clip).
                                sendStart = ((cut.residualStartUs * 100.0) / cut.durationUs)
                                    .toInt().coerceIn(0, 99)
                                sendEnd = 100
                            }
                        }
                    }
                }

                var lastProgressPush = 0L
                val path = uploadGate(settings).withPermit {
                    if (trimmedFile == null) {
                        // No local trim is needed: stream the gallery URI directly
                        // to OkHttp instead of first copying the entire video into
                        // the cache. This removes one full disk read/write pass.
                        hf.uploadUri(
                            settings, app.contentResolver, uri,
                            "phoenix_video_${System.currentTimeMillis()}.mp4", mime
                        ) { written, total ->
                            val now = System.currentTimeMillis()
                            if (now - lastProgressPush >= 400L || (total > 0 && written >= total)) {
                                lastProgressPush = now
                                val doneMb = "%.1f".format(written / 1_048_576.0)
                                status = if (total > 0) {
                                    val totalMb = "%.1f".format(total / 1_048_576.0)
                                    val pct = (written.toFloat() / total * 100).toInt()
                                    "Uploading video — $doneMb of $totalMb MB ($pct%)"
                                } else {
                                    "Uploading video — $doneMb MB so far"
                                }
                                onProgress?.invoke(written, total)
                            }
                        }
                    } else {
                        hf.upload(settings, trimmedFile!!, mime) { written, total ->
                        val now = System.currentTimeMillis()
                        if (now - lastProgressPush >= 400L || (total > 0 && written >= total)) {
                            lastProgressPush = now
                            val doneMb = "%.1f".format(written / 1_048_576.0)
                            status = if (total > 0) {
                                val totalMb = "%.1f".format(total / 1_048_576.0)
                                val pct = (written.toFloat() / total * 100).toInt()
                                "Uploading video — $doneMb of $totalMb MB ($pct%)"
                            } else {
                                "Uploading video — $doneMb MB so far"
                            }
                            onProgress?.invoke(written, total)
                        }
                        }
                    }
                }
                trimmedFile?.let { safeDelete(it) }
                UploadedVideo(path, sendStart, sendEnd)
            }
        }
        return try {
            deferred.await()
        } catch (e: Throwable) {
            // A failed upload must not poison the cache for a later retry.
            videoUploadCache.remove(key, deferred)
            throw e
        }
    }
    private val local by lazy { LocalRuntime(getApplication()) }
    private val hf = HfApi { secure.getToken() }
    private val prefs = app.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
    private var processingJob: Job? = null
    // Concurrent map so multiple job coroutines can safely add/remove their
    // own entries without corrupting the map's internal structure - a plain
    // mutableMapOf is not safe for concurrent modification even for simple
    // put/remove calls from different threads.
    private val parallelJobs = java.util.concurrent.ConcurrentHashMap<String, Job>()
    private var latestJobLocalId: String? = null

    var appMode by mutableStateOf(AppMode.HUGGING_FACE); private set
    var activeTab by mutableStateOf("Video"); private set
    var status by mutableStateOf("Ready — Phoenix Cloud mode"); private set
    var eta by mutableStateOf("ETA —"); private set
    var progress by mutableFloatStateOf(0f); private set
    var farmPhase by mutableStateOf(""); private set
    var farmPercent by mutableIntStateOf(0); private set
    var running by mutableStateOf(false); private set

    private fun setFarm(phase: String, pct: Int) {
        farmPhase = phase
        farmPercent = pct.coerceIn(0, 100)
        progress = farmPercent / 100f
        status = if (phase.isBlank()) status else "$phase  ${farmPercent}%"
        eta = if (phase.isBlank()) eta else "${farmPercent}%"
    }

    var selectedVideo by mutableStateOf<Uri?>(null); private set
    var selectedImage by mutableStateOf<Uri?>(null); private set
    var face1 by mutableStateOf<Uri?>(null); private set
    var face2 by mutableStateOf<Uri?>(null); private set
    var face3 by mutableStateOf<Uri?>(null); private set
    var face4 by mutableStateOf<Uri?>(null); private set
    var resultUri by mutableStateOf<Uri?>(null); private set
    var imageResultUri by mutableStateOf<Uri?>(null); private set

    var framePercent by mutableStateOf(0); private set
    var detectedFrameUri by mutableStateOf<Uri?>(null); private set
    var detectedFaces by mutableStateOf<List<Uri?>>(listOf(null, null, null, null)); private set
    private var targetRefs = mutableListOf<Uri?>(null, null, null, null)
    var detectingFrame by mutableStateOf(false); private set
    var detectingImage by mutableStateOf(false); private set

    var options by mutableStateOf(PhoenixOptions()); private set
    private var _imageQuality by mutableStateOf("Best")
    val imageQuality: String get() = _imageQuality
    fun updateImageQuality(value: String) { _imageQuality = value }

    var hfTokenSet by mutableStateOf(!secure.getToken().isNullOrBlank()); private set
    var packFiles by mutableStateOf<List<Uri>>(emptyList()); private set
    var packPushing by mutableStateOf(false); private set
    var packStatus by mutableStateOf(""); private set
    var selectedSpaceIndex by mutableStateOf(0); private set
    var spaces by mutableStateOf(defaultSpaces()); private set
    val spaceUrl: String get() = spaces[selectedSpaceIndex].url
    val spaceLabel: String get() = spaces[selectedSpaceIndex].label
    var activeJobCount by mutableIntStateOf(0); private set
    var endpoint by mutableStateOf(DEFAULT_ENDPOINT)
    var modelRepo by mutableStateOf("")
    var modelRevision by mutableStateOf("main")
    var discoveredApi by mutableStateOf(""); private set
    var modelReady by mutableStateOf(models.isComplete()); private set
    // One single lock for every history mutation. AppV16's attempt at this
    // used a coroutine Mutex in updateHistory() but a separate, unrelated
    // synchronized(Any()) lock in addHistory/removeHistory/clearHistory/
    // refreshHistory/cleanupHistoryMetadata - two independent primitives
    // provide zero protection against each other, so the exact race it was
    // meant to fix was still fully possible between those two groups. A
    // plain JVM lock works correctly from both suspend and non-suspend call
    // sites (none of these bodies need to suspend while holding it), so one
    // consistent primitive covers every mutation path without that trap.
    private val historyLock = Any()
    var history by mutableStateOf(loadHistory()); private set

    /**
     * A user-chosen SAF tree URI results save into instead of the default
     * Movies/Phoenix or Pictures/Phoenix. Null means "use the default" -
     * this is purely additive, nothing about the existing MediaStore path
     * changes for anyone who never sets one.
     */
    var outputTreeUri by mutableStateOf<Uri?>(
        prefs.getString(OUTPUT_TREE_KEY, null)?.let { runCatching { Uri.parse(it) }.getOrNull() }
    ); private set

    fun setOutputTree(uri: Uri?) {
        outputTreeUri = uri
        prefs.edit().apply {
            if (uri == null) remove(OUTPUT_TREE_KEY) else putString(OUTPUT_TREE_KEY, uri.toString())
        }.apply()
        status = if (uri == null) "Output folder reset to default (Movies/Phoenix)" else "Output folder set — new results will save there"
    }

    /** Human-readable label for whatever's currently configured, for the UI. */
    fun outputFolderLabel(): String {
        val uri = outputTreeUri ?: return "Default (Movies/Phoenix, Pictures/Phoenix)"
        val docUri = runCatching {
            DocumentsContract.buildDocumentUriUsingTree(uri, DocumentsContract.getTreeDocumentId(uri))
        }.getOrNull() ?: return "Custom folder"
        val name = runCatching {
            getApplication<Application>().contentResolver.query(
                docUri, arrayOf(DocumentsContract.Document.COLUMN_DISPLAY_NAME), null, null, null
            )?.use { c -> if (c.moveToFirst()) c.getString(0) else null }
        }.getOrNull()
        return name?.let { "Custom: $it" } ?: "Custom folder"
    }

    init {
        runCatching { loadSpaceSettings() }
        watchNetwork()
        resumeTrackedJobs()
    }

    private fun defaultSpaces(): List<SpaceSlot> {
        val gradio = (2..20).map { n ->
            SpaceSlot(
                "SwamitechGradio$n",
                "https://swamivicky-swamitechgradio$n.hf.space"
            )
        }
        val uat = listOf(
            SpaceSlot("SG_UAT2", "https://swamivicky-sg-uat2.hf.space"),
            SpaceSlot("SwamitechUAT2", "https://swamivicky-swamitechuat2.hf.space")
        )
        return (gradio + uat).take(MAX_SPACES)
    }

    private fun spaceKey(slot: SpaceSlot): String {
        var lab = slot.label.trim().lowercase()
            .replace(" ", "").replace("_", "").replace("-", "")
            .removePrefix("swamitech")
        when {
            lab.matches(Regex("gradio([2-9]|1[0-9]|20)")) ->
                lab = "sg" + lab.removePrefix("gradio")
            lab == "uat" || lab == "sguat" || lab == "sguat2" -> lab = "sguat2"
            lab == "uat2" -> lab = "uat2"
        }
        return lab
    }

    private fun normSpaceUrl(url: String): String =
        url.trim().trimEnd('/').lowercase()

    fun selectProcessingSpace(index: Int) {
        selectedSpaceIndex = index.coerceIn(0, MAX_SPACES - 1)
        prefs.edit().putInt(SELECTED_SPACE_KEY, selectedSpaceIndex).apply()
        status = "${spaceLabel} selected"
    }

    /**
     * Edits a Space by index WITHOUT changing which Space is selected for the
     * next job. An earlier version required the caller to select the Space
     * first, so simply typing in another Space's URL field silently rerouted
     * the next job to it.
     */
    fun updateSpaceLabel(index: Int, value: String) {
        if (index !in spaces.indices) return
        spaces = spaces.toMutableList().also { it[index] = it[index].copy(label = value) }
        persistSpaces()
    }

    fun updateSpaceUrl(index: Int, value: String) {
        if (index !in spaces.indices) return
        spaces = spaces.toMutableList().also { it[index] = it[index].copy(url = value) }
        persistSpaces()
    }

    private fun persistSpaces() {
        val arr = JSONArray()
        spaces.forEach { arr.put(JSONObject().put("label", it.label).put("url", it.url)) }
        prefs.edit().putString(SPACES_JSON_KEY, arr.toString()).apply()
    }

    fun setSelectedSpaceLabel(value: String) = updateSpaceLabel(selectedSpaceIndex, value)

    fun setSelectedSpaceUrl(value: String) = updateSpaceUrl(selectedSpaceIndex, value)

    /**
     * Why the primary Swap command is unavailable, or null when it can run.
     * The bottom action bar surfaces this instead of silently doing nothing.
     *
     * FIX: this used to check the global `running` flag, which goes true the
     * instant ANY job is active on ANY Space - so starting a job on Space 1
     * silently blocked starting a second one on Space 2, defeating the whole
     * point of running Spaces concurrently. It should only ever care about
     * how many concurrent jobs are actually active, exactly like the video
     * path below already did correctly.
     */
    val swapBlockReason: String?
        get() {
            if (activeTab == "Image") {
                if (selectedImage == null) return "Select a target image to enable the swap."
                if (listOf(face1, face2, face3, face4).all { it == null })
                    return "Add at least one replacement face."
                if (activeJobCount >= MAX_SPACES) return "All $MAX_SPACES Phoenix jobs are already active. Stop one in History."
                return null
            }
            if (selectedVideo == null) return "Select a target video to enable the swap."
            if (listOf(face1, face2, face3, face4).all { it == null })
                return "Add at least one replacement face."
            if (spaceUrl.isBlank()) return "Configure a URL for ${spaceLabel} first."
            if (activeJobCount >= MAX_SPACES) return "All $MAX_SPACES Phoenix jobs are already active. Stop one in History."
            return null
        }

    fun configuredSpaceNames(): List<String> = spaces.map {
        "${it.label}${if (it.url.isBlank()) " · not configured" else ""}"
    }

    fun isSpaceBusy(url: String): Boolean =
        url.isNotBlank() && history.any {
            it.spaceUrl == url && it.status !in setOf("Done", "Failed", "Monitoring stopped", "Replaced")
        }


    private fun selectedSettings(): HfSettings = HfSettings(
        spaceUrl.trim().trimEnd('/'), endpoint.trim().ifBlank { DEFAULT_ENDPOINT }, modelRepo, modelRevision
    )

    private fun loadSpaceSettings() {
        val defaults = defaultSpaces()
        val saved = mutableListOf<SpaceSlot>()
        val json = prefs.getString(SPACES_JSON_KEY, null)
        if (json != null) {
            runCatching {
                val arr = JSONArray(json)
                for (i in 0 until arr.length()) {
                    val o = arr.getJSONObject(i)
                    saved += SpaceSlot(o.optString("label"), o.optString("url"))
                }
            }
        } else {
            val l1 = prefs.getString(SPACE1_LABEL_KEY, null)
            val u1 = prefs.getString(SPACE1_URL_KEY, null)
            if (!l1.isNullOrBlank() || !u1.isNullOrBlank()) {
                saved += SpaceSlot(l1 ?: "SwamitechGradio2", u1 ?: "")
            }
            val l2 = prefs.getString(SPACE2_LABEL_KEY, null)
            val u2 = prefs.getString(SPACE2_URL_KEY, null)
            if (!l2.isNullOrBlank() || !u2.isNullOrBlank()) {
                saved += SpaceSlot(l2 ?: "SwamitechGradio3", u2 ?: "")
            }
        }

        val keptSaved = saved
        val byKey = LinkedHashMap<String, SpaceSlot>()
        for (s in keptSaved) {
            val k = spaceKey(s)
            if (k.isNotBlank()) byKey.putIfAbsent(k, s)
            val u = normSpaceUrl(s.url)
            if (u.isNotBlank()) byKey.putIfAbsent(u, s)
        }

        val ordered = defaults.map { d ->
            val hit = byKey[spaceKey(d)] ?: byKey[normSpaceUrl(d.url)]
            val url = if (hit != null && hit.url.isNotBlank()) hit.url else d.url
            SpaceSlot(d.label, url)
        }.toMutableList()

        val used = ordered.map { normSpaceUrl(it.url) }.filter { it.isNotBlank() }.toMutableSet()
        for (s in keptSaved) {
            val u = normSpaceUrl(s.url)
            if (u.isNotBlank() && u !in used && ordered.size < MAX_SPACES) {
                ordered += s
                used += u
            }
        }
        while (ordered.size < MAX_SPACES) {
            ordered += SpaceSlot("Space ${ordered.size + 1}", "")
        }
        spaces = ordered.take(MAX_SPACES)
        persistSpaces()
        selectedSpaceIndex = prefs.getInt(SELECTED_SPACE_KEY, 0).coerceIn(0, MAX_SPACES - 1)
    }

    fun setTab(tab: String) { activeTab = tab; status = "${tab} workspace ready" }

    fun setMode(mode: AppMode) { appMode = mode; status = if (mode == AppMode.HUGGING_FACE) "Phoenix Cloud mode selected" else "Local mode selected" }

    /** Deletes the cache files backing the currently-shown detection/frame
     *  previews before they're replaced or discarded - otherwise every past
     *  detection's frame and face crops just accumulate on disk forever. */
    private fun clearDisplayedDetectionFiles() {
        safeDelete(detectedFrameUri)
        detectedFaces.forEach { safeDelete(it) }
    }

    fun selectVideo(uri: Uri?) {
        clearDisplayedDetectionFiles()
        selectedVideo = uri; resultUri = null; detectedFrameUri = null; detectedFaces = List(4) { null }
        progress = 0f; eta = "ETA —"; status = if (uri == null) "No video selected" else "Video selected"
        // A new video invalidates any previously-uploaded copy sitting on a
        // Space - see uploadVideoOnce() for why this cache exists.
        videoUploadCache.clear()
    }

    fun selectImage(uri: Uri?) {
        clearDisplayedDetectionFiles()
        selectedImage = uri; imageResultUri = null; detectedFaces = List(4) { null }; status = if (uri == null) "No image selected" else "Target image selected"
    }

    fun clearFace(slot: Int) { setFace(slot, null); status = "Replacement face $slot cleared" }

    fun selectFace(slot: Int, uri: Uri?) { setFace(slot, uri); status = if (uri == null) "Face $slot cleared" else "Replacement face $slot selected" }

    private fun setFace(slot: Int, uri: Uri?) { when (slot) { 1 -> face1 = uri; 2 -> face2 = uri; 3 -> face3 = uri; 4 -> face4 = uri } }

    fun saveToken(token: String) {
        secure.putToken(token.trim()); hfTokenSet = !secure.getToken().isNullOrBlank()
        status = if (hfTokenSet) "Hugging Face token saved securely" else "Hugging Face token removed"
    }

    fun choosePackFiles(uris: List<Uri>) {
        packFiles = uris.filterNotNull()
        packStatus = if (packFiles.isEmpty()) "No files selected" else "${packFiles.size} files ready to push"
    }

    /**
     * Sends the unzipped Phoenix files (app.py, config.py, and the rest) to
     * every configured Space. Each Space gets one commit and then restarts.
     */
    fun pushPackToAllSpaces() {
        if (packPushing) return
        if (!hfTokenSet) {
            packStatus = "Save a Hugging Face token with write access first"
            return
        }
        val chosen = packFiles
        if (chosen.isEmpty()) {
            packStatus = "Pick the unzipped package files first"
            return
        }
        val targets = spaces.filter { it.url.isNotBlank() }
        if (targets.isEmpty()) {
            packStatus = "No Space URLs are configured"
            return
        }
        packPushing = true
        viewModelScope.launch(Dispatchers.IO) {
            val resolver = getApplication<Application>().contentResolver
            val blobs = mutableListOf<Pair<String, ByteArray>>()
            for (uri in chosen) {
                val name = runCatching {
                    resolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
                        ?.use { c -> if (c.moveToFirst()) c.getString(0) else null }
                }.getOrNull()?.substringAfterLast('/')?.trim().orEmpty()
                if (name.isBlank() || name.contains("..") || name.endsWith(".zip", true) || name.endsWith(".apk", true)) {
                    continue
                }
                val bytes = runCatching { resolver.openInputStream(uri)?.use { it.readBytes() } }.getOrNull()
                    ?: continue
                if (bytes.size > 8 * 1024 * 1024) continue
                blobs += name to bytes
            }
            if (blobs.isEmpty()) {
                withContext(Dispatchers.Main) {
                    packPushing = false
                    packStatus = "Nothing to send. Unzip the pack and pick the files inside, not the zip."
                }
                return@launch
            }
            var ok = 0
            var failed = 0
            // Keep the FIRST error in full. Showing only the last one, cut at 120
            // characters, hid the part of the Hub's answer that says what is wrong.
            var firstError = ""
            targets.forEachIndexed { index, slot ->
                withContext(Dispatchers.Main) {
                    packStatus = "Updating ${slot.label} (${index + 1}/${targets.size})"
                }
                try {
                    hf.commitSpaceFiles(repoIdOf(slot), blobs)
                    ok++
                } catch (e: Exception) {
                    failed++
                    if (firstError.isEmpty()) {
                        firstError = e.message?.take(600) ?: "${slot.label}: ${e.javaClass.simpleName}"
                    }
                }
            }
            withContext(Dispatchers.Main) {
                packPushing = false
                packStatus = if (failed == 0) {
                    "Updated $ok Spaces. Each one restarts on its own."
                } else {
                    "Updated $ok, failed $failed. First error: $firstError"
                }
            }
        }
    }

    private fun repoIdOf(slot: SpaceSlot): String {
        val host = slot.url.trim()
            .removePrefix("https://")
            .removePrefix("http://")
            .substringBefore('/')
            .lowercase()
        val sub = host.removeSuffix(".hf.space")
        val repo = slot.label.trim().lowercase().replace(' ', '-').replace('_', '-')
        val owner = if (repo.isNotBlank() && sub.endsWith("-$repo")) {
            sub.removeSuffix("-$repo")
        } else {
            sub.substringBefore('-')
        }
        if (owner.isBlank() || slot.label.isBlank()) {
            throw IllegalStateException("Cannot read a Space name from ${slot.label}")
        }
        return "$owner/${slot.label.trim()}"
    }

    fun discoverSpace() {
        val url = spaceUrl.trim()
        if (url.isBlank()) { status = "Enter the Hugging Face Space URL"; return }
        viewModelScope.launch(Dispatchers.IO) {
            val settings = HfSettings(url.trim().trimEnd('/'), endpoint.trim().ifBlank { DEFAULT_ENDPOINT }, modelRepo, modelRevision)
            runCatching { hf.discoverSpace(settings) }
                .onSuccess { json -> discoveredApi = json; status = "Space reachable · API route probing enabled" }
                .onFailure { status = "API connection failed: ${it.message ?: it.javaClass.simpleName}" }
        }
    }

    /**
     * Named Gradio routes this client cannot work without. If the Space does not
     * publish one of these, every call to it returns HTTP 404 and the feature
     * fails with a message that says nothing useful on its own.
     */
    val requiredEndpoints: List<String> = listOf(
        "phoenix_submit_video",
        "phoenix_job_status",
        "phoenix_download",
        "phoenix_detect_video_frame",
        "detect_image",
        "swap_image"
    )

    var spaceEndpoints by mutableStateOf<List<String>>(emptyList()); private set
    var spaceSignatures by mutableStateOf<Map<String, String>>(emptyMap()); private set

    /** What this client actually sends, for side-by-side comparison. */
    val clientSignatures: Map<String, String> = mapOf(
        "phoenix_detect_video_frame" to "2 in (video FileData, frame percent)",
        "detect_image" to "1 in (image FileData)",
        "swap_image" to "7 in (target, face1-4, quality, reserved)",
        "phoenix_submit_video" to "6 in (video, face1-4, options JSON)",
        "phoenix_job_status" to "1 in (job id)",
        "phoenix_download" to "1 in (job id)"
    )
    var diagnosing by mutableStateOf(false); private set
    var lastError by mutableStateOf<String?>(null); private set

    val missingEndpoints: List<String>
        get() = if (spaceEndpoints.isEmpty()) emptyList()
        else requiredEndpoints.filterNot { spaceEndpoints.contains(it) }

    fun diagnoseSpace() {
        val url = spaceUrl.trim()
        if (url.isBlank()) { status = "Enter the Hugging Face Space URL first"; return }
        if (diagnosing) return
        diagnosing = true
        status = "Reading the API description from ${spaceLabel}…"
        viewModelScope.launch(Dispatchers.IO) {
            try {
                val found = hf.listEndpoints(settings())
                spaceEndpoints = found
                spaceSignatures = runCatching { hf.endpointSignatures(settings()) }.getOrDefault(emptyMap())
                val missing = requiredEndpoints.filterNot { found.contains(it) }
                status = when {
                    found.isEmpty() ->
                        "${spaceLabel} answered but published no named API routes. The Space app needs api_name= on its functions."
                    missing.isEmpty() ->
                        "${spaceLabel} exposes all ${requiredEndpoints.size} required routes."
                    else ->
                        "${spaceLabel} is missing ${missing.size} required route(s): ${missing.joinToString(", ")}"
                }
                lastError = null
            } catch (e: Throwable) {
                spaceEndpoints = emptyList()
                spaceSignatures = emptyMap()
                lastError = e.message ?: e.javaClass.simpleName
                status = "Diagnosis failed: ${lastError}"
            } finally {
                diagnosing = false
            }
        }
    }

    fun clearError() { lastError = null }

    fun updateFrame(value: Int) { framePercent = value.coerceIn(0, 100) }
    fun updateOptions(transform: (PhoenixOptions) -> PhoenixOptions) { options = transform(options) }
    fun resetOptions() { options = PhoenixOptions(); _imageQuality = "Best"; status = "Controls reset to recommended defaults" }

    fun detectFrame() {
        val uri = selectedVideo ?: run { status = "Select a target video first"; return }
        if (detectingFrame) return
        detectingFrame = true; lastError = null; status = "Uploading frame source and detecting faces…"
        viewModelScope.launch(Dispatchers.IO) {
            try {
                val settings = settings()
                val alreadyCached = videoUploadCache.containsKey(
                    videoCacheKey(settings, uri, options.trimStart, options.trimEnd)
                )
                status = if (alreadyCached) "Detecting faces (video already on ${spaceLabel})…" else "Uploading video and detecting faces…"
                val videoPath = uploadVideoOnce(settings, uri).path
                val detection = hf.detectVideoFrameFaces(settings, videoPath, framePercent)

                // Each crop is downloaded independently and kept at its own
                // index (2 detected faces on a 4-slot form stay in slots 1-2,
                // never shifted by a missing/null crop elsewhere).
                val annotated = detection.annotatedPath?.let { downloadToCache(settings, it, "frame") }
                val faces = detection.facePaths.mapIndexed { i, path ->
                    path?.let { runCatching { downloadToCache(settings, it, "detected_${i + 1}") }.getOrNull() }
                }

                detectedFrameUri?.let { safeDelete(it) }
                detectedFaces.forEach { safeDelete(it) }
                detectedFrameUri = annotated?.let(Uri::fromFile)
                detectedFaces = (faces.map { it?.let(Uri::fromFile) } + List(4 - faces.size) { null }).take(4)

                val count = detectedFaces.count { it != null }
                status = detection.message?.let { "$it (frame $framePercent%)" }
                    ?: if (count > 0) "Detected $count face(s) at $framePercent%"
                    else "Frame returned, but no faces were detected. Try a frame where the face is clearly visible."
            } catch (e: Throwable) {
                val detail = e.message ?: e.javaClass.simpleName
                lastError = when {
                    // A download 404 means the route ran and returned a real
                    // result - it is not missing. Reporting it as "route
                    // missing" here was actively wrong, not just unhelpful.
                    detail.contains("Output download failed") ->
                        "phoenix_detect_video_frame ran and returned a result, but downloading the returned file from " +
                            "${spaceLabel} failed ($detail). If this repeats after updating the app, the Space's /file= " +
                            "route may be rejecting the path it returned."
                    detail.contains("404") && (detail.contains("Queue request failed") || detail.contains("Polling failed")) ->
                        "The Space did not answer on phoenix_detect_video_frame (HTTP 404). " +
                            "That named route is missing from ${spaceLabel}. Run Diagnose Space to see what it does publish."
                    else -> detail
                }
                status = "Frame detection failed: $detail"
            }
            finally { detectingFrame = false }
        }
    }


    fun detectImage() {
        val uri = selectedImage ?: run { status = "Select a target image first"; return }
        if (detectingImage) return
        detectingImage = true; lastError = null; status = "Uploading target image and detecting faces…"
        viewModelScope.launch(Dispatchers.IO) {
            try {
                val settings = settings()
                val file = LocalFileBridge.copyToCache(getApplication(), uri, "phoenix_target_image")
                val paths = hf.detectImage(settings, file)
                safeDelete(file)
                val faces = paths.drop(3).take(4).mapIndexed { i, p ->
                    p?.let { runCatching { downloadToCache(settings, it, "image_detected_${i + 1}") }.getOrNull() }
                }
                detectedFaces.forEach { safeDelete(it) }
                detectedFaces = (faces.map { it?.let(Uri::fromFile) } + List(4 - faces.size) { null }).take(4)
                status = paths.getOrNull(1) ?: "Detected ${detectedFaces.count { it != null }} face(s)"
            } catch (e: Throwable) {
                val detail = e.message ?: e.javaClass.simpleName
                lastError = when {
                    detail.contains("Output download failed") ->
                        "detect_image ran and returned a result, but downloading the returned file from " +
                            "${spaceLabel} failed ($detail)."
                    detail.contains("404") && (detail.contains("Queue request failed") || detail.contains("Polling failed")) ->
                        "The Space did not answer on detect_image (HTTP 404). That named route is missing from ${spaceLabel}."
                    else -> detail
                }
                status = "Image detection failed: $detail"
            }
            finally { detectingImage = false }
        }
    }

    fun captureDetectedFace(slot: Int) {
        detectedFaces.getOrNull(slot - 1)?.let { setFace(slot, it); status = "Detected face captured into replacement slot #$slot" }
            ?: run { status = "No detected face in slot #$slot" }
    }

    /**
     * Send detected face [detectedIndex] (0-based, as shown in the Detected
     * faces strip) into replacement slot [slot] (1-based).
     *
     * Unlike [captureDetectedFace], the source index and the destination slot
     * are independent. That is what makes multi-frame capture possible: scrub
     * to a frame where only person A is visible, send the single detected face
     * to slot #1; scrub to a different frame showing person B, and send that
     * frame's single detected face to slot #2. Neither frame has to contain
     * both people at once.
     */
    fun captureDetectedFaceTo(detectedIndex: Int, slot: Int) {
        if (slot !in 1..4) { status = "Invalid replacement slot #$slot"; return }
        val uri = detectedFaces.getOrNull(detectedIndex)
        if (uri == null) { status = "No detected face at position #${detectedIndex + 1}"; return }

        // Take an independent copy rather than aliasing the detected-crop Uri.
        // detectFrame() safeDelete()s every entry in detectedFaces before it
        // repopulates them, so a slot pointing straight at a detected crop
        // would silently go dangling the next time the user scrubs and
        // re-detects - which is precisely the multi-frame flow this exists to
        // support. Owning a private copy makes the slot survive re-detection.
        val copied = copyToSlotFile(uri, slot)
        if (copied == null) { status = "Could not copy detected face into slot #$slot"; return }
        val refCopy = runCatching {
            val dest = File(getApplication<Application>().cacheDir, "phoenix_tref${slot}_${System.currentTimeMillis()}")
            copied.copyTo(dest, overwrite = true)
            dest
        }.getOrNull()
        if (refCopy != null) {
            targetRefs[slot - 1]?.takeIf { it.scheme == "file" }?.path?.let { runCatching { File(it).delete() } }
            targetRefs[slot - 1] = Uri.fromFile(refCopy)
        }
        val already = when (slot) { 1 -> face1; 2 -> face2; 3 -> face3; else -> face4 }
        if (already == null) setFace(slot, Uri.fromFile(copied))
        status = "Identity for slot #$slot locked from this frame (extras ignored). Replacement photo stays in the slot."
    }

    /** Copies a file:// crop into a slot-owned cache file. Returns null on failure. */
    private fun copyToSlotFile(src: Uri, slot: Int): File? {
        val srcPath = src.takeIf { it.scheme == "file" }?.path ?: return null
        val srcFile = File(srcPath)
        if (!srcFile.exists()) return null
        return runCatching {
            val dest = File(
                getApplication<Application>().cacheDir,
                "phoenix_slot${slot}_${System.currentTimeMillis()}"
            )
            srcFile.copyTo(dest, overwrite = true)
            dest
        }.getOrNull()
    }

    fun installModels() {
        viewModelScope.launch(Dispatchers.IO) {
            status = "Downloading verified model pack…"
            runCatching { models.installFromHf(HfSettings(spaceUrl, endpoint, modelRepo, modelRevision)) { name, done, total -> progress = if (total > 0) done.toFloat() / total else 0f; status = "Downloading $name · ${(progress * 100).toInt()}%" } }
                .onSuccess { modelReady = models.isComplete(); progress = 1f; status = "Model pack installed and hash-verified" }
                .onFailure { status = "Model install failed: ${it.message ?: it.javaClass.simpleName}" }
        }
    }

    fun startProcessing() {
        if (appMode == AppMode.LOCAL) {
            if (running) { status = "A local job is already running"; return }
            startLocal(selectedVideo)
        } else if (activeTab == "Image") {
            startImage()
        } else {
            if (parallelJobs.size >= MAX_SPACES) {
                status = "All $MAX_SPACES Phoenix jobs are already being monitored. Use History to track them."
                return
            }
            val uri = selectedVideo ?: run { status = "Select a target video first"; return }
            val parts = options.splitParts.substringBefore(" ").toIntOrNull()?.takeIf { it in 2..10 } ?: 1
            if (parts == 1) startRemote(uri) else startSplitFarm(uri, parts)
        }
    }

    private fun startLocal(uri: Uri?) {
        if (uri == null) { status = "Select a target video first"; return }
        if (!modelReady) { status = "Install the verified local model pack first"; return }
        running = true
        processingJob = viewModelScope.launch(Dispatchers.Default) {
            runCatching { local.initialize() }
                .onSuccess { info ->
                    status = "$info · ${local.engineStatus()}"
                    if (!local.processingReady()) {
                        status = "Local face engine is not available in this release. No video was modified."
                    }
                }
                .onFailure { status = "Local model initialization failed: ${it.message ?: it.javaClass.simpleName}" }
            running = false
        }
    }

    private fun settings() = selectedSettings()

    private fun startImage() {
        val uri = selectedImage ?: run { status = "Select a target image first"; return }
        val faces = listOf(face1, face2, face3, face4)
        if (faces.all { it == null }) { status = "Select or capture at least one replacement face"; return }
        running = true; progress = 0f; imageResultUri = null; status = "Preparing image upload…"
        processingJob = viewModelScope.launch(Dispatchers.IO) {
            try {
                val s = settings(); hf.discoverSpace(s)
                val target = LocalFileBridge.copyToCache(getApplication(), uri, "phoenix_image")
                val sourceFiles = faces.mapIndexed { i, f -> f?.let { LocalFileBridge.copyToCache(getApplication(), it, "phoenix_image_face_${i + 1}") } }
                progress = .15f; status = "Processing image…"
                val path = uploadGate(s).withPermit { hf.swapImage(s, target, sourceFiles, imageQuality) } ?: throw IllegalStateException("Phoenix returned no image")
                safeDelete(target)
                sourceFiles.forEach { safeDelete(it) }
                val output = downloadToCache(s, path, "phoenix_image_result")
                val published = publishImageResult(output, "Phoenix_Image_${System.currentTimeMillis()}.jpg")
                safeDelete(output)
                imageResultUri = published; progress = 1f; eta = "Done"; status = "Complete — image saved to Pictures/Phoenix"
                addHistory(HistoryItem("IMG-${System.currentTimeMillis()}", "Image", "Phoenix image result", published.toString(), "Done", System.currentTimeMillis(), spaceLabel, s.spaceUrl))
            } catch (e: Throwable) {
                lastError = e.message ?: e.javaClass.simpleName
                status = "Image processing failed: ${lastError}"
            }
            finally { running = false }
        }
    }

    /**
     * Long-running cloud jobs must not depend on the Activity remaining in the
     * foreground. A lightweight foreground service keeps this process alive
     * while the ViewModel polls the remote job and downloads the final file.
     * The server-side render itself continues independently on Hugging Face.
     */
    /**
     * Not START_STICKY. A sticky service that failed on Android 14/15 restarted
     * itself forever and the app "kept stopping". This one only lives while a
     * job is being watched, and a refused start is ignored.
     */
    private fun startBackgroundGuard() {
        // The foreground service was closing the app in a loop on this phone.
        // The Space keeps the job. Resume in History reconnects to it.
    }

    private fun stopBackgroundGuardIfIdle() {
        if (parallelJobs.isEmpty()) {
            runCatching {
                getApplication<Application>().stopService(
                    Intent(getApplication<Application>(), PhoenixKeepAliveService::class.java)
                )
            }
        }
    }

    fun splitFaceModeAt(index: Int): String = when (index) {
        0 -> options.splitFaceMode1
        1 -> options.splitFaceMode2
        2 -> options.splitFaceMode3
        3 -> options.splitFaceMode4
        4 -> options.splitFaceMode5
        5 -> options.splitFaceMode6
        6 -> options.splitFaceMode7
        7 -> options.splitFaceMode8
        8 -> options.splitFaceMode9
        9 -> options.splitFaceMode10
        else -> options.faceMode
    }

    fun setSplitFaceMode(index: Int, value: String) {
        updateOptions { o ->
            when (index) {
                0 -> o.copy(splitFaceMode1 = value)
                1 -> o.copy(splitFaceMode2 = value)
                2 -> o.copy(splitFaceMode3 = value)
                3 -> o.copy(splitFaceMode4 = value)
                4 -> o.copy(splitFaceMode5 = value)
                5 -> o.copy(splitFaceMode6 = value)
                6 -> o.copy(splitFaceMode7 = value)
                7 -> o.copy(splitFaceMode8 = value)
                8 -> o.copy(splitFaceMode9 = value)
                9 -> o.copy(splitFaceMode10 = value)
                else -> o
            }
        }
    }

    private fun splitFaceMode(index: Int): String = splitFaceModeAt(index)

    private fun startSplitFarm(uri: Uri, parts: Int) {
        val free = spaces.filter { it.url.isNotBlank() && !isSpaceBusy(it.url) }
        if (free.size < parts) {
            status = "Split needs $parts free Spaces. ${free.size} idle now. Stop a job or add URLs."
            return
        }
        val faces = listOf(face1, face2, face3, face4)
        if (faces.all { it == null }) { status = "Select or capture at least one replacement face"; return }
        val groupId = "SPLIT-${UUID.randomUUID().toString().take(8)}"
        val winStart = options.trimStart.coerceIn(0, 99)
        val winEnd = options.trimEnd.coerceIn(winStart + 1, 100)
        val span = winEnd - winStart
        setFarm("Split", 1)
        viewModelScope.launch(Dispatchers.IO) {
            val app = getApplication<Application>()
            val totalUs = VideoTrimmer.durationUs(app, uri) ?: 0L
            for (i in 0 until parts) {
                val startPct = winStart + span * i / parts
                val endPct = if (i == parts - 1) winEnd else winStart + span * (i + 1) / parts
                if (endPct <= startPct) continue
                var partUri = uri
                var sendStart = startPct
                var sendEnd = endPct
                var partSecs = options.durationSec
                if (totalUs > 0L) {
                    val startUs = totalUs * startPct / 100L
                    val endUs = totalUs * endPct / 100L
                    partSecs = (((endUs - startUs) / 1_000_000L).toInt() + 3).coerceIn(10, 360)
                    setFarm("Split ${i + 1}/$parts", ((i) * 18 / parts).coerceAtLeast(2))
                    val dest = File(app.cacheDir, "phoenix_part_${groupId}_$i.mp4")
                    val cut = VideoTrimmer.trim(app, uri, startUs, endUs, dest)
                    if (cut != null && cut.file.exists() && cut.durationUs > 0L) {
                        partUri = Uri.fromFile(cut.file)
                        sendStart = ((cut.residualStartUs * 100.0) / cut.durationUs).toInt().coerceIn(0, 5)
                        sendEnd = 100
                        partSecs = ((cut.durationUs / 1_000_000L).toInt() + 3).coerceIn(10, 360)
                    }
                }
                val slot = free[i]
                val mode = if (options.splitSharedSettings) options.faceMode else splitFaceMode(i)
                setFarm("Upload ${i + 1}/$parts → ${slot.label}", 18 + (i * 20 / parts))
                startRemote(
                    partUri,
                    space = HfSettings(slot.url.trim().trimEnd('/'), endpoint.trim().ifBlank { DEFAULT_ENDPOINT }, modelRepo, modelRevision),
                    label = slot.label,
                    trimStart = sendStart,
                    trimEnd = sendEnd,
                    faceMode = mode,
                    groupId = groupId,
                    partIndex = i,
                    partCount = parts,
                    durationSec = partSecs
                )
            }
        }
    }

    fun mergeableGroupId(): String? {
        val groups = history.mapNotNull { it.groupId }.distinct()
        return groups.firstOrNull { gid ->
            val parts = history.filter { it.groupId == gid }
            val n = parts.maxOfOrNull { it.partCount } ?: 0
            n >= 2 && parts.count { it.status == "Done" && !it.uri.isNullOrBlank() } >= n &&
                history.none { it.groupId == gid && it.kind == "Merged" && it.status == "Done" }
        }
    }

    fun mergeSplitGroup(groupId: String? = mergeableGroupId()) {
        val gid = groupId ?: run { status = "No completed split group to merge"; return }
        val parts = history.filter { it.groupId == gid && it.kind == "Video" }
            .sortedBy { it.partIndex }
        val n = parts.maxOfOrNull { it.partCount } ?: 0
        val ready = parts.filter { it.status == "Done" && !it.uri.isNullOrBlank() }
        if (n < 2 || ready.size < n) {
            status = "Merge waits for all $n parts. ${ready.size} ready."
            return
        }
        viewModelScope.launch(Dispatchers.IO) {
            try {
                setFarm("Merge", 90)
                val dest = File(getApplication<Application>().cacheDir, "phoenix_merge_${System.currentTimeMillis()}.mp4")
                val ok = VideoConcat.concat(getApplication(), ready.map { Uri.parse(it.uri) }, dest) { done, total ->
                    val pct = 90 + ((done * 9) / total.coerceAtLeast(1))
                    setFarm("Merge $done/$total", pct)
                }
                if (!ok) {
                    setFarm("Merge failed", farmPercent)
                    status = "Merge failed — parts may differ in codec. Save parts from History and join in an editor."
                    return@launch
                }
                val published = publishResult(dest, "Phoenix_merged_${gid}.mp4")
                safeDelete(dest)
                resultUri = published
                addHistory(
                    HistoryItem(
                        "MERGE-${System.currentTimeMillis()}", "Merged", "Phoenix_merged_${gid}.mp4",
                        published.toString(), "Done", System.currentTimeMillis(),
                        "Merged", "", null, 1f, "Done", gid, 0, n
                    )
                )
                setFarm("Merge done", 100)
                status = "Merged $n parts → Movies/Phoenix"
                toast("Merged video saved to Movies/Phoenix")
            } catch (e: Throwable) {
                status = "Merge failed: ${e.message ?: e.javaClass.simpleName}"
            }
        }
    }

    private fun startRemote(
        uri: Uri,
        space: HfSettings = selectedSettings(),
        label: String = spaceLabel,
        trimStart: Int = options.trimStart,
        trimEnd: Int = options.trimEnd,
        faceMode: String = options.faceMode,
        groupId: String? = null,
        partIndex: Int = 0,
        partCount: Int = 1,
        durationSec: Int = options.durationSec
    ) {
        val targetSpace = space
        if (targetSpace.spaceUrl.isBlank()) { status = "Configure the selected Phoenix Space first"; return }
        if (parallelJobs.size >= MAX_SPACES) { status = "All $MAX_SPACES Phoenix jobs are already active"; return }
        val faces = listOf(face1, face2, face3, face4)
        if (faces.all { it == null }) { status = "Select or capture at least one replacement face"; return }
        val localId = "JOB-${UUID.randomUUID()}"
        val created = System.currentTimeMillis()
        latestJobLocalId = localId
        val partTag = if (partCount > 1) " · part ${partIndex + 1}/$partCount ${trimStart}-${trimEnd}%" else ""
        val sourcePath = if (uri.scheme == "file") uri.path else null
        addHistory(HistoryItem(localId, "Video", "Phoenix job — $label$partTag", null, "Starting", created, label, targetSpace.spaceUrl, null, 0f, "ETA —", groupId, partIndex, partCount, sourcePath))
        val controlJob = Job()
        parallelJobs[localId] = controlJob
        startBackgroundGuard()
        val job = viewModelScope.launch(controlJob + Dispatchers.IO) {
            try {
            while (true) {
            try {
                val cur = history.find { it.id == localId }?.status
                if (cur.isNullOrBlank() || cur == "Starting") {
                    updateHistory(localId, status = "Connecting", progress = .01f, eta = "Checking Space…")
                }
                hf.discoverSpace(targetSpace)
                val alreadyCached = videoUploadCache.containsKey(videoCacheKey(targetSpace, uri, trimStart, trimEnd))
                updateHistory(localId, status = if (alreadyCached) "Reusing upload" else "Uploading video", progress = .02f, eta = if (alreadyCached) "Video already on Space…" else "Uploading…")
                val videoUpload = uploadVideoOnce(targetSpace, uri, trimStart, trimEnd) { written, total ->
                    if (total > 0) {
                        val pct = (written.toFloat() / total).coerceIn(0f, 1f)
                        val doneMb = "%.1f".format(written / 1_048_576.0)
                        val totalMb = "%.1f".format(total / 1_048_576.0)
                        val jobPct = (pct * 100).toInt()
                        updateHistory(localId, progress = pct * 0.35f, eta = "Upload $jobPct% · $doneMb / $totalMb MB")
                        if (partCount > 1) {
                            val base = 18 + (partIndex * 30 / partCount)
                            setFarm("Upload ${partIndex + 1}/$partCount", base + (jobPct * 30 / partCount / 100))
                        } else {
                            setFarm("Upload", (jobPct * 40 / 100).coerceAtLeast(1))
                        }
                    }
                }
                updateHistory(localId, status = "Uploading faces", progress = .04f, eta = "Uploading…")
                // Face uploads are independent. Upload up to three at once; the
                // per-Space semaphore prevents this from overwhelming a Space.
                val uploaded = arrayOfNulls<String>(4)
                val faceJobs = faces.mapIndexedNotNull { i, faceUri ->
                    faceUri?.let { uriFace ->
                        async(Dispatchers.IO) {
                            val mimeFace = getApplication<Application>().contentResolver.getType(uriFace) ?: "image/jpeg"
                            uploaded[i] = uploadGate(targetSpace).withPermit {
                                hf.uploadUri(
                                    targetSpace, getApplication<Application>().contentResolver, uriFace,
                                    "phoenix_face_${localId}_${i + 1}.jpg", mimeFace
                                )
                            }
                        }
                    }
                }
                faceJobs.awaitAll()
                val uploadedRefs = arrayOfNulls<String>(4)
                val refJobs = targetRefs.mapIndexedNotNull { i, refUri ->
                    refUri?.let { uriRef ->
                        async(Dispatchers.IO) {
                            val mimeRef = getApplication<Application>().contentResolver.getType(uriRef) ?: "image/jpeg"
                            uploadedRefs[i] = uploadGate(targetSpace).withPermit {
                                hf.uploadUri(
                                    targetSpace, getApplication<Application>().contentResolver, uriRef,
                                    "phoenix_tref_${localId}_${i + 1}.jpg", mimeRef
                                )
                            }
                        }
                    }
                }
                refJobs.awaitAll()
                updateHistory(localId, status = "Uploading faces", progress = .10f, eta = "Ready to submit")
                // phoenix_submit_video's 6th input is a Textbox (confirmed via
                // Diagnose Space), not a JSON-object-typed input. Putting the
                // JSONObject in directly meant Gradio received an object where
                // it expected a string and stringified it with Python's str(),
                // which produces single-quoted {'key': 'value'} text - exactly
                // what made the server's json.loads() fail with "Expecting
                // property name enclosed in double quotes". Send the actual
                // JSON-encoded string instead.
                val data = JSONArray().put(fileData(videoUpload.path)).put(uploaded[0]?.let(::fileData) ?: JSONObject.NULL)
                    .put(uploaded[1]?.let(::fileData) ?: JSONObject.NULL).put(uploaded[2]?.let(::fileData) ?: JSONObject.NULL)
                    .put(uploaded[3]?.let(::fileData) ?: JSONObject.NULL)
                    .put(optionsJson(videoUpload.trimStartPct, videoUpload.trimEndPct, uploadedRefs, faceMode, durationSec, partIndex, partCount).toString())
                updateHistory(localId, status = "Submitting", progress = .08f, eta = "Queueing…")
                val submitEndpoint = targetSpace.endpoint.ifBlank { DEFAULT_ENDPOINT }
                val submit = hf.callAndWait(targetSpace, submitEndpoint, data, 90_000L)
                val response = submit.optJSONObject(0) ?: throw IllegalStateException("Phoenix returned invalid submit response")
                if (!response.optBoolean("ok", false)) throw IllegalStateException(response.optJSONObject("error")?.optString("message") ?: "Phoenix rejected the job")
                val jobId = response.optString("job_id").takeIf { it.isNotBlank() } ?: throw IllegalStateException("Phoenix returned no job ID")
                updateHistory(localId, name = "Phoenix ${jobId}", status = "Queued", remoteJobId = jobId, progress = .08f, eta = "Queued")
                pollPhoenixJob(targetSpace, jobId, localId)
                break
            } catch (e: kotlinx.coroutines.CancellationException) {
                updateHistory(localId, status = "Monitoring stopped", eta = "Stopped")
                throw e
            } catch (e: Throwable) {
                val existing = history.find { it.id == localId }
                val remoteId = existing?.remoteJobId
                if (!remoteId.isNullOrBlank()) {
                    updateHistory(localId, eta = "No network — will update when back")
                    status = "No network. Job $remoteId stays ${existing?.status ?: "in progress"}. It will update when the signal returns."
                    pollPhoenixJob(targetSpace, remoteId, localId)
                    break
                } else if (isOffline(e)) {
                    val kept = existing?.status?.takeIf { it.isNotBlank() && it != "Failed" } ?: "Uploading video"
                    updateHistory(localId, status = kept, eta = "No network — stays here until the signal is back")
                    status = "No network. This part stays at $kept and continues when the signal returns."
                    waitForNetwork()
                    continue
                } else {
                    lastError = e.message ?: e.javaClass.simpleName
                    updateHistory(localId, status = "Failed", eta = "—", errorName = lastError)
                    status = "Job failed on $spaceLabel: ${lastError}"
                    break
                }
            }
            }
        } finally {
            parallelJobs.remove(localId)
            syncRunning()
        }
        }
        syncRunning()
        status = "${spaceLabel}: job started · ${parallelJobs.size}/${MAX_SPACES} active"
    }

    private fun syncRunning() {
        running = parallelJobs.isNotEmpty()
        activeJobCount = parallelJobs.size
        if (running) startBackgroundGuard() else stopBackgroundGuardIfIdle()
    }

    fun cancelJob(localId: String) {
        parallelJobs[localId]?.cancel()
        updateHistory(localId, status = "Monitoring stopped", eta = "Stopped")
        status = "Stopped monitoring ${history.find { it.id == localId }?.name ?: "job"}. Remote processing may continue on the Space."
    }

    fun cancelProcessing() {
        latestJobLocalId?.let { cancelJob(it) } ?: run { status = "No active Phoenix job" }
    }

    private suspend fun pollPhoenixJob(s: HfSettings, jobId: String, localId: String) {
        var lastProgress = .08f
        var consecutiveFailures = 0
        var backoffMs = 2500L
        // A poll checking on an already-running job is not the job itself - the
        // remote job keeps processing regardless of whether THIS check-in
        // succeeded. Network loss must NEVER mark the job Failed: the Space
        // keeps the result for 3 hours for a later download.
        while (true) {
            delay(backoffMs)
            val r = try {
                val result = hf.callAndWait(s, "phoenix_job_status", JSONArray().put(jobId), 30_000L, allowResubmitOnDeadSession = true)
                consecutiveFailures = 0
                backoffMs = 2500L
                result.optJSONObject(0) ?: throw IllegalStateException("Invalid Phoenix status response")
            } catch (e: kotlinx.coroutines.CancellationException) {
                throw e
            } catch (e: Throwable) {
                consecutiveFailures++
                backoffMs = minOf(60_000L, 2500L * (1L shl minOf(consecutiveFailures, 5)))
                lastError = null
                updateHistory(localId, eta = "No network — will update when back")
                status = "No network. The job stays as it is and will update when the signal returns."
                waitForNetwork()
                continue
            }
            if (!r.optBoolean("ok", false)) {
                val code = r.optJSONObject("error")?.optString("code").orEmpty()
                val why = r.optJSONObject("error")?.optString("message").orEmpty()
                if (code == "not_found") {
                    updateHistory(
                        localId,
                        status = "On server",
                        eta = "Job $jobId not visible now — still checking"
                    )
                    status = "Job $jobId was not returned. Still checking. It is not failed."
                    delay(15_000L)
                    continue
                }
                consecutiveFailures++
                backoffMs = minOf(60_000L, 2500L * (1L shl minOf(consecutiveFailures, 5)))
                updateHistory(
                    localId,
                    eta = "Still checking job $jobId" + if (why.isBlank()) "" else " · ${why.take(80)}"
                )
                status = "Space did not answer for job $jobId. Still watching. ${why}"
                continue
            }
            val state = r.optString("status", "processing")
            val p = r.optInt("progress", -1)
            if (p >= 0) lastProgress = (p.coerceIn(0, 100) / 100f).coerceAtLeast(.08f)
            val etaSeconds = r.optLong("eta_seconds", -1L)
            val etaText = if (etaSeconds >= 0) "ETA ${etaSeconds}s" else "Server processing…"
            val message = r.optString("message", state)
            updateHistory(localId, status = state.replaceFirstChar { it.uppercase() }, progress = lastProgress, eta = etaText)
            status = "${history.find { it.id == localId }?.spaceLabel ?: "Phoenix"}: $message"
            when (state.lowercase()) {
                "done" -> {
                    if (!r.optBoolean("download_ready", false)) throw IllegalStateException("Phoenix completed but no result is ready")
                    val itemNow = history.find { it.id == localId }
                    val resultName = partSaveName(jobId, itemNow?.partIndex ?: 0, itemNow?.partCount ?: 1)
                    val claimed = synchronized(saveLock) {
                        if (alreadySaved(jobId) || jobId in savingJobIds) {
                            false
                        } else {
                            savingJobIds.add(jobId)
                            true
                        }
                    }
                    if (!claimed) {
                        repeat(30) {
                            existingVideoUri(resultName)?.let { resultUri = it; return@repeat }
                            delay(400)
                        }
                        val already = history.find { it.id == localId }?.uri
                        existingVideoUri(resultName)?.let { resultUri = it }
                        updateHistory(localId, name = resultName, uri = already ?: resultUri?.toString(), status = "Done", progress = 1f, eta = "Done")
                        persistHistoryNow()
                        status = "Complete — already saved"
                        return
                    }
                    try {
                    val already = history.find { it.id == localId }?.uri
                    if (!already.isNullOrBlank() || alreadySaved(jobId)) {
                        existingVideoUri(resultName)?.let { resultUri = it }
                        markSaved(jobId)
                        updateHistory(localId, name = resultName, uri = already ?: resultUri?.toString(), status = "Done", progress = 1f, eta = "Done")
                        persistHistoryNow()
                        status = "Complete — already saved"
                        return
                    }
                    existingVideoUri(resultName)?.let { existing ->
                        resultUri = existing
                        markSaved(jobId)
                        updateHistory(localId, name = resultName, uri = existing.toString(), status = "Done", progress = 1f, eta = "Done")
                        persistHistoryNow()
                        status = "Complete — ${history.find { it.id == localId }?.spaceLabel ?: "Phoenix Space"} result saved"
                        return
                    }
                    updateHistory(localId, status = "Downloading", progress = .96f, eta = "Downloading…")

                    // The render itself is already finished server-side at this
                    // point - only the final file fetch remains. A transient
                    // network interruption here ("Software caused connection
                    // abort" is Android suspending a backgrounded app's network
                    // mid-transfer during a long job) should not throw away a
                    // fully completed render just because the last download
                    // attempt hit a blip. Retry this last step, the same way
                    // the status poll above already tolerates one.
                    var lastDownloadError: Throwable? = null
                    var published: Uri? = null
                    var lastProgressPush = 0L
                    for (attempt in 1..12) {
                        try {
                            val download = hf.callAndWait(s, "phoenix_download", JSONArray().put(jobId), 60_000L)
                            val path = extractPath(download) ?: throw IllegalStateException("Phoenix download returned no file")
                            // Real byte-level progress instead of a fixed 96% for
                            // the whole transfer - a large rendered video (easily
                            // 50-300MB) could otherwise sit motionless on screen
                            // for many minutes, indistinguishable from being stuck.
                            val output = downloadToCache(s, path, "phoenix_$jobId") { downloaded, total ->
                                val now = System.currentTimeMillis()
                                if (now - lastProgressPush < 400L && (total <= 0 || downloaded < total)) return@downloadToCache
                                lastProgressPush = now
                                val doneMb = "%.1f".format(downloaded / 1_048_576.0)
                                if (total > 0) {
                                    val pct = (downloaded.toFloat() / total).coerceIn(0f, 1f)
                                    val totalMb = "%.1f".format(total / 1_048_576.0)
                                    updateHistory(localId, progress = 0.96f + pct * 0.03f, eta = "$doneMb / $totalMb MB")
                                    status = "Downloading result — $doneMb of $totalMb MB (${(pct * 100).toInt()}%)"
                                } else {
                                    updateHistory(localId, eta = "$doneMb MB…")
                                    status = "Downloading result — $doneMb MB so far"
                                }
                            }
                            published = publishResult(output, resultName)
                            safeDelete(output)
                            markSaved(jobId)
                            break
                        } catch (e: kotlinx.coroutines.CancellationException) {
                            throw e
                        } catch (e: Throwable) {
                            lastDownloadError = e
                            status = "Render is on the Space — waiting to download (${attempt}/12)…"
                            updateHistory(localId, eta = "No network — download continues when back")
                            delay(4000L * attempt)
                        }
                    }
                    val finalUri = published
                    if (finalUri == null) {
                        lastError = null
                        updateHistory(
                            localId,
                            eta = "Finished on the Space — download continues when the signal is back"
                        )
                        status = "Finished on the Space. Phone was offline. Job $jobId is kept 3 hours — tap Retry download."
                        delay(10_000L)
                        continue
                    }
                    resultUri = finalUri
                    markSaved(jobId)
                    updateHistory(localId, name = resultName, uri = finalUri.toString(), status = "Done", progress = 1f, eta = "Done")
                    persistHistoryNow()
                    status = "Complete — ${history.find { it.id == localId }?.spaceLabel ?: "Phoenix Space"} result saved"
                    val item = history.find { it.id == localId }
                    if (options.autoMergeSplits && item?.groupId != null && mergeableGroupId() == item.groupId) {
                        mergeSplitGroup(item.groupId)
                    }
                    return
                    } finally {
                        synchronized(saveLock) { savingJobIds.remove(jobId) }
                    }
                }
                "error", "failed" -> {
                    val why = r.optString("message").ifBlank { "Phoenix job $state" }
                    lastError = why
                    updateHistory(localId, status = "Failed", eta = why.take(140))
                    status = "Job $jobId failed on the Space: $why"
                    return
                }
                "cancelled" -> {
                    updateHistory(localId, status = "Monitoring stopped", eta = "Cancelled on the Space")
                    status = "Job $jobId was cancelled."
                    return
                }
            }
        }
    }

    private fun updateHistory(
        id: String,
        name: String? = null,
        uri: String? = null,
        status: String? = null,
        remoteJobId: String? = null,
        progress: Float? = null,
        eta: String? = null,
        errorName: String? = null
    ) {
        synchronized(historyLock) {
            val current = history.find { it.id == id } ?: return
            val finalName = if (errorName != null) "${current.name} · $errorName" else name ?: current.name
            history = history.map { if (it.id == id) it.copy(
                name = finalName,
                uri = uri ?: it.uri,
                status = status ?: it.status,
                remoteJobId = remoteJobId ?: it.remoteJobId,
                progress = progress ?: it.progress,
                eta = eta ?: it.eta
            ) else it }
            persistHistory()
        }
    }

    private var networkCallback: ConnectivityManager.NetworkCallback? = null

    private fun isOffline(e: Throwable): Boolean {
        var t: Throwable? = e
        while (t != null) {
            if (t is java.io.IOException) return true
            val m = t.message?.lowercase().orEmpty()
            if ("cannot resolve" in m || "could not reach" in m || "unable to resolve" in m ||
                "failed to connect" in m || "timeout" in m || "timed out" in m ||
                "connection abort" in m || "connection reset" in m || "network is unreachable" in m ||
                "stream was reset" in m || "software caused connection abort" in m
            ) return true
            t = t.cause
        }
        return false
    }

    /** Pause until the phone has a network again, or one minute, whichever is first. */
    private suspend fun waitForNetwork() {
        val cm = getApplication<Application>().getSystemService(ConnectivityManager::class.java)
        val online = cm?.activeNetwork?.let { n ->
            cm.getNetworkCapabilities(n)?.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET) == true
        } == true
        if (!online && cm != null) {
            withTimeoutOrNull(60_000L) {
                suspendCancellableCoroutine { cont ->
                    val cb = object : ConnectivityManager.NetworkCallback() {
                        override fun onAvailable(network: Network) {
                            runCatching { cm.unregisterNetworkCallback(this) }
                            if (cont.isActive) cont.resume(Unit)
                        }
                    }
                    val req = NetworkRequest.Builder()
                        .addCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET)
                        .build()
                    runCatching { cm.registerNetworkCallback(req, cb) }
                    cont.invokeOnCancellation { runCatching { cm.unregisterNetworkCallback(cb) } }
                }
            }
        }
        delay(1_500L)
    }

    private fun watchNetwork() {
        val cm = getApplication<Application>().getSystemService(ConnectivityManager::class.java) ?: return
        if (networkCallback != null) return
        val cb = object : ConnectivityManager.NetworkCallback() {
            override fun onAvailable(network: Network) {
                resumeTrackedJobs()
            }
        }
        val req = NetworkRequest.Builder()
            .addCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET)
            .build()
        runCatching { cm.registerNetworkCallback(req, cb) }
        networkCallback = cb
    }

    override fun onCleared() {
        val cm = getApplication<Application>().getSystemService(ConnectivityManager::class.java)
        networkCallback?.let { cb -> runCatching { cm?.unregisterNetworkCallback(cb) } }
        networkCallback = null
        super.onCleared()
    }

    private fun resumeTrackedJobs() {
        val resumable = history.filter {
            it.kind == "Video" &&
                !it.remoteJobId.isNullOrBlank() &&
                it.status != "Done" &&
                it.status != "Monitoring stopped"
        }.take(MAX_SPACES)
        if (resumable.isNotEmpty()) startBackgroundGuard()
        resumable.forEach { item ->
            if (parallelJobs.containsKey(item.id)) return@forEach
            if (parallelJobs.size >= MAX_SPACES) return@forEach
            val remoteId = item.remoteJobId ?: return@forEach
            if (alreadySaved(remoteId) || !item.uri.isNullOrBlank()) {
                updateHistory(item.id, status = "Done", progress = 1f, eta = "Done")
                return@forEach
            }
            val url = item.spaceUrl.trim().trimEnd('/')
            if (url.isBlank()) return@forEach
            val settings = HfSettings(url, endpoint.trim().ifBlank { DEFAULT_ENDPOINT }, modelRepo, modelRevision)
            val controlJob = Job()
            parallelJobs[item.id] = controlJob
            viewModelScope.launch(controlJob + Dispatchers.IO) {
                try { pollPhoenixJob(settings, remoteId, item.id) }
                catch (e: kotlinx.coroutines.CancellationException) { updateHistory(item.id, status = "Monitoring stopped", eta = "Stopped") }
                catch (e: Throwable) {
                    updateHistory(item.id, eta = "No network — will update when back")
                }
                finally { parallelJobs.remove(item.id); syncRunning() }
            }
        }
        syncRunning()
    }

    // trimStartPct/trimEndPct are what the SERVER should still trim. When the
    // clip was cut locally these are the tiny sync-frame residue (often 0 /
    // 100); when it wasn't, they're the user's original selection and the
    // server trims exactly as it always did.
    /** Phoenix_<real job id>-1.mp4 so a split sorts by the part number. */
    private fun partSaveName(jobId: String, partIndex: Int, partCount: Int): String {
        return if (partCount > 1) "Phoenix_${jobId}-${partIndex + 1}.mp4" else "Phoenix_${jobId}.mp4"
    }

    private fun optionsJson(
        trimStartPct: Int = options.trimStart,
        trimEndPct: Int = options.trimEnd,
        targetRefPaths: Array<String?> = arrayOfNulls(4),
        faceModeOverride: String? = null,
        durationSecOverride: Int? = null,
        partIndex: Int = 0,
        partCount: Int = 1
    ) = JSONObject().apply {
        put("secs", durationSecOverride ?: options.durationSec); put("fps", options.fps); put("res", options.resolution); put("quality", options.quality)
        put("enhancer", options.enhancer); put("swap_n", options.swapEvery); put("det_n", options.detectEvery); put("det_int", options.detectInterval)
        put("password", options.password); put("trim_start", trimStartPct); put("trim_end", trimEndPct)
        // The Face mode Dropdown is a separate control from which face tiles
        // are actually filled - attach 2 replacement faces without touching
        // it, and it silently stays on its default "1 face (fastest)". The
        // server then only ever attempts a single-face swap regardless of
        // how many faces were uploaded, which is what caused one replacement
        // face to get pasted onto everyone in frame. Only override when the
        // current selection would understate reality; an intentional "All
        // detected faces" choice (or a correct "2 faces" one) is left alone.
        val filledFaces = listOf(face1, face2, face3, face4).count { it != null }
        val chosen = faceModeOverride ?: options.faceMode
        val effectiveFaceMode = if (filledFaces >= 2 && chosen.startsWith("1")) {
            if (filledFaces == 2) "2 faces" else "All detected faces"
        } else chosen
        put("face_mode", effectiveFaceMode)
        put("enhance_scope", options.enhanceScope); put("device_mode", options.deviceMode)
        put("primary_slot", options.primarySlot); put("smooth_motion", options.smoothMotion)
        if (partCount > 1) {
            put("part_index", partIndex + 1)
            put("part_count", partCount)
        }
        val refs = JSONArray()
        targetRefPaths.forEach { p -> if (p.isNullOrBlank()) refs.put(JSONObject.NULL) else refs.put(p) }
        put("target_refs", refs)
    }

    private fun fileData(path: String) = JSONObject().put("path", path).put("meta", JSONObject().put("_type", "gradio.FileData"))
    private fun extractPath(data: JSONArray): String? {
        fun walk(v: Any?): String? = when (v) {
            is String -> v.takeIf { it.contains("/") || it.startsWith("http://") || it.startsWith("https://") }
            is JSONObject -> v.optString("path").takeIf { it.isNotBlank() }
            is JSONArray -> (0 until v.length()).asSequence().map { walk(v.opt(it)) }.firstOrNull { it != null }
            else -> null
        }
        return walk(data)
    }

    private fun downloadToCache(
        s: HfSettings,
        path: String,
        prefix: String,
        onProgress: ((Long, Long) -> Unit)? = null
    ): File = File(getApplication<Application>().cacheDir, "${prefix}_${System.currentTimeMillis()}")
        .also { hf.downloadOutput(s, path, it, onProgress) }

    /**
     * Every uploaded source file and every downloaded intermediate/result file
     * was landing in the app's cache directory with nothing ever deleting it -
     * meaning every video and face ever processed accumulated on-device
     * indefinitely, well past the point any of it was still needed. Source
     * uploads are safe to delete the moment the upload call returns (nothing
     * references the local copy after that); result downloads are safe to
     * delete the moment they've been copied into MediaStore.
     */
    private fun safeDelete(file: File?) {
        if (file == null) return
        runCatching { file.delete() }
    }

    private fun safeDelete(uri: Uri?) {
        val path = uri?.takeIf { it.scheme == "file" }?.path ?: return
        safeDelete(File(path))
    }

    /** Manual, on-demand sweep for anything from before this cleanup existed,
     *  or any edge case the per-job cleanup didn't catch. Never touches
     *  MediaStore results (Movies/Phoenix, Pictures/Phoenix) - only this
     *  app's own private cache directory. */
    fun clearLocalCache() {
        val dir = getApplication<Application>().cacheDir
        val files = dir.listFiles()?.filter { it.name.startsWith("phoenix_") || it.name.startsWith("frame_") || it.name.startsWith("detected_") || it.name.startsWith("image_detected_") } ?: emptyList()
        var freedBytes = 0L
        files.forEach { freedBytes += it.length(); safeDelete(it) }
        detectedFrameUri = null
        detectedFaces = List(4) { null }
        val freedMb = freedBytes / 1_048_576.0
        status = if (files.isEmpty()) "No cached files to clear" else "Cleared %.1f MB of cached files".format(freedMb)
    }

    private fun alreadySaved(jobId: String): Boolean {
        if (jobId.isBlank()) return false
        return prefs.getStringSet(SAVED_JOBS_KEY, emptySet())?.contains(jobId) == true
    }

    private fun markSaved(jobId: String) {
        if (jobId.isBlank()) return
        val next = (prefs.getStringSet(SAVED_JOBS_KEY, emptySet()) ?: emptySet()).toMutableSet()
        next.add(jobId)
        prefs.edit().putStringSet(SAVED_JOBS_KEY, next).commit()
    }

    private fun existingVideoUri(name: String): Uri? {
        val resolver = getApplication<Application>().contentResolver
        val q = resolver.query(
            MediaStore.Video.Media.EXTERNAL_CONTENT_URI,
            arrayOf(MediaStore.Video.Media._ID),
            "${MediaStore.Video.Media.DISPLAY_NAME}=?",
            arrayOf(name),
            "${MediaStore.Video.Media.DATE_ADDED} DESC"
        ) ?: return null
        q.use { c ->
            if (!c.moveToFirst()) return null
            val id = c.getLong(0)
            return android.content.ContentUris.withAppendedId(MediaStore.Video.Media.EXTERNAL_CONTENT_URI, id)
        }
    }

    private fun persistHistoryNow() {
        persistHistory(commit = true)
    }

    private fun publishResult(source: File, name: String): Uri {
        synchronized(saveLock) {
        existingVideoUri(name)?.let { return it }
        outputTreeUri?.let { tree ->
            val custom = runCatching { writeToCustomTree(tree, source, name, "video/mp4") }.getOrNull()
            if (custom != null) return custom
            // Falls through to the default MediaStore path below - a
            // revoked permission or a folder the user deleted must not
            // fail an already-fully-rendered job.
        }
        val resolver = getApplication<Application>().contentResolver
        val values = ContentValues().apply {
            put(MediaStore.Video.Media.DISPLAY_NAME, name); put(MediaStore.Video.Media.MIME_TYPE, "video/mp4")
            put(MediaStore.Video.Media.RELATIVE_PATH, Environment.DIRECTORY_MOVIES + "/Phoenix"); if (Build.VERSION.SDK_INT >= 29) put(MediaStore.Video.Media.IS_PENDING, 1)
        }
        val uri = resolver.insert(MediaStore.Video.Media.EXTERNAL_CONTENT_URI, values) ?: throw IllegalStateException("Could not create output media")
        // insert() above already created a real, persisted file entry - if
        // anything below fails, that entry is now an orphan unless we remove
        // it ourselves. Left behind, a retried publishResult() call creates a
        // second entry alongside it: two files in the gallery for one job.
        try {
            resolver.openOutputStream(uri)?.use { source.inputStream().use { input -> input.copyTo(it) } } ?: throw IllegalStateException("Could not write output")
            if (Build.VERSION.SDK_INT >= 29) resolver.update(uri, ContentValues().apply { put(MediaStore.Video.Media.IS_PENDING, 0) }, null, null)
        } catch (e: Throwable) {
            runCatching { resolver.delete(uri, null, null) }
            throw e
        }
        return uri
        }
    }

    private fun publishImageResult(source: File, name: String): Uri {
        outputTreeUri?.let { tree ->
            val custom = runCatching { writeToCustomTree(tree, source, name, "image/jpeg") }.getOrNull()
            if (custom != null) return custom
        }
        val resolver = getApplication<Application>().contentResolver
        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, name); put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
            put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/Phoenix"); if (Build.VERSION.SDK_INT >= 29) put(MediaStore.Images.Media.IS_PENDING, 1)
        }
        val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values) ?: throw IllegalStateException("Could not create output image")
        resolver.openOutputStream(uri)?.use { source.inputStream().use { input -> input.copyTo(it) } } ?: throw IllegalStateException("Could not write output image")
        if (Build.VERSION.SDK_INT >= 29) resolver.update(uri, ContentValues().apply { put(MediaStore.Images.Media.IS_PENDING, 0) }, null, null)
        return uri
    }

    /**
     * Writes [source] into the user-chosen SAF tree, returning the new
     * document's URI, or null if the tree is no longer usable (permission
     * revoked, folder deleted, etc.) so the caller falls back to the
     * default MediaStore location instead of failing the job outright.
     */
    private fun writeToCustomTree(tree: Uri, source: File, name: String, mime: String): Uri? {
        val app = getApplication<Application>()
        val hasPermission = app.contentResolver.persistedUriPermissions.any {
            it.uri == tree && it.isWritePermission
        }
        if (!hasPermission) return null
        val parentDoc = DocumentsContract.buildDocumentUriUsingTree(tree, DocumentsContract.getTreeDocumentId(tree))
        val newDocUri = DocumentsContract.createDocument(app.contentResolver, parentDoc, mime, name) ?: return null
        // createDocument() above already created a real, persisted file - if
        // the copy fails, that document is now an orphan unless removed here.
        try {
            app.contentResolver.openOutputStream(newDocUri)?.use { out ->
                source.inputStream().use { input -> input.copyTo(out) }
            } ?: throw IllegalStateException("Could not open output stream")
        } catch (e: Throwable) {
            runCatching { DocumentsContract.deleteDocument(app.contentResolver, newDocUri) }
            return null
        }
        return newDocUri
    }

    fun removeHistory(id: String) { history.find { it.id == id }?.uri?.let { runCatching { getApplication<Application>().contentResolver.delete(Uri.parse(it), null, null) } }; synchronized(historyLock) { history = history.filterNot { it.id == id }; persistHistory() } }
    /**
     * Copies an already-completed job's video (saved in Movies/Phoenix at job
     * completion) into the public Downloads/Phoenix folder, so it shows up in
     * the Downloads app like any other download - distinct from just being
     * able to view it in place.
     */
    /**
     * Toast for a fire-and-forget confirmation the user might otherwise miss.
     * A status-bar text change alone isn't enough for something like a
     * background download finishing - it's easy to be scrolled past the
     * header, or not looking at it at that exact moment, and a silently
     * updated string gives no cue that anything happened at all.
     */
    private suspend fun toast(message: String) = withContext(Dispatchers.Main) {
        Toast.makeText(getApplication(), message, Toast.LENGTH_LONG).show()
    }

    fun downloadToDownloads(item: HistoryItem) {
        val sourceUriString = item.uri
        if (sourceUriString.isNullOrBlank()) { status = "No saved file for this job"; return }
        if (Build.VERSION.SDK_INT < 29) {
            status = "Saving to Downloads needs Android 10+. Your video is already in Movies/Phoenix."
            Toast.makeText(getApplication(), "Can't save to Downloads on this Android version — already in Movies/Phoenix", Toast.LENGTH_LONG).show()
            return
        }
        viewModelScope.launch(Dispatchers.IO) {
            try {
                val resolver = getApplication<Application>().contentResolver
                val sourceUri = Uri.parse(sourceUriString)
                val values = ContentValues().apply {
                    put(MediaStore.Downloads.DISPLAY_NAME, item.name)
                    put(MediaStore.Downloads.MIME_TYPE, "video/mp4")
                    put(MediaStore.Downloads.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS + "/Phoenix")
                    put(MediaStore.Downloads.IS_PENDING, 1)
                }
                val destUri = resolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values)
                    ?: throw IllegalStateException("Could not create a download entry")
                resolver.openInputStream(sourceUri)?.use { input ->
                    resolver.openOutputStream(destUri)?.use { output -> input.copyTo(output) }
                        ?: throw IllegalStateException("Could not write the download")
                } ?: throw IllegalStateException("Could not read the saved video")
                resolver.update(destUri, ContentValues().apply { put(MediaStore.Downloads.IS_PENDING, 0) }, null, null)
                status = "Saved to Downloads/Phoenix — ${item.name}"
                toast("Downloaded to Downloads/Phoenix: ${item.name}")
            } catch (e: Throwable) {
                val detail = e.message ?: e.javaClass.simpleName
                status = "Download failed: $detail"
                toast("Download failed: $detail")
            }
        }
    }

    fun clearHistory() { history.forEach { it.uri?.let { u -> runCatching { getApplication<Application>().contentResolver.delete(Uri.parse(u), null, null) } } }; synchronized(historyLock) { history = emptyList(); persistHistory() }; status = "History cleared" }
    fun refreshHistory() {
        synchronized(historyLock) { history = loadHistory() }
        cleanupHistoryMetadata()
        status = "History refreshed. Resume keeps the Space job. Redo part runs only that piece again."
    }

    fun retryJob(localId: String) {
        val item = history.find { it.id == localId } ?: return
        val remoteId = item.remoteJobId
        if (remoteId.isNullOrBlank()) {
            status = "This part never reached the Space. Use Redo part."
            return
        }
        if (parallelJobs.containsKey(localId)) {
            status = "Already watching ${item.name}"
            return
        }
        status = "Watching job $remoteId again. The card stays as it is until the Space answers."
        resumeOne(item)
    }

    fun redoSplitPart(localId: String) {
        val item = history.find { it.id == localId } ?: return
        if (item.partCount < 2 || item.groupId.isNullOrBlank()) {
            status = "Redo part is only for one piece of a split."
            return
        }
        if (listOf(face1, face2, face3, face4).all { it == null }) {
            status = "Put the same replacement faces in the slots, then tap Redo part."
            return
        }
        val file = resolvePartFile(item)
        if (file == null) {
            status = "That part clip was cleared when the app closed. Run the split again from the original video."
            return
        }
        parallelJobs[localId]?.cancel()
        updateHistory(localId, status = "Replaced", eta = "Running this part again")
        val space = HfSettings(
            item.spaceUrl.trim().trimEnd('/'),
            endpoint.trim().ifBlank { DEFAULT_ENDPOINT },
            modelRepo,
            modelRevision
        )
        startRemote(
            Uri.fromFile(file),
            space = space,
            label = item.spaceLabel,
            trimStart = 0,
            trimEnd = 100,
            groupId = item.groupId,
            partIndex = item.partIndex,
            partCount = item.partCount,
            durationSec = 360
        )
        status = "Part ${item.partIndex + 1} of ${item.partCount} started again. Parts that already finished are kept."
    }

    private fun resolvePartFile(item: HistoryItem): File? {
        val named = item.sourcePath?.let { File(it) }?.takeIf { it.isFile && it.length() > 0L }
        if (named != null) return named
        val gid = item.groupId ?: return null
        val guess = File(getApplication<Application>().cacheDir, "phoenix_part_${gid}_${item.partIndex}.mp4")
        return guess.takeIf { it.isFile && it.length() > 0L }
    }

    private fun resumeOne(item: HistoryItem) {
        if (parallelJobs.containsKey(item.id)) return
        if (parallelJobs.size >= MAX_SPACES) return
        val remoteId = item.remoteJobId ?: return
        if (alreadySaved(remoteId) || !item.uri.isNullOrBlank()) {
            updateHistory(item.id, status = "Done", progress = 1f, eta = "Done")
            return
        }
        val url = item.spaceUrl.trim().trimEnd('/')
        if (url.isBlank()) return
        val settings = HfSettings(url, endpoint.trim().ifBlank { DEFAULT_ENDPOINT }, modelRepo, modelRevision)
        val controlJob = Job()
        parallelJobs[item.id] = controlJob
        viewModelScope.launch(controlJob + Dispatchers.IO) {
            try { pollPhoenixJob(settings, remoteId, item.id) }
            catch (e: kotlinx.coroutines.CancellationException) { updateHistory(item.id, status = "Monitoring stopped", eta = "Stopped") }
            catch (e: Throwable) {
                updateHistory(item.id, eta = "No network — will update when back")
            }
            finally { parallelJobs.remove(item.id); syncRunning() }
        }
        syncRunning()
    }

    private fun addHistory(item: HistoryItem) { synchronized(historyLock) { history = (listOf(item) + history).take(50); persistHistory() } }
    private fun persistHistory(commit: Boolean = false) {
        val arr = JSONArray(); history.forEach { h -> arr.put(JSONObject().put("id", h.id).put("kind", h.kind).put("name", h.name).put("uri", h.uri ?: JSONObject.NULL).put("status", h.status).put("createdAt", h.createdAt).put("spaceLabel", h.spaceLabel).put("spaceUrl", h.spaceUrl).put("remoteJobId", h.remoteJobId ?: JSONObject.NULL).put("progress", h.progress).put("eta", h.eta).put("groupId", h.groupId ?: JSONObject.NULL).put("partIndex", h.partIndex).put("partCount", h.partCount).put("sourcePath", h.sourcePath ?: JSONObject.NULL)) }
        val ed = prefs.edit().putString(HISTORY_KEY, arr.toString())
        if (commit) ed.commit() else ed.apply()
    }
    private fun loadHistory(): List<HistoryItem> = runCatching {
        val arr = JSONArray(prefs.getString(HISTORY_KEY, "[]")); (0 until arr.length()).map { i -> val o = arr.getJSONObject(i); HistoryItem(
            o.optString("id"), o.optString("kind"), o.optString("name"), o.optString("uri").takeIf { it.isNotBlank() && it != "null" },
            o.optString("status"), o.optLong("createdAt"), o.optString("spaceLabel", "Space 1"), o.optString("spaceUrl", ""),
            o.optString("remoteJobId").takeIf { it.isNotBlank() && it != "null" }, o.optDouble("progress", 0.0).toFloat(), o.optString("eta", "ETA —"),
            o.optString("groupId").takeIf { it.isNotBlank() && it != "null" }, o.optInt("partIndex", 0), o.optInt("partCount", 1),
            o.optString("sourcePath").takeIf { it.isNotBlank() && it != "null" }
        ) }
    }.getOrDefault(emptyList())
    private fun cleanupHistoryMetadata() { synchronized(historyLock) { history = history.filter { it.uri == null || runCatching { getApplication<Application>().contentResolver.openAssetFileDescriptor(Uri.parse(it.uri), "r") != null }.getOrDefault(false) }; persistHistory() } }
}
