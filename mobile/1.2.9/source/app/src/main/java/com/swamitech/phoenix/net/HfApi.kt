package com.swamitech.phoenix.net

import android.content.ContentResolver
import android.net.Uri
import com.swamitech.phoenix.HfSettings
import okhttp3.Dns
import okhttp3.HttpUrl.Companion.toHttpUrl
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody
import okhttp3.RequestBody.Companion.asRequestBody
import okhttp3.RequestBody.Companion.toRequestBody
import okio.Buffer
import okio.BufferedSink
import okio.ForwardingSink
import okio.buffer
import okhttp3.dnsoverhttps.DnsOverHttps
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.net.InetAddress
import java.net.UnknownHostException
import java.util.concurrent.TimeUnit
import java.util.concurrent.ConcurrentHashMap

/** Transport for the Phoenix Hugging Face Space. */
class HfApi(private val tokenProvider: () -> String?) {

    // HF/Gradio route discovery is metadata, not job state. Cache it briefly so
    // every image/video operation does not pay the /info -> /config probing
    // sequence again. A short TTL still recovers from Space redeploys.
    private data class CacheEntry<T>(val atMs: Long, val value: T)
    private val discoveryCache = ConcurrentHashMap<String, CacheEntry<String>>()
    private val endpointCache = ConcurrentHashMap<String, CacheEntry<List<String>>>()
    private val signatureCache = ConcurrentHashMap<String, CacheEntry<Map<String, String>>>()
    private val routeCacheTtlMs = 30 * 60 * 1000L

    fun clearDiscoveryCache(settings: HfSettings) {
        val key = base(settings)
        discoveryCache.remove(key)
        endpointCache.remove(key)
        signatureCache.remove(key)
    }

    private fun <T> cached(cache: ConcurrentHashMap<String, CacheEntry<T>>, key: String): T? {
        val hit = cache[key] ?: return null
        return if (System.currentTimeMillis() - hit.atMs < routeCacheTtlMs) hit.value else {
            cache.remove(key, hit)
            null
        }
    }

    private fun <T> putCached(cache: ConcurrentHashMap<String, CacheEntry<T>>, key: String, value: T): T {
        cache[key] = CacheEntry(System.currentTimeMillis(), value)
        return value
    }

    /**
     * Uploads previously had zero progress visibility - the UI set a fixed
     * "2%" once when the upload started and never moved it again, no matter
     * whether the transfer took 5 seconds or 20+ minutes on a poor
     * connection. Combined with a generous 15-minute write timeout (any
     * trickle of bytes keeps OkHttp from calling it a failure), a slow
     * upload was indistinguishable from a frozen one. This mirrors the same
     * fix already applied to downloads: wrap the request body so real bytes-
     * written can be reported as they go.
     */
    private class ProgressRequestBody(
        private val delegate: RequestBody,
        private val onProgress: (written: Long, total: Long) -> Unit
    ) : RequestBody() {
        override fun contentType() = delegate.contentType()
        override fun contentLength() = delegate.contentLength()
        override fun writeTo(sink: BufferedSink) {
            val total = contentLength()
            var written = 0L
            val countingSink = object : ForwardingSink(sink) {
                override fun write(source: Buffer, byteCount: Long) {
                    super.write(source, byteCount)
                    written += byteCount
                    onProgress(written, total)
                }
            }
            val buffered = countingSink.buffer()
            delegate.writeTo(buffered)
            buffered.flush()
        }
    }

    /**
     * Repeated "Cannot resolve Phoenix server" failures - including ones that
     * persisted through the full 24-attempt poll-retry budget - point at the
     * carrier/Wi-Fi DNS resolver itself, not at anything the app controls.
     * Rather than widen retry budgets further, fall back to resolving via
     * DNS-over-HTTPS (Cloudflare) whenever the device's system resolver fails.
     * This routes around a broken or blackholing local resolver instead of
     * just waiting longer for it to recover.
     *
     * Bootstrapped with literal IPs (no DNS lookup needed to reach them) so
     * there is no chicken-and-egg dependency on the resolver we are working
     * around.
     */
    private val dohBootstrapClient = OkHttpClient.Builder()
        .connectTimeout(10, TimeUnit.SECONDS)
        .readTimeout(10, TimeUnit.SECONDS)
        .build()

    private val dohFallback: Dns by lazy {
        DnsOverHttps.Builder()
            .client(dohBootstrapClient)
            .url("https://cloudflare-dns.com/dns-query".toHttpUrl())
            .bootstrapDnsHosts(
                InetAddress.getByName("1.1.1.1"),
                InetAddress.getByName("1.0.0.1")
            )
            .build()
    }

    private val resilientDns = object : Dns {
        override fun lookup(hostname: String): List<InetAddress> {
            return try {
                Dns.SYSTEM.lookup(hostname)
            } catch (systemFailure: UnknownHostException) {
                try {
                    dohFallback.lookup(hostname)
                } catch (dohFailure: Exception) {
                    // Both paths failed - surface the original system error,
                    // since that's the one existing error classification and
                    // messaging already understands.
                    throw systemFailure
                }
            }
        }
    }

    private val http = OkHttpClient.Builder()
        .dns(resilientDns)
        .connectTimeout(45, TimeUnit.SECONDS)
        .readTimeout(5, TimeUnit.MINUTES)
        .writeTimeout(15, TimeUnit.MINUTES)
        .build()

    private fun base(settings: HfSettings): String = settings.spaceUrl.trim().trimEnd('/')

    private fun auth(builder: Request.Builder): Request.Builder {
        val token = tokenProvider()?.trim().orEmpty()
        if (token.isNotEmpty()) builder.header("Authorization", "Bearer $token")
        return builder
    }

    private fun routeCandidates(path: String): List<String> = listOf("/gradio_api$path", path).distinct()

    private fun friendlyNetworkError(error: Throwable, host: String): IllegalStateException {
        val cause = error.cause
        val isDns = error is UnknownHostException || cause is UnknownHostException
        val isConnectFailure = error is java.net.ConnectException || cause is java.net.ConnectException ||
            error is java.net.SocketTimeoutException || cause is java.net.SocketTimeoutException
        return when {
            isDns -> IllegalStateException(
                "Cannot resolve Phoenix server '$host'. Android DNS lookup failed. " +
                    "Check Wi‑Fi/mobile data, Private DNS, VPN/ad-blocking DNS and try again."
            )
            isConnectFailure -> IllegalStateException(
                "Could not reach Phoenix server '$host' - the connection timed out or was refused " +
                    "after retrying. This is usually a brief network drop (Wi-Fi/mobile data switch, " +
                    "weak signal) rather than a problem with the Space itself. Check your connection " +
                    "and try again."
            )
            else -> IllegalStateException(error.message ?: error.javaClass.simpleName)
        }
    }

    /**
     * Runs one HTTP call with a short, bounded retry for DNS resolution
     * failures only. A mobile resolver hiccup - a Wi-Fi/cellular handover,
     * a Private DNS blip - can fail one lookup and succeed the very next one
     * a second later; retrying absorbs exactly that instead of failing an
     * entire detection or swap job over a one-off glitch. Every other error
     * (timeout, HTTP error status, TLS failure) is not retried here and
     * surfaces immediately, unchanged from before.
     */
    /**
     * Widened from DNS-only. UnknownHostException was the first failure mode
     * found, but ConnectException ("failed to connect to /IP:443") and
     * SocketTimeoutException (a connect that started but never completed) are
     * a distinct, at least as common, class of brief network blip - a cell
     * tower handover, momentary carrier congestion, a Wi-Fi/mobile-data
     * switch mid-request. None of those are DNS problems, so none were ever
     * retried here before; every one failed on the very first attempt.
     * execute() only ever throws IOException (or a subclass) for a genuine
     * network-level failure - an HTTP error status comes back as a normal
     * Response, not an exception - so retrying on any IOException is safe
     * and correctly scoped: it can never mask a real API error, only a
     * connection that didn't succeed this one attempt.
     */
    private fun executeWithRetry(request: Request, attempts: Int = 3): okhttp3.Response {
        var lastError: Throwable? = null
        repeat(attempts) { attempt ->
            try {
                return http.newCall(request).execute()
            } catch (e: Throwable) {
                val isRetryable = e is java.io.IOException
                lastError = e
                if (!isRetryable || attempt == attempts - 1) throw e
                Thread.sleep(600L * (attempt + 1))
            }
        }
        throw lastError ?: IllegalStateException("Network call failed")
    }

    fun discoverSpace(settings: HfSettings): String {
        val baseUrl = base(settings)
        if (baseUrl.isBlank()) throw IllegalStateException("Phoenix Space URL is empty")
        cached(discoveryCache, baseUrl)?.let { return it }
        var lastDetail = "HTTP 404"
        val candidates = listOf("/gradio_api/info", "/info", "/config", "/gradio_api/openapi.json", "/openapi.json")
        try {
            for (path in candidates) {
                runCatching {
                    val request = auth(Request.Builder().url(baseUrl + path).get()).build()
                    executeWithRetry(request).use { response ->
                        val body = response.body?.string().orEmpty()
                        if (response.isSuccessful) return putCached(discoveryCache, baseUrl, body.ifBlank { "{}" })
                        lastDetail = "HTTP ${response.code}: ${body.take(180)}"
                    }
                }.onFailure { lastDetail = it.message ?: it.javaClass.simpleName }
            }
            val request = auth(Request.Builder().url(baseUrl).get()).build()
            executeWithRetry(request).use { response ->
                if (response.isSuccessful || response.code in 300..399 || response.code == 401 || response.code == 403) {
                    return putCached(discoveryCache, baseUrl, "{\"reachable\":true,\"discovery\":false,\"detail\":${JSONObject.quote(lastDetail)}}")
                }
                lastDetail = "HTTP ${response.code}: ${response.body?.string()?.take(180).orEmpty()}"
            }
        } catch (e: Throwable) {
            throw friendlyNetworkError(e, baseUrl.removePrefix("https://").removePrefix("http://"))
        }
        throw IllegalStateException("Phoenix Space is not reachable. Last discovery result: $lastDetail")
    }

    fun discover(settings: HfSettings): String = discoverSpace(settings)

    /**
     * Returns every named API route the Space actually publishes.
     *
     * A generic "detection failed" almost always means the Space does not expose
     * the endpoint the client asked for, so listing what IS there turns a dead
     * end into a diagnosis.
     */
    fun listEndpoints(settings: HfSettings): List<String> {
        val baseUrl = base(settings)
        if (baseUrl.isBlank()) throw IllegalStateException("Phoenix Space URL is empty")
        cached(endpointCache, baseUrl)?.let { return it }
        val names = linkedSetOf<String>()
        var reachable = false

        for (path in listOf("/gradio_api/info", "/info", "/config", "/gradio_api/config")) {
            val body = runCatching {
                val request = auth(Request.Builder().url(baseUrl + path).get()).build()
                executeWithRetry(request).use { response ->
                    if (response.isSuccessful) response.body?.string().orEmpty() else null
                }
            }.getOrNull() ?: continue

            reachable = true
            runCatching {
                val root = JSONObject(body)

                // Gradio /info shape
                root.optJSONObject("named_endpoints")?.let { endpoints ->
                    val keys = endpoints.keys()
                    while (keys.hasNext()) {
                        val key = keys.next()
                        if (key.isNotBlank()) names.add(key.trim().trimStart('/'))
                    }
                }

                // Gradio /config shape
                root.optJSONArray("dependencies")?.let { deps ->
                    for (i in 0 until deps.length()) {
                        val apiName = deps.optJSONObject(i)?.optString("api_name").orEmpty()
                        if (apiName.isNotBlank() && apiName != "null") {
                            names.add(apiName.trim().trimStart('/'))
                        }
                    }
                }
            }
            if (names.isNotEmpty()) break
        }

        if (!reachable) {
            throw IllegalStateException(
                "Could not read the Space API description. Check the URL and, for a private Space, the HF token."
            )
        }
        return putCached(endpointCache, baseUrl, names.sorted())
    }

    /**
     * Streams a content:// URI directly to OkHttp. This avoids copying a large
     * gallery video/image into the app cache before the network transfer.
     * Providers that expose a length let OkHttp send Content-Length; virtual
     * providers may return -1 and are still streamed correctly.
     */
    private class ContentUriRequestBody(
        private val resolver: ContentResolver,
        private val uri: Uri,
        private val mime: String
    ) : RequestBody() {
        override fun contentType() = mime.toMediaType()

        override fun contentLength(): Long = try {
            resolver.openAssetFileDescriptor(uri, "r")?.use { it.length } ?: -1L
        } catch (_: Throwable) {
            -1L
        }

        override fun writeTo(sink: BufferedSink) {
            resolver.openInputStream(uri).use { input ->
                requireNotNull(input) { "Unable to open selected file" }
                input.copyTo(sink.outputStream(), 256 * 1024)
            }
        }
    }

    fun uploadUri(
        settings: HfSettings,
        resolver: ContentResolver,
        uri: Uri,
        filename: String,
        mime: String,
        onProgress: ((Long, Long) -> Unit)? = null
    ): String {
        val fileBody = ContentUriRequestBody(resolver, uri, mime)
        val trackedBody: RequestBody = if (onProgress != null) ProgressRequestBody(fileBody, onProgress) else fileBody
        val requestBody = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart("files", filename, trackedBody)
            .build()
        var lastCode = 0
        var lastDetail = ""
        try {
            for (path in routeCandidates("/upload")) {
                val request = auth(Request.Builder().url(base(settings) + path).post(requestBody)).build()
                executeWithRetry(request).use { response ->
                    val raw = response.body?.string().orEmpty()
                    if (response.isSuccessful) {
                        val array = JSONArray(raw)
                        return array.getString(0)
                    }
                    lastCode = response.code; lastDetail = raw.take(300)
                }
            }
        } catch (e: Throwable) {
            throw friendlyNetworkError(e, base(settings).removePrefix("https://").removePrefix("http://"))
        }
        throw IllegalStateException("Upload failed: HTTP $lastCode. ${lastDetail.ifBlank { "Check Space URL, HF token and Gradio upload route." }}")
    }

    fun upload(settings: HfSettings, file: File, mime: String, onProgress: ((Long, Long) -> Unit)? = null): String {
        val fileBody = file.asRequestBody(mime.toMediaType())
        val trackedBody: RequestBody = if (onProgress != null) ProgressRequestBody(fileBody, onProgress) else fileBody
        val requestBody = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart("files", file.name, trackedBody)
            .build()
        var lastCode = 0
        var lastDetail = ""
        try {
            for (path in routeCandidates("/upload")) {
                val request = auth(Request.Builder().url(base(settings) + path).post(requestBody)).build()
                executeWithRetry(request).use { response ->
                    val raw = response.body?.string().orEmpty()
                    if (response.isSuccessful) {
                        val array = JSONArray(raw)
                        return array.getString(0)
                    }
                    lastCode = response.code; lastDetail = raw.take(300)
                }
            }
        } catch (e: Throwable) {
            throw friendlyNetworkError(e, base(settings).removePrefix("https://").removePrefix("http://"))
        }
        throw IllegalStateException("Upload failed: HTTP $lastCode. ${lastDetail.ifBlank { "Check Space URL, HF token and Gradio upload route." }}")
    }

    fun submit(settings: HfSettings, endpoint: String, data: JSONArray): String {
        val payload = JSONObject().put("data", data).toString()
        var lastCode = 0; var lastBody = ""
        try {
            for (prefix in listOf("/call/", "/gradio_api/call/")) {
                val request = auth(Request.Builder()
                    .url(base(settings) + prefix + endpoint.trim().trim('/'))
                    .post(payload.toRequestBody("application/json".toMediaType()))).build()
                executeWithRetry(request).use { response ->
                    val body = response.body?.string().orEmpty()
                    if (response.isSuccessful) return JSONObject(body).getString("event_id")
                    lastCode = response.code; lastBody = body.take(400)
                }
            }
        } catch (e: Throwable) {
            throw friendlyNetworkError(e, base(settings).removePrefix("https://").removePrefix("http://"))
        }
        throw IllegalStateException("Queue request failed: HTTP $lastCode: $lastBody")
    }

    fun poll(settings: HfSettings, endpoint: String, eventId: String): String {
        var lastCode = 0; var lastBody = ""
        try {
            for (prefix in listOf("/call/", "/gradio_api/call/")) {
                val request = auth(Request.Builder()
                    .url(base(settings) + prefix + endpoint.trim().trim('/') + "/$eventId").get()).build()
                executeWithRetry(request).use { response ->
                    val body = response.body?.string().orEmpty()
                    if (response.isSuccessful) return body
                    lastCode = response.code; lastBody = body.take(400)
                }
            }
        } catch (e: Throwable) {
            throw friendlyNetworkError(e, base(settings).removePrefix("https://").removePrefix("http://"))
        }
        throw IllegalStateException("Polling failed: HTTP $lastCode: $lastBody")
    }

    /**
     * "Session not found" (404) means Gradio has discarded this specific
     * event_id server-side - retrying the same URL can never succeed once
     * that's happened, only a fresh submit() (a new event_id) can. Confirmed
     * by a Space log showing a job that had genuinely started processing
     * (frames already running server-side) while the client stayed stuck
     * showing stale status - the render was fine, only the client's ability
     * to keep hearing about its progress had broken.
     *
     * First attempt at this only caught a clean "HTTP 404: Session not
     * found" message - too narrow. The server log shows the SSE stream
     * terminating ABRUPTLY mid-stream (an exception inside anyio's task
     * group), not a clean 404 response with a readable body, so the client
     * most likely sees a connection-level failure with entirely different
     * message text, wrapped by friendlyNetworkError - which the original
     * text-matching never recognized, so it fell straight through and
     * reproduced the same bug. For phoenix_job_status specifically (a pure
     * read), there's no reason to be choosy about the failure shape: ANY
     * poll failure is safe to recover from by asking fresh, so this now
     * does that regardless of exactly how the failure presented, bounded to
     * a handful of fresh attempts so a truly dead Space still fails
     * eventually rather than retrying forever.
     *
     * allowResubmitOnDeadSession must stay opt-in: for phoenix_job_status
     * re-invoking submit() is harmless - it just asks the same question
     * again. For phoenix_submit_video, blindly resubmitting would start an
     * entire second, duplicate render job, which is a worse outcome than
     * the stuck-status bug this fixes. Every other caller keeps the
     * original behavior unless it explicitly opts in.
     */
    fun callAndWait(
        settings: HfSettings,
        endpoint: String,
        data: JSONArray,
        timeoutMs: Long = 30_000L,
        allowResubmitOnDeadSession: Boolean = false
    ): JSONArray {
        var eventId = submit(settings, endpoint, data)
        val deadline = System.currentTimeMillis() + timeoutMs
        var freshAttemptsLeft = if (allowResubmitOnDeadSession) 8 else 0
        while (System.currentTimeMillis() < deadline) {
            val raw = try {
                poll(settings, endpoint, eventId)
            } catch (e: kotlinx.coroutines.CancellationException) {
                throw e
            } catch (e: Throwable) {
                if (freshAttemptsLeft > 0) {
                    freshAttemptsLeft--
                    eventId = submit(settings, endpoint, data)
                    Thread.sleep(500L)
                    continue
                }
                throw e
            }
            val lower = raw.lowercase()
            if (lower.contains("event: complete")) return extractFinalArray(raw)
            if (lower.contains("event: error") || lower.contains("event: failure")) {
                val detail = extractError(raw)
                throw IllegalStateException(
                    detail ?: (
                        "'$endpoint' exists on the Space but raised an exception server-side, " +
                            "and the Space returned no detail. This is usually an argument mismatch " +
                            "(the client sent ${data.length()} value(s)) or a crash inside the function. " +
                            "Open the Space's Logs tab for the Python traceback, and relaunch it with " +
                            "show_error=True to get the message here."
                        )
                )
            }
            Thread.sleep(700L)
        }
        throw IllegalStateException("Phoenix API call timed out: $endpoint")
    }

    /** Structured result for phoenix_detect_video_frame, matching the documented,
     *  stable 6-value contract in phoenix_api_adapter.api_detect_video_frame:
     *  [0] annotated frame, [1] status message, [2..5] up to four face crops. */
    data class FrameDetection(
        val annotatedPath: String?,
        val message: String?,
        val facePaths: List<String?>
    )

    fun detectVideoFrameFaces(settings: HfSettings, videoServerPath: String, positionPct: Int): FrameDetection {
        val data = JSONArray().put(fileData(videoServerPath)).put(positionPct.coerceIn(0, 100))
        val result = callAndWait(settings, "phoenix_detect_video_frame", data, 240_000L)

        /*
         * Parse positionally against the KNOWN server contract instead of
         * flattening every element that looks path-like.
         *
         * The previous version treated any string containing "/" as a file
         * path, which meant the status message itself - literally formatted
         * as "Frame {idx}/{total} ..." - was picked up as if it were a face
         * crop and the app tried to download the sentence as a file. That
         * always failed and was reported as "Frame detection failed".
         */
        val annotated = if (result.length() > 0) extractPath(result.opt(0)) else null
        val message = if (result.length() > 1) {
            result.optString(1).takeIf { it.isNotBlank() && it != "null" }
        } else null
        val faces = (2..5).map { i -> if (i < result.length()) extractPath(result.opt(i)) else null }

        return FrameDetection(annotated, message, faces)
    }

    fun detectImage(settings: HfSettings, imageFile: File): List<String?> {
        val imagePath = upload(settings, imageFile, "image/jpeg")
        val result = callAndWait(settings, "detect_image", JSONArray().put(fileData(imagePath)), 180_000L)
        return (0 until 7).map { i -> if (i < result.length()) extractPath(result.opt(i)) else null }
    }

    fun swapImage(settings: HfSettings, imageFile: File, faceFiles: List<File?>, quality: String): String? {
        val targetPath = upload(settings, imageFile, "image/jpeg")
        val uploaded = faceFiles.map { f -> f?.let { upload(settings, it, "image/jpeg") } }
        val data = JSONArray().put(fileData(targetPath))
        uploaded.take(4).forEach { data.put(it?.let(::fileData) ?: JSONObject.NULL) }
        while (data.length() < 5) data.put(JSONObject.NULL)
        data.put(quality).put(JSONObject.NULL)
        val result = callAndWait(settings, "swap_image", data, 180_000L)
        return extractPath(result)
    }

    private fun fileData(path: String) = JSONObject().put("path", path).put("meta", JSONObject().put("_type", "gradio.FileData"))

    private fun extractPath(value: Any?): String? = when (value) {
        is JSONObject -> value.optString("path").takeIf { it.isNotBlank() && it != "null" }
        is String -> value.takeIf {
            // A real Gradio file path/URL always has a scheme or a short file
            // extension. Plain status text ("Frame 12/30 · found 2 face(s)")
            // contains a slash too, so a bare contains("/") check is not
            // enough to tell the two apart - that ambiguity is exactly what
            // caused status messages to be downloaded as if they were files.
            it.startsWith("http://") || it.startsWith("https://") ||
                (it.contains("/") && Regex("""\.[A-Za-z0-9]{2,5}$""").containsMatchIn(it))
        }
        is JSONArray -> (0 until value.length()).asSequence().map { extractPath(value.opt(it)) }.firstOrNull { it != null }
        else -> null
    }

    /**
     * Reads the declared input signature of each named route from the Space's
     * /info schema, e.g. "phoenix_detect_video_frame" -> "2 inputs: filepath, number".
     * Comparing this with what the client sends is how an argument-count or
     * argument-type mismatch is caught, since Gradio itself reports it as a
     * bare server-side error with no detail.
     */
    fun endpointSignatures(settings: HfSettings): Map<String, String> {
        val baseUrl = base(settings)
        if (baseUrl.isBlank()) return emptyMap()
        cached(signatureCache, baseUrl)?.let { return it }
        val signatures = linkedMapOf<String, String>()

        for (path in listOf("/gradio_api/info", "/info")) {
            val body = runCatching {
                val request = auth(Request.Builder().url(baseUrl + path).get()).build()
                executeWithRetry(request).use { response ->
                    if (response.isSuccessful) response.body?.string().orEmpty() else null
                }
            }.getOrNull() ?: continue

            val endpoints = runCatching {
                JSONObject(body).optJSONObject("named_endpoints")
            }.getOrNull() ?: continue

            val keys = endpoints.keys()
            while (keys.hasNext()) {
                val key = keys.next()
                val spec = endpoints.optJSONObject(key) ?: continue
                val params = spec.optJSONArray("parameters")
                val types = mutableListOf<String>()
                if (params != null) {
                    for (i in 0 until params.length()) {
                        val p = params.optJSONObject(i) ?: continue
                        val component = p.optString("component").takeIf { it.isNotBlank() && it != "null" }
                        val pythonType = p.optJSONObject("python_type")?.optString("type")
                            ?.takeIf { it.isNotBlank() && it != "null" }
                        types.add(component ?: pythonType ?: "?")
                    }
                }
                val returns = spec.optJSONArray("returns")?.length() ?: 0
                signatures[key.trim().trimStart('/')] =
                    "${types.size} in (${types.joinToString(", ").ifBlank { "none" }}) \u2192 $returns out"
            }
            if (signatures.isNotEmpty()) break
        }
        return putCached(signatureCache, baseUrl, signatures)
    }

    /**
     * A plain `copyTo()` here gave zero visibility into a large transfer -
     * the UI showed a fixed 96% for the entire download regardless of
     * whether it took 5 seconds or 15 minutes, which is indistinguishable
     * from "stuck" to anyone watching it. This streams in a larger buffer
     * (better throughput on a big sequential file than Kotlin's default 8KB)
     * and reports real bytes-transferred so the caller can show it moving.
     */
    fun downloadOutput(
        settings: HfSettings,
        serverPath: String,
        destination: File,
        onProgress: ((downloadedBytes: Long, totalBytes: Long) -> Unit)? = null
    ) {
        // Gradio's /file= route expects the exact absolute path it returned
        // (e.g. "/tmp/gradio/abc123/frame.jpg"). Stripping the leading slash
        // made the server resolve a RELATIVE path against its own working
        // directory instead of the real absolute one - a path that never
        // exists, hence a 404 on every single download.
        val urls = if (serverPath.startsWith("http://") || serverPath.startsWith("https://")) {
            listOf(serverPath)
        } else {
            val absolute = if (serverPath.startsWith("/")) serverPath else "/$serverPath"
            listOf("${base(settings)}/file=$absolute", "${base(settings)}/gradio_api/file=$absolute")
        }
        var lastCode = 0
        try {
            for (url in urls) {
                val request = auth(Request.Builder().url(url).get()).build()
                executeWithRetry(request).use { response ->
                    if (response.isSuccessful) {
                        val body = response.body ?: throw IllegalStateException("Empty output file")
                        val total = body.contentLength()
                        destination.outputStream().use { out ->
                            body.byteStream().use { input ->
                                val buffer = ByteArray(256 * 1024)
                                var downloaded = 0L
                                while (true) {
                                    val n = input.read(buffer)
                                    if (n <= 0) break
                                    out.write(buffer, 0, n)
                                    downloaded += n
                                    onProgress?.invoke(downloaded, total)
                                }
                            }
                        }
                        return
                    }
                    lastCode = response.code
                }
            }
        } catch (e: Throwable) {
            throw friendlyNetworkError(e, base(settings).removePrefix("https://").removePrefix("http://"))
        }
        throw IllegalStateException("Output download failed: HTTP $lastCode")
    }

    private fun extractFinalArray(raw: String): JSONArray {
        for (line in raw.lineSequence().toList().asReversed()) {
            val t = line.trim(); if (!t.startsWith("data:")) continue
            val payload = t.removePrefix("data:").trim(); if (payload.isBlank()) continue
            runCatching { JSONArray(payload) }.getOrNull()?.let { return it }
        }
        throw IllegalStateException("Phoenix API returned no final data payload")
    }

    /**
     * Gradio signals a server-side exception with `event: error` followed by
     * `data: null` when the Space is not launched with show_error=True.
     * The old implementation returned that literal "null" string as the error
     * text, which is how a real server exception surfaced as "failed: null".
     */
    private fun extractError(raw: String): String? {
        val payloads = raw.lineSequence()
            .map { it.trim() }
            .filter { it.startsWith("data:") }
            .map { it.removePrefix("data:").trim() }
            .filter { it.isNotBlank() && it != "null" && it != "\"null\"" }
            .toList()

        for (payload in payloads.asReversed()) {
            runCatching { JSONObject(payload) }.getOrNull()?.let { obj ->
                val message = listOf("message", "error", "detail", "title")
                    .map { obj.optString(it) }
                    .firstOrNull { it.isNotBlank() && it != "null" }
                if (message != null) return message
            }
            if (!payload.startsWith("{") && !payload.startsWith("[")) return payload
        }
        return null
    }
}
