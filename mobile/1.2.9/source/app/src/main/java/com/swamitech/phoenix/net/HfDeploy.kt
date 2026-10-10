package com.swamitech.phoenix.net

import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.InputStream
import java.security.MessageDigest
import java.util.Base64
import java.util.concurrent.TimeUnit
import java.util.zip.ZipInputStream

/**
 * Pushes one Phoenix Space code package (app.py, config.py, ...) to many
 * Hugging Face Spaces, so an upgrade no longer means uploading the same eight
 * files into every Space by hand.
 *
 * It speaks the same protocol as Hugging Face's own huggingface_hub client:
 * "preupload" (is each file a plain git file, and is it already identical?)
 * then one "commit" per Space with every changed file in it. A Space whose
 * files are already identical is left alone, so it is not restarted for
 * nothing. A commit restarts and rebuilds that Space - exactly what a manual
 * upload does.
 *
 * Plain JVM on purpose (OkHttp + org.json only, no Android types) so it is
 * tested off-device against the wire format huggingface_hub produces.
 */
class HfDeploy(
    private val tokenProvider: () -> String?,
    private val hub: String = "https://huggingface.co",
    client: OkHttpClient? = null
) {
    class PackageFile(val path: String, val bytes: ByteArray)

    class SpacePackage(
        val files: List<PackageFile>,
        val version: String?,
        val engine: String?,
        val ignored: List<String>
    ) {
        fun summary(): String =
            "${version ?: "unknown version"} · ${files.size} files (${files.joinToString(", ") { it.path }})"
    }

    class PackageException(message: String) : Exception(message)

    class Who(val name: String, val orgs: List<String>, val role: String?)

    data class Result(
        val ok: Boolean,
        val message: String,
        val repoId: String? = null,
        val commit: String? = null,
        val changed: Int = 0
    )

    class HubException(val code: Int, message: String) : Exception(message)

    companion object {
        /** The files that make up a Space. Anything else in a pack (SDOS notes, tools/) is never pushed. */
        val SPACE_FILES = listOf(
            "app.py", "config.py", "core_pipeline.py", "phoenix_api_adapter.py",
            "swap_engine.py", "packages.txt", "requirements.txt", "README.md"
        )

        /** A Space running half of one version and half of another is the failure this refuses. */
        val REQUIRED = listOf("app.py", "config.py", "core_pipeline.py", "phoenix_api_adapter.py", "swap_engine.py")

        const val README = "README.md"

        // Hugging Face refuses plain (non-LFS) git files over 10 MB.
        const val MAX_FILE_BYTES = 8 * 1024 * 1024
        private const val MAX_ZIP_TOTAL_BYTES = 64L * 1024 * 1024

        private val REPO_ID = Regex("^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")

        /**
         * Builds a package from what was picked: one .zip (as shipped), or the
         * loose files. Each item is (display name, opener).
         */
        fun readPackage(items: List<Pair<String, () -> InputStream>>): SpacePackage {
            val found = LinkedHashMap<String, ByteArray>()
            val ignored = mutableListOf<String>()

            fun add(base: String, bytes: ByteArray) {
                if (found.containsKey(base)) {
                    throw PackageException("Two copies of $base were picked. Pick one package.")
                }
                found[base] = bytes
            }

            for ((rawName, open) in items) {
                val name = rawName.substringAfterLast('/').trim()
                if (name.endsWith(".zip", ignoreCase = true)) {
                    var total = 0L
                    open().use { input ->
                        ZipInputStream(input).use { zip ->
                            while (true) {
                                val e = zip.nextEntry ?: break
                                val path = e.name.replace('\\', '/').trimStart('/')
                                val segments = path.split('/').filter { it.isNotEmpty() }
                                val base = segments.lastOrNull() ?: continue
                                val hidden = segments.any { it.startsWith(".") || it == "__MACOSX" }
                                // Only the package's own top level: "app.py" or
                                // "Phoenix_v11.2.80_Resume/app.py". Never tools/... etc.
                                if (e.isDirectory || hidden || segments.size > 2 || base !in SPACE_FILES) {
                                    if (!e.isDirectory && !hidden) ignored += path
                                    continue
                                }
                                val bytes = readCapped(zip, base)
                                total += bytes.size
                                if (total > MAX_ZIP_TOTAL_BYTES) throw PackageException("The zip is larger than a Phoenix package can be.")
                                add(base, bytes)
                            }
                        }
                    }
                } else if (name in SPACE_FILES) {
                    open().use { add(name, readCapped(it, name)) }
                } else {
                    ignored += name
                }
            }

            val missing = REQUIRED.filter { it !in found }
            if (missing.isNotEmpty()) {
                throw PackageException(
                    "Missing ${missing.joinToString(", ")}. Updating only some files would leave a Space " +
                        "running two versions at once - pick the whole package (the .zip, or all files)."
                )
            }
            for ((base, bytes) in found) {
                if (bytes.isEmpty()) throw PackageException("$base is empty.")
            }
            val config = String(found.getValue("config.py"), Charsets.UTF_8)
            val engine = String(found.getValue("swap_engine.py"), Charsets.UTF_8)
            val ordered = SPACE_FILES.filter { it in found }.map { PackageFile(it, found.getValue(it)) }
            return SpacePackage(
                files = ordered,
                version = Regex("""(?m)^VERSION\s*=\s*["']([^"']+)["']""").find(config)?.groupValues?.get(1),
                engine = Regex("""(?m)^ENGINE_VERSION\s*=\s*["']([^"']+)["']""").find(engine)?.groupValues?.get(1),
                ignored = ignored
            )
        }

        private fun readCapped(input: InputStream, name: String): ByteArray {
            val out = ByteArrayOutputStream()
            val buf = ByteArray(64 * 1024)
            var total = 0
            while (true) {
                val n = input.read(buf)
                if (n < 0) break
                total += n
                if (total > MAX_FILE_BYTES) throw PackageException("$name is larger than ${MAX_FILE_BYTES / (1024 * 1024)} MB.")
                out.write(buf, 0, n)
            }
            return out.toByteArray()
        }

        /** git's blob id for a file: how the Hub reports what a Space already has. */
        fun gitBlobSha1(bytes: ByteArray): String {
            val md = MessageDigest.getInstance("SHA-1")
            md.update("blob ${bytes.size}\u0000".toByteArray(Charsets.UTF_8))
            md.update(bytes)
            return md.digest().joinToString("") { "%02x".format(it) }
        }

        /** "https://owner-name.hf.space/x" -> "owner-name" */
        fun subdomainOf(spaceUrl: String): String? {
            val host = spaceUrl.trim().lowercase().removePrefix("https://").removePrefix("http://")
                .substringBefore('/').substringBefore('?').substringBefore(':')
            if (!host.endsWith(".hf.space")) return null
            return host.removeSuffix(".hf.space").ifBlank { null }
        }

        /** Hugging Face's own mapping from a repo id to its *.hf.space name. */
        fun subdomainForRepo(repoId: String): String =
            repoId.replace('/', '-').replace('_', '-').replace('.', '-').lowercase()

        /**
         * The pack's README.md is the Space's settings header (sdk, app_file...)
         * and carries ONE title. Keep each Space's own title instead of renaming
         * every Space to the pack's. Null when the header is missing or does not
         * look like a Gradio Space header - such a README would break the Space.
         */
        fun readmeForSpace(readme: ByteArray, title: String?): ByteArray? {
            val text = String(readme, Charsets.UTF_8)
            val normalized = text.removePrefix("﻿")
            if (!normalized.startsWith("---")) return null
            val end = normalized.indexOf("\n---", 3)
            if (end < 0) return null
            val header = normalized.substring(0, end)
            if (!Regex("(?m)^sdk:\\s*\\S").containsMatchIn(header) ||
                !Regex("(?m)^app_file:\\s*\\S").containsMatchIn(header)
            ) return null
            val t = title?.trim()?.replace(Regex("[\\r\\n\"]"), " ")?.take(30)
            if (t.isNullOrBlank()) return readme
            val newHeader = if (Regex("(?m)^title:.*$").containsMatchIn(header)) {
                header.replaceFirst(Regex("(?m)^title:.*$"), "title: \"$t\"")
            } else {
                header.replaceFirst("---", "---\ntitle: \"$t\"")
            }
            return (newHeader + normalized.substring(end)).toByteArray(Charsets.UTF_8)
        }
    }

    private val http: OkHttpClient = client ?: OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(120, TimeUnit.SECONDS)
        .writeTimeout(120, TimeUnit.SECONDS)
        .callTimeout(5, TimeUnit.MINUTES)
        .build()

    private val jsonType = "application/json".toMediaType()
    private val ndjsonType = "application/x-ndjson".toMediaType()

    private fun request(path: String): Request.Builder {
        val token = tokenProvider()?.trim().orEmpty()
        if (token.isBlank()) throw HubException(401, "No Hugging Face token saved.")
        val url = if (path.startsWith("http")) path else hub.trimEnd('/') + path
        return Request.Builder().url(url).header("Authorization", "Bearer $token")
    }

    private fun <T> call(req: Request, what: String, read: (Response, String) -> T): T {
        http.newCall(req).execute().use { resp ->
            val body = resp.body?.string().orEmpty()
            if (!resp.isSuccessful) throw HubException(resp.code, explain(resp.code, body, what))
            return read(resp, body)
        }
    }

    private fun explain(code: Int, body: String, what: String): String {
        val server = runCatching { JSONObject(body).optString("error") }.getOrNull()
            ?.takeIf { it.isNotBlank() } ?: body.take(160)
        val why = when (code) {
            401 -> "the token was rejected (wrong, expired or revoked)"
            403 -> "this token does not have permission (updating a Space needs a token with write access)"
            404 -> "not found, or this token cannot see it"
            409, 412 -> "it changed while updating; try again"
            413 -> "a file is too large"
            429 -> "Hugging Face is rate limiting; wait a minute and retry"
            in 500..599 -> "Hugging Face had a server error; retry shortly"
            else -> "HTTP $code"
        }
        return "$what: $why" + if (server.isNotBlank()) " ($server)" else ""
    }

    /** Who the token belongs to, its organisations, and its role (read / write / fineGrained). */
    fun whoami(): Who = call(request("/api/whoami-v2").get().build(), "Token check") { _, body ->
        val o = JSONObject(body)
        val orgs = mutableListOf<String>()
        o.optJSONArray("orgs")?.let { arr ->
            for (i in 0 until arr.length()) arr.optJSONObject(i)?.optString("name")?.takeIf { it.isNotBlank() }?.let(orgs::add)
        }
        val role = o.optJSONObject("auth")?.optJSONObject("accessToken")?.optString("role")?.takeIf { it.isNotBlank() }
        Who(o.optString("name"), orgs, role)
    }

    /**
     * Maps each *.hf.space URL to its repo id ("Owner/Name") by listing the
     * Spaces the token's user and organisations own. A URL alone cannot be
     * turned back into a repo id (case and '_' are lost), so it is looked up.
     */
    fun resolveRepoIds(spaceUrls: List<String>, owners: List<String>): Map<String, String> {
        val bySub = HashMap<String, String>()
        for (owner in owners.filter { Regex("^[A-Za-z0-9][A-Za-z0-9._-]*$").matches(it) }.distinct()) {
            var next: String? = "/api/spaces?author=$owner&limit=1000&expand=subdomain"
            var pages = 0
            while (next != null && pages < 20) {
                pages++
                next = call(request(next).get().build(), "Listing Spaces of $owner") { resp, body ->
                    val arr = JSONArray(body)
                    for (i in 0 until arr.length()) {
                        val s = arr.optJSONObject(i) ?: continue
                        val id = s.optString("id").takeIf { REPO_ID.matches(it) } ?: continue
                        val sub = s.optString("subdomain").ifBlank { subdomainForRepo(id) }.lowercase()
                        bySub.putIfAbsent(sub, id)
                        bySub.putIfAbsent(subdomainForRepo(id), id)
                    }
                    // Follow the next page only on the Hub itself: the token is sent with it.
                    Regex("""<([^>]+)>\s*;\s*rel="next"""").find(resp.header("Link").orEmpty())?.groupValues?.get(1)
                        ?.takeIf { it.startsWith(hub.trimEnd('/') + "/") }
                }
            }
        }
        val out = LinkedHashMap<String, String>()
        for (u in spaceUrls) {
            val sub = subdomainOf(u) ?: continue
            bySub[sub]?.let { out[u] = it }
        }
        return out
    }

    /** Current stage of a Space (RUNNING, BUILDING, RUNTIME_ERROR, PAUSED, SLEEPING...). */
    fun runtime(repoId: String): String {
        require(REPO_ID.matches(repoId)) { "bad repo id" }
        return call(request("/api/spaces/$repoId/runtime").get().build(), repoId) { _, body ->
            val o = JSONObject(body)
            val stage = o.optString("stage").ifBlank { "UNKNOWN" }
            val err = o.optString("errorMessage").takeIf { it.isNotBlank() && it != "null" }
            if (err != null) "$stage - ${err.lineSequence().firstOrNull().orEmpty().take(120)}" else stage
        }
    }

    /**
     * Commits the package to one Space. Files the Space already has
     * byte-for-byte are left out; if nothing changed, nothing is committed and
     * the Space is not restarted.
     */
    fun deploy(repoId: String, pkg: SpacePackage, title: String?, includeReadme: Boolean, summary: String): Result {
        if (!REPO_ID.matches(repoId)) return Result(false, "Not a Hugging Face Space id: $repoId", repoId)
        return try {
            deployOrThrow(repoId, pkg, title, includeReadme, summary)
        } catch (e: HubException) {
            Result(false, e.message ?: "HTTP ${e.code}", repoId)
        } catch (e: java.io.IOException) {
            Result(false, "Network problem (${e.javaClass.simpleName}): ${e.message.orEmpty().take(120)} - nothing was changed unless it says updated", repoId)
        }
    }

    private fun deployOrThrow(repoId: String, pkg: SpacePackage, title: String?, includeReadme: Boolean, summary: String): Result {
        val files = pkg.files.mapNotNull { f ->
            if (f.path != README) f
            else if (!includeReadme) null
            else readmeForSpace(f.bytes, title)?.let { PackageFile(README, it) }
        }
        val readmeDropped = includeReadme && pkg.files.any { it.path == README } && files.none { it.path == README }

        // 1. preupload: plain git file or LFS, and what the Space already has.
        val pre = JSONObject().put("files", JSONArray().also { arr ->
            files.forEach { f ->
                arr.put(
                    JSONObject()
                        .put("path", f.path)
                        .put("sample", Base64.getEncoder().encodeToString(f.bytes.copyOfRange(0, minOf(512, f.bytes.size))))
                        .put("size", f.bytes.size)
                )
            }
        })
        val modes = call(
            request("/api/spaces/$repoId/preupload/main").post(pre.toString().toRequestBody(jsonType)).build(),
            repoId
        ) { _, body ->
            val arr = JSONObject(body).optJSONArray("files") ?: throw HubException(502, "$repoId: unexpected preupload reply")
            (0 until arr.length()).associate { i ->
                val o = arr.getJSONObject(i)
                o.getString("path") to Triple(
                    o.optString("uploadMode"),
                    o.optBoolean("shouldIgnore", false),
                    o.optString("oid").takeIf { it.isNotBlank() && it != "null" }
                )
            }
        }
        val lfs = files.filter { modes[it.path]?.first == "lfs" }.map { it.path }
        if (lfs.isNotEmpty()) {
            return Result(false, "${lfs.joinToString()} would have to go through Git LFS on this Space; upload those by hand.", repoId)
        }
        val changed = files.filter { f ->
            val m = modes[f.path]
            m == null || (!m.second && m.third != gitBlobSha1(f.bytes))
        }
        val note = if (readmeDropped) " (README.md skipped: it has no valid Space header)" else ""
        if (changed.isEmpty()) {
            return Result(true, "Already up to date - not restarted$note", repoId, changed = 0)
        }

        // 2. commit: one NDJSON body, header line then one line per file.
        val ndjson = StringBuilder()
        ndjson.append(JSONObject().put("key", "header").put("value", JSONObject().put("summary", summary).put("description", "")).toString()).append('\n')
        for (f in changed) {
            ndjson.append(
                JSONObject().put("key", "file").put(
                    "value",
                    JSONObject()
                        .put("content", Base64.getEncoder().encodeToString(f.bytes))
                        .put("path", f.path)
                        .put("encoding", "base64")
                ).toString()
            ).append('\n')
        }
        return call(
            request("/api/spaces/$repoId/commit/main").post(ndjson.toString().toRequestBody(ndjsonType)).build(),
            repoId
        ) { _, body ->
            val oid = runCatching { JSONObject(body).optString("commitOid") }.getOrNull().orEmpty()
            Result(
                true,
                "Updated ${changed.size} file(s)${if (oid.length >= 7) " · commit ${oid.take(7)}" else ""} · rebuilding$note",
                repoId, oid.ifBlank { null }, changed.size
            )
        }
    }
}
