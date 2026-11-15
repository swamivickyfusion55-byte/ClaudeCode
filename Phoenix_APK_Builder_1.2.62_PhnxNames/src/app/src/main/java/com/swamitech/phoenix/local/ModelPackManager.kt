package com.swamitech.phoenix.local

import android.content.Context
import com.swamitech.phoenix.HfSettings
import com.swamitech.phoenix.security.SecureStore
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.io.FileInputStream
import java.security.MessageDigest
import java.util.concurrent.TimeUnit
import org.json.JSONObject

class ModelPackManager(private val context: Context) {
    private val root = File(context.filesDir, "phoenix/models")
    private val secure = SecureStore(context)
    private val http = OkHttpClient.Builder().readTimeout(20, TimeUnit.MINUTES).build()

    fun root(): File = root.apply { mkdirs() }

    fun requiredFiles(): List<String> = listOf(
        "detector.onnx", "recognizer.onnx", "landmarks.onnx", "swapper.onnx"
    )

    fun isComplete(): Boolean = requiredFiles().all { File(root, it).isFile && File(root, it).length() > 0 }

    fun installFromHf(settings: HfSettings, progress: (String, Long, Long) -> Unit) {
        require(settings.modelRepo.isNotBlank()) { "Set a private/public Hugging Face model repository first" }
        val manifestUrl = "https://huggingface.co/${settings.modelRepo}/resolve/${settings.modelRevision}/phoenix-model-manifest.json"
        val manifest = downloadBytes(manifestUrl)
        val json = JSONObject(String(manifest))
        val files = json.getJSONArray("files")
        root.mkdirs()
        for (i in 0 until files.length()) {
            val item = files.getJSONObject(i)
            val name = item.getString("name")
            val sha = item.getString("sha256")
            val target = File(root, name)
            target.parentFile?.mkdirs()
            val url = "https://huggingface.co/${settings.modelRepo}/resolve/${settings.modelRevision}/${name}"
            downloadTo(url, target) { done, total -> progress(name, done, total) }
            require(sha256(target).equals(sha, ignoreCase = true)) { "SHA-256 mismatch for $name" }
        }
    }

    private fun request(url: String): Request.Builder {
        val b = Request.Builder().url(url)
        secure.getToken()?.let { b.header("Authorization", "Bearer $it") }
        return b
    }

    private fun downloadBytes(url: String): ByteArray = http.newCall(request(url).get().build()).execute().use { r ->
        if (!r.isSuccessful) error("Model manifest download failed: HTTP ${r.code}")
        r.body?.bytes() ?: error("Empty model manifest")
    }

    private fun downloadTo(url: String, target: File, progress: (Long, Long) -> Unit) {
        http.newCall(request(url).get().build()).execute().use { r ->
            if (!r.isSuccessful) error("Model download failed for ${target.name}: HTTP ${r.code}")
            val total = r.body?.contentLength() ?: -1L
            r.body?.byteStream()?.use { input ->
                target.outputStream().use { out ->
                    val buf = ByteArray(1024 * 1024)
                    var done = 0L
                    while (true) {
                        val n = input.read(buf)
                        if (n <= 0) break
                        out.write(buf, 0, n)
                        done += n
                        progress(done, total)
                    }
                }
            }
        }
    }

    private fun sha256(file: File): String {
        val md = MessageDigest.getInstance("SHA-256")
        FileInputStream(file).use { input ->
            val buf = ByteArray(1024 * 1024)
            while (true) { val n = input.read(buf); if (n <= 0) break; md.update(buf, 0, n) }
        }
        return md.digest().joinToString("") { "%02x".format(it) }
    }
}
