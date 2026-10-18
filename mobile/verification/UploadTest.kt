import com.swamitech.phoenix.HfSettings
import com.swamitech.phoenix.net.HfApi
import java.io.File
import java.io.RandomAccessFile
import java.security.MessageDigest

fun sha(f: File): String = MessageDigest.getInstance("SHA-256").digest(f.readBytes()).joinToString("") { "%02x".format(it) }

fun main(args: Array<String>) {
    val url = args[0]; val file = File(args[1]); val token = args[2]
    val api = HfApi { "" }
    val raf = RandomAccessFile(file, "r")
    var lastPct = -1
    val t0 = System.currentTimeMillis()
    val path = api.uploadResumable(
        HfSettings(spaceUrl = url), "phoenix_video_test.mp4", file.length(), token,
        readChunk = { off, n -> val stop = System.getenv("STOP_AT")?.toLong() ?: Long.MAX_VALUE; if (off >= stop) { println("simulated app death at byte $off"); kotlin.system.exitProcess(3) }; val b = ByteArray(n); raf.seek(off); raf.readFully(b); b },
        onProgress = { w, t -> val p = (w * 100 / t).toInt(); if (p / 10 != lastPct / 10) { lastPct = p; println("progress $p%") } },
        maxStallMs = 120_000L
    )
    println("RESULT path=$path seconds=${(System.currentTimeMillis() - t0) / 1000}")
    if (path != null) {
        val got = File(path)
        println("server file exists=${got.isFile} size=${if (got.isFile) got.length() else -1} sameBytes=${got.isFile && sha(got) == sha(file)}")
    }
}
