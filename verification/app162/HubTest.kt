import com.swamitech.phoenix.net.HfApi

// v1.2.62 check: the real HfApi.spaceRuntime / pauseSpace / restartSpace against a local stand-in for the Hugging Face Hub
// (fake_hub.py), which records every request and answers as the Hub does (huggingface_hub 0.27.1 hf_api.py).
fun main(args: Array<String>) {
    var fails = 0
    fun check(name: String, ok: Boolean, extra: String = "") { println((if (ok) "PASS " else "FAIL ") + name + (if (extra.isNotEmpty()) "  [$extra]" else "")); if (!ok) fails++ }
    val api = HfApi { "hf_TESTTOKEN" }; api.hubBase = "http://127.0.0.1:${args[0]}"
    val r0 = api.spaceRuntime("owner/SwamitechGradio3")
    check("status: GET /api/spaces/owner/SwamitechGradio3/runtime, with the token; stage read", r0.stage == "RUNNING" && r0.hardware == "cpu-basic", r0.toString())
    val r1 = api.pauseSpace("owner/SwamitechGradio3")
    check("pause: POST .../pause; the Hub's answer read (PAUSED)", r1.stage == "PAUSED", r1.toString())
    check("then status reads PAUSED", api.spaceRuntime("owner/SwamitechGradio3").stage == "PAUSED")
    val r2 = api.restartSpace("owner/SwamitechGradio3")
    check("resume: POST .../restart on the paused Space; it builds again", r2.stage == "BUILDING" || r2.stage == "APP_STARTING", r2.toString())
    val e403 = runCatching { api.pauseSpace("owner/ReadOnly") }.exceptionOrNull()?.message.orEmpty()
    check("a token without write access: says so plainly (HTTP 403)", e403.contains("write access") && e403.contains("403"), e403.take(120))
    val e404 = runCatching { api.restartSpace("owner/NoSuchSpace") }.exceptionOrNull()?.message.orEmpty()
    check("a Space name the Hub does not know: says to check the name (HTTP 404)", e404.contains("no Space owner/NoSuchSpace"), e404.take(120))
    val noTok = HfApi { null }; noTok.hubBase = api.hubBase
    val eTok = runCatching { noTok.pauseSpace("owner/SwamitechGradio3") }.exceptionOrNull()?.message.orEmpty()
    check("no token saved: refused before any request", eTok.contains("Save a Hugging Face token"), eTok)
    val down = HfApi { "x" }; down.hubBase = "http://127.0.0.1:1"
    val eNet = runCatching { down.spaceRuntime("owner/X") }.exceptionOrNull()
    check("Hub unreachable: an error, no crash", eNet != null, eNet?.message?.take(100) ?: "")
    println(if (fails == 0) "all checks passed" else "$fails check(s) failed")
}
