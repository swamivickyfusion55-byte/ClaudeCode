import com.swamitech.phoenix.net.HfApi
import java.net.HttpURLConnection
import java.net.URL

fun post(u: String) { (URL(u).openConnection() as HttpURLConnection).run { requestMethod = "POST"; responseCode; disconnect() } }
var fails = 0
fun check(name: String, got: Any?, want: Any?) { val ok = got == want; if (!ok) fails++; println("${if (ok) "PASS" else "FAIL"} $name: got $got, want $want") }

// usage: BusyTest <v92 url> <v91 url>
fun main(args: Array<String>) {
    val api = HfApi { "" }
    val v92 = args[0]; val v91 = args[1]
    check("v92 idle Space reports not busy", api.spaceBusy(v92), false)
    post("$v92/test/job/AAAA1111/processing")
    check("v92 Space with a running job reports busy", api.spaceBusy(v92), true)
    check("v92 job status: running", api.jobRunningOnSpace(v92, "AAAA1111"), true)
    post("$v92/test/job/AAAA1111/done")
    check("v92 Space after the job ended reports not busy", api.spaceBusy(v92), false)
    check("v92 finished job is not running", api.jobRunningOnSpace(v92, "AAAA1111"), false)
    post("$v92/test/job/BBBB2222/error")
    check("v92 failed job does not count as busy", api.spaceBusy(v92), false)
    check("v91 Space has no busy route -> unknown (null)", api.spaceBusy(v91), null)
    post("$v91/test/job/CCCC3333/processing")
    check("v91 running job seen through /phoenix/status", api.jobRunningOnSpace(v91, "CCCC3333"), true)
    post("$v91/test/job/CCCC3333/done")
    check("v91 finished job", api.jobRunningOnSpace(v91, "CCCC3333"), false)
    check("v91 unknown job (Space restarted) is not running", api.jobRunningOnSpace(v91, "DDDD4444"), false)
    check("unreachable Space -> unknown (null)", api.spaceBusy("http://127.0.0.1:9"), null)
    check("unreachable Space job -> unknown (null)", api.jobRunningOnSpace("http://127.0.0.1:9", "AAAA1111"), null)
    println(if (fails == 0) "ALL PASS" else "$fails FAILED")
}
