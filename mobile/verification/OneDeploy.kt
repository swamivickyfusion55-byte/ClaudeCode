import com.swamitech.phoenix.net.HfDeploy
import java.io.File
fun main(args: Array<String>) {
    val hub = "http://127.0.0.1:${args[0]}"; val dir = File(args[1])
    val pkg = HfDeploy.readPackage(HfDeploy.SPACE_FILES.map { n -> n to { File(dir, n).inputStream() as java.io.InputStream } })
    val r = HfDeploy({ "tok_write" }, hub).deploy(args[2], pkg, "SwamitechGradio5", true, "wire")
    println("my deploy: ${r.ok} ${r.message}")
}
