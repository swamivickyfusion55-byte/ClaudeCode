import com.swamitech.phoenix.net.HfApi
import okhttp3.Interceptor
import okhttp3.OkHttpClient
import okhttp3.HttpUrl.Companion.toHttpUrl
import java.io.File

// Runs THEIR commitSpaceFiles() unchanged. The URL is hard-coded to https://huggingface.co, so an
// application interceptor (injected by reflection) redirects it to the local mock; everything
// OkHttp puts on the wire after that is real.
fun main(args: Array<String>) {
    val port = args[0]; val pack = File(args[1]); val repo = args[2]
    val api = HfApi { "tok_write" }
    val f = HfApi::class.java.getDeclaredField("http"); f.isAccessible = true
    val base = f.get(api) as OkHttpClient
    val redirect = Interceptor { chain ->
        val r = chain.request()
        chain.proceed(r.newBuilder().url(r.url.toString().replace("https://huggingface.co", "http://127.0.0.1:$port").toHttpUrl()).build())
    }
    f.set(api, base.newBuilder().addInterceptor(redirect).build())
    val names = listOf("app.py", "config.py", "core_pipeline.py", "phoenix_api_adapter.py", "swap_engine.py", "packages.txt", "requirements.txt")
    api.commitSpaceFiles(repo, names.map { it to File(pack, it).readBytes() })
    println("their commitSpaceFiles: OK")
}
