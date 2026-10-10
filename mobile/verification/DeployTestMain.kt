import com.swamitech.phoenix.net.HfDeploy
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.MediaType.Companion.toMediaType
import org.json.JSONObject
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.File
import java.util.Base64
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

var pass = 0; var fail = 0
fun check(name: String, ok: Boolean, extra: String = "") {
    if (ok) pass++ else fail++
    println((if (ok) "PASS " else "FAIL ") + name + if (!ok && extra.isNotEmpty()) "  [$extra]" else "")
}
inline fun <reified T : Throwable> throws(block: () -> Unit): T? = try { block(); null } catch (t: Throwable) { t as? T }

fun main(args: Array<String>) {
    val port = args[0]; val zipPath = args[1]; val packDir = File(args[2])
    val hub = "http://127.0.0.1:$port"
    val http = OkHttpClient()
    fun ctl(path: String, body: String? = null): JSONObject {
        val b = Request.Builder().url(hub + path).header("Authorization", "Bearer tok_write")
        val r = if (body == null) b.get() else b.post(body.toRequestBody("application/json".toMediaType()))
        http.newCall(r.build()).execute().use { return JSONObject(it.body!!.string()) }
    }
    ctl("/api/control/reset")

    // ---------------- package reading ----------------
    val zipBytes = File(zipPath).readBytes()
    val pkg = HfDeploy.readPackage(listOf("Phoenix_v11.2.80_Resume.zip" to { ByteArrayInputStream(zipBytes) }))
    check("K1 the shipped v11.2.80 zip -> the 8 Space files", pkg.files.map { it.path } == HfDeploy.SPACE_FILES, pkg.files.map { it.path }.toString())
    check("K1 version and engine read from the files", pkg.version == "v11.2.80" && pkg.engine == "aequus-1.2.79-reid", "${pkg.version} ${pkg.engine}")
    check("K1 SDOS notes and tools/ are never pushed", pkg.ignored.any { it.endsWith("tools/phoenix_resumable_client.py") } && pkg.ignored.any { it.endsWith("SDOS-079_Resume.md") } && pkg.files.none { it.path.startsWith("SDOS") })
    check("K1 bytes identical to the files in the pack", pkg.files.all { it.bytes.contentEquals(File(packDir, it.path).readBytes()) })
    val loose = HfDeploy.SPACE_FILES.map { n -> n to { File(packDir, n).inputStream() as java.io.InputStream } } + listOf("SDOS-079_Resume.md" to { File(packDir, "SDOS-079_Resume.md").inputStream() as java.io.InputStream })
    val pkg2 = HfDeploy.readPackage(loose)
    check("K2 the same 8 files picked loose give the same package", pkg2.files.map { it.path } == pkg.files.map { it.path } && pkg2.files.zip(pkg.files).all { (a, b) -> a.bytes.contentEquals(b.bytes) } && "SDOS-079_Resume.md" in pkg2.ignored)
    val e3 = throws<HfDeploy.PackageException> { HfDeploy.readPackage(loose.filter { it.first != "swap_engine.py" && it.first != "config.py" }) }
    check("K3 a partial set is refused, naming what is missing", e3 != null && e3.message!!.contains("config.py") && e3.message!!.contains("swap_engine.py"), e3?.message ?: "no exception")
    val e4 = throws<HfDeploy.PackageException> { HfDeploy.readPackage(listOf("p.zip" to { ByteArrayInputStream(zipBytes) as java.io.InputStream }, "app.py" to { File(packDir, "app.py").inputStream() as java.io.InputStream })) }
    check("K4 zip plus a loose app.py -> refused as two copies", e4 != null && e4.message!!.contains("Two copies of app.py"), e4?.message ?: "")
    val big = ByteArray(HfDeploy.MAX_FILE_BYTES + 1)
    val e5 = throws<HfDeploy.PackageException> { HfDeploy.readPackage(loose.map { if (it.first == "core_pipeline.py") "core_pipeline.py" to { ByteArrayInputStream(big) as java.io.InputStream } else it }) }
    check("K5 a file over the size cap is refused", e5 != null && e5.message!!.contains("core_pipeline.py"), e5?.message ?: "")
    // a zip with two package folders inside it
    val twice = ByteArrayOutputStream().also { bo -> ZipOutputStream(bo).use { z -> for (d in listOf("a", "b")) for (f in HfDeploy.REQUIRED) { z.putNextEntry(ZipEntry("$d/$f")); z.write(File(packDir, f).readBytes()); z.closeEntry() } } }.toByteArray()
    check("K5 a zip holding two versions is refused", throws<HfDeploy.PackageException> { HfDeploy.readPackage(listOf("x.zip" to { ByteArrayInputStream(twice) as java.io.InputStream })) } != null)
    // __MACOSX junk and nested folders are ignored
    val mac = ByteArrayOutputStream().also { bo -> ZipOutputStream(bo).use { z -> for (f in HfDeploy.REQUIRED) { z.putNextEntry(ZipEntry("pkg/$f")); z.write(File(packDir, f).readBytes()); z.closeEntry(); z.putNextEntry(ZipEntry("__MACOSX/pkg/._$f")); z.write(byteArrayOf(1)); z.closeEntry() }; z.putNextEntry(ZipEntry("pkg/deep/app.py")); z.write(byteArrayOf(9)); z.closeEntry() } }.toByteArray()
    val pm = HfDeploy.readPackage(listOf("m.zip" to { ByteArrayInputStream(mac) as java.io.InputStream }))
    check("K5 macOS junk and nested copies are ignored; 5 required files without the optional 3 is fine", pm.files.size == 5 && pm.files.first { it.path == "app.py" }.bytes.size > 100)
    check("K6 git blob id matches huggingface_hub's documented example", HfDeploy.gitBlobSha1("Hello, World!".toByteArray()) == "b45ef6fec89518d314f546fd6c3025367b721684")
    check("K7 URL -> subdomain", HfDeploy.subdomainOf("https://swamivicky-swamitechgradio11.hf.space/") == "swamivicky-swamitechgradio11" && HfDeploy.subdomainOf(" HTTPS://Swamivicky-SG-UAT2.hf.space/?x") == "swamivicky-sg-uat2" && HfDeploy.subdomainOf("https://example.com") == null)
    check("K7 repo id -> subdomain", HfDeploy.subdomainForRepo("Swamivicky/SG_UAT2") == "swamivicky-sg-uat2")
    val rd = File(packDir, "README.md").readBytes()
    val r1 = HfDeploy.readmeForSpace(rd, "SwamitechGradio11")?.toString(Charsets.UTF_8)
    check("K8 README keeps each Space's own title, everything else unchanged", r1 != null && r1.contains("\ntitle: \"SwamitechGradio11\"\n") && !r1.contains("title: SG UAT2") && r1.replace("title: \"SwamitechGradio11\"", "title: SG UAT2") == String(rd, Charsets.UTF_8))
    check("K8 a README without the Space header is refused (it would break the Space)", HfDeploy.readmeForSpace("# docs\n".toByteArray(), "x") == null && HfDeploy.readmeForSpace("---\ntitle: a\n---\n".toByteArray(), "x") == null)

    // ---------------- talking to the (mock) Hub ----------------
    val urls = (2..20).map { "https://swamivicky-swamitechgradio$it.hf.space" } + listOf("https://swamivicky-sg-uat2.hf.space", "https://swamivicky-swamitechuat2.hf.space")
    val labels = (2..20).map { "SwamitechGradio$it" } + listOf("SG_UAT2", "SwamitechUAT2")
    val d = HfDeploy({ "tok_write" }, hub)
    val who = d.whoami()
    check("K9 whoami: user, orgs and token role", who.name == "Swamivicky" && who.orgs == listOf("SwamiOrg") && who.role == "write")
    val ids = d.resolveRepoIds(urls + "https://nobody-nothing.hf.space", listOf(who.name) + who.orgs)
    check("K10 all 21 app Spaces resolved to their repo ids (paged listing, subdomain fallback)", urls.all { ids[it] != null } && ids["https://swamivicky-sg-uat2.hf.space"] == "Swamivicky/SG_UAT2" && ids["https://swamivicky-swamitechgradio7.hf.space"] == "Swamivicky/SwamitechGradio7", ids.toString())
    check("K10 an unknown Space is simply not resolved", ids["https://nobody-nothing.hf.space"] == null)

    val results = urls.mapIndexed { i, u -> d.deploy(ids.getValue(u), pkg, labels[i], true, "Phoenix ${pkg.version} (Phoenix Mobile)") }
    val okCount = results.count { it.ok && it.changed == 8 }
    val denied = results.filter { !it.ok }
    check("K11 20 Spaces updated with all 8 files in one commit each", okCount == 20, results.map { it.message }.toString())
    check("K11 the Space without write permission reports it clearly", denied.size == 1 && denied[0].repoId == "Swamivicky/SwamitechGradio13" && denied[0].message.contains("write access"), denied.map { it.message }.toString())
    val st = ctl("/api/control/state")
    val sp11 = st.getJSONObject("Swamivicky/SwamitechGradio11")
    fun f(sp: JSONObject, p: String) = Base64.getDecoder().decode(sp.getJSONObject("files").getString(p))
    check("K11 server holds the exact package bytes", HfDeploy.SPACE_FILES.filter { it != "README.md" }.all { f(sp11, it).contentEquals(File(packDir, it).readBytes()) })
    check("K11 README on the server keeps that Space's title", String(f(sp11, "README.md")).contains("title: \"SwamitechGradio11\"") && String(f(sp11, "README.md")).contains("sdk_version: 4.44.1"))
    check("K11 files not in the package are untouched (.gitattributes)", String(f(sp11, ".gitattributes")) == "*.bin filter=lfs\n")
    check("K11 exactly one commit per updated Space", (2..20).filter { it != 13 }.all { st.getJSONObject("Swamivicky/SwamitechGradio$it").getJSONArray("commits").length() == 1 })
    val again = urls.mapIndexed { i, u -> d.deploy(ids.getValue(u), pkg, labels[i], true, "again") }
    val st2 = ctl("/api/control/state")
    check("K12 running it again changes nothing and restarts nothing", again.filter { it.ok }.all { it.changed == 0 && it.message.startsWith("Already up to date") } && (2..20).filter { it != 13 }.all { st2.getJSONObject("Swamivicky/SwamitechGradio$it").getJSONArray("commits").length() == 1 })
    // a new version where only one file changed -> one-file commit
    val bumped = HfDeploy.SpacePackage(pkg.files.map { if (it.path == "config.py") HfDeploy.PackageFile("config.py", it.bytes + "\n# v-next\n".toByteArray()) else it }, "v-next", pkg.engine, emptyList())
    val r3 = d.deploy("Swamivicky/SwamitechGradio5", bumped, "SwamitechGradio5", true, "Phoenix v-next")
    check("K12 only the changed file is sent", r3.ok && r3.changed == 1)
    val r4 = d.deploy("Swamivicky/SwamitechGradio6", bumped, "Anything", false, "no readme")
    val sp6 = ctl("/api/control/state").getJSONObject("Swamivicky/SwamitechGradio6")
    check("K13 'include README' off leaves the Space's README alone", r4.ok && String(f(sp6, "README.md")).contains("title: \"SwamitechGradio6\""))
    // read-only token
    val ro = HfDeploy({ "tok_read" }, hub)
    check("K14 a read-only token is visible up front (role=read)", ro.whoami().role == "read")
    val r5 = ro.deploy("Swamivicky/SwamitechGradio9", bumped, "SwamitechGradio9", true, "x")
    check("K14 ...and a commit with it is refused with a clear message", !r5.ok && r5.message.contains("write access"), r5.message)
    val dead = HfDeploy({ "tok_write" }, "http://127.0.0.1:1").deploy("Swamivicky/SwamitechGradio9", bumped, null, true, "x")
    check("K14 no network -> a result saying so, not a crash", !dead.ok && dead.message.startsWith("Network problem"), dead.message)
    // LFS-mode file
    ctl("/api/control/set", """{"deny_commit":[],"lfs_paths":["core_pipeline.py"]}""")
    val r6 = d.deploy("Swamivicky/SwamitechGradio8", bumped, "SwamitechGradio8", true, "x")
    val c8 = ctl("/api/control/state").getJSONObject("Swamivicky/SwamitechGradio8").getJSONArray("commits").length()
    check("K15 a file the Space would store in Git LFS stops that Space, nothing half-committed", !r6.ok && r6.message.contains("core_pipeline.py") && c8 == 1)
    ctl("/api/control/set", """{"deny_commit":[],"lfs_paths":[],"stage":{"Swamivicky/SwamitechGradio4":"RUNTIME_ERROR"}}""")
    check("K16 runtime stage after a deploy", d.runtime("Swamivicky/SwamitechGradio3") == "BUILDING")
    check("K16 runtime error shows its first line", d.runtime("Swamivicky/SwamitechGradio4").startsWith("RUNTIME_ERROR - Exit code: 1"))
    val bad = HfDeploy({ "nope" }, hub)
    val e17 = throws<HfDeploy.HubException> { bad.whoami() }
    check("K17 a wrong token -> 401 'rejected'", e17 != null && e17.code == 401 && e17.message!!.contains("rejected"))
    val none = HfDeploy({ null }, hub)
    check("K17 no token -> clear error, no request sent", throws<HfDeploy.HubException> { none.whoami() }?.message?.contains("No Hugging Face token") == true)
    check("K18 an invalid repo id is never sent to the Hub", !d.deploy("../../etc", pkg, null, true, "x").ok)
    // conformance: the same files, the official client vs this one
    val conf = d.deploy("Swamivicky/AppConformance", HfDeploy.SpacePackage(pkg.files.filter { it.path != "README.md" }, pkg.version, pkg.engine, emptyList()), null, false, "conformance")
    check("K19 conformance commit made", conf.ok && conf.changed == 7)
    println("\n$pass/${pass + fail} passed")
    kotlin.system.exitProcess(if (fail == 0) 0 else 1)
}
