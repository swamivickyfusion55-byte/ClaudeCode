import com.swamitech.phoenix.FaceLibrary
import com.swamitech.phoenix.SavedFace
import java.io.File

class FilesCtx(private val dir: File) : android.content.ContextWrapper(null) {
    override fun getFilesDir(): File = dir
}
var ok = true
fun check(name: String, c: Boolean, extra: String = "") { ok = ok && c; println((if (c) "PASS " else "FAIL ") + name + if (extra.isNotEmpty()) " | $extra" else "") }

fun main() {
    val root = File("/tmp/claude-0/facelib_test").apply { deleteRecursively(); mkdirs() }
    val faces = File(root, "faces").apply { mkdirs() }
    // two pictures "already saved" (add() itself needs Android's bitmap codec, which the JVM does not have)
    for (id in listOf("aaaa1111", "bbbb2222")) { File(faces, "$id.jpg").writeBytes(ByteArray(10)); File(faces, "${id}_t.jpg").writeBytes(ByteArray(5)) }
    File(faces, "index.json").writeText("""[{"id":"aaaa1111","name":"Priya","created":1},{"id":"bbbb2222","name":"Rahul","created":2},{"id":"cccc3333","name":"Gone","created":3}]""")
    val lib = FaceLibrary(FilesCtx(root))
    val l = lib.list()
    check("list reads names in order and skips an entry whose picture is missing", l.map { it.name } == listOf("Priya", "Rahul"), l.map { it.name }.toString())
    check("a face's Uri is a file Uri to its picture", l[0].uri.toString().endsWith("/faces/aaaa1111.jpg") && l[0].uri.scheme == "file", l[0].uri.toString())
    check("byUri finds the saved face behind a slot's Uri", lib.byUri(l[1].uri)?.name == "Rahul")
    check("rename", lib.rename("aaaa1111", "  Priya S.  ") && lib.list()[0].name == "Priya S.")
    check("renaming to a name in use gets a number", lib.rename("bbbb2222", "priya s.") && lib.list()[1].name == "priya s. 2", lib.list()[1].name)
    check("names are cleaned (newlines out, 40 characters max, blank -> Face)", FaceLibrary.cleanName("a\nb") == "a b" && FaceLibrary.cleanName("x".repeat(60)).length == 40 && FaceLibrary.cleanName("  ") == "Face")
    check("unique names: Priya, Priya 2, Priya 3", FaceLibrary.uniqueName("Priya", listOf(sf("Priya"), sf("Priya 2"))) == "Priya 3")
    check("delete removes the entry and both files", lib.delete("aaaa1111") && lib.list().size == 1 && !File(faces, "aaaa1111.jpg").exists() && !File(faces, "aaaa1111_t.jpg").exists())
    check("delete of an unknown id is a no-op", !lib.delete("nope"))
    File(faces, "index.json").writeText("{ this is not json")
    val rec = lib.list()
    check("a damaged index is kept aside and the faces are recovered from the folder", rec.size == 1 && rec[0].id == "bbbb2222" && faces.listFiles()!!.any { it.name.startsWith("index_unreadable_") }, rec.map { it.name }.toString())
    check("the rebuilt index is readable afterwards", FaceLibrary(FilesCtx(root)).list().size == 1)
    println(if (ok) "ALL PASS" else "FAILURES")
}
fun sf(n: String) = SavedFace("x", n, File("x"), File("y"), 0)
