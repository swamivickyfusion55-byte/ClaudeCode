// 1.2.63: SplitLogic against the History situations behind "after Redo part the merge doesn't happen".
// Plain JVM: kotlinc SplitLogic.kt SplitLogicTest.kt, run MainKt.
import com.swamitech.phoenix.SplitLogic
import com.swamitech.phoenix.SplitLogic.Part

var fails = 0
fun check(name: String, ok: Boolean, extra: String = "") {
    println((if (ok) "PASS " else "FAIL ") + name + (if (extra.isNotEmpty()) "  [$extra]" else ""))
    if (!ok) fails++
}
fun part(id: String, gid: String, idx: Int, n: Int, status: String, t: Long, uri: String? = if (status == "Done") "content://media/$id" else null) =
    Part(id, "Video", status, uri, gid, idx, n, t)
fun merged(id: String, gid: String, n: Int, t: Long) = Part(id, "Merged", "Done", "content://media/$id", gid, 0, n, t)

fun main() {
    // 1. part 2 of 3 was lost on its Space, then redone: the old card is Replaced, the new one Done
    val g = listOf(
        part("p2b", "G", 1, 3, "Done", 400), part("p3", "G", 2, 3, "Done", 120),
        part("p2", "G", 1, 3, "Replaced", 110, null), part("p1", "G", 0, 3, "Done", 100)
    )
    val pl = SplitLogic.plan(g, "G")
    check("a redone part completes the group: it can be merged", SplitLogic.mergeable(g, "G"))
    check("...the parts are joined in order, the redone one in place of the lost one", pl.chosen.map { it.id } == listOf("p1", "p2b", "p3"), pl.chosen.map { it.id }.toString())

    // 2. the lost job came back and finished too: two finished copies of part 2 - joined once, the newest
    val g2 = listOf(part("p2old", "G", 1, 3, "Done", 110)) + g.filter { it.id != "p2" }
    val pl2 = SplitLogic.plan(g2, "G")
    check("two finished copies of one part: joined once (the newest), 3 parts not 4", pl2.chosen.map { it.id } == listOf("p1", "p2b", "p3"), pl2.chosen.map { it.id }.toString())

    // 3. 1.2.62: another, newer split complete but not merged (its merge had failed) blocked this one's automatic merge
    val other = listOf(part("q1", "H", 0, 2, "Done", 500), part("q2", "H", 1, 2, "Done", 510))
    val both = other + g
    check("another complete, unmerged split does not stop this one from being merged", SplitLogic.mergeable(both, "G"))
    check("...Merge parts picks the newest mergeable split", SplitLogic.newestMergeable(both) == "H")

    // 4. merged, then a part redone: the group can be merged again, as a new file
    val m = listOf(merged("m1", "G", 3, 300)) + g.filter { it.id != "p2b" } + listOf(part("p2c", "G", 1, 3, "Replaced", 200, null))
    val withDone2 = m + part("p2x", "G", 1, 3, "Done", 150)
    check("a merged split is not merged again by itself", !SplitLogic.mergeable(withDone2, "G"))
    val redoneAfter = withDone2 + part("p2y", "G", 1, 3, "Done", 350)
    check("a part redone after the merge makes it mergeable again", SplitLogic.mergeable(redoneAfter, "G"))
    check("...and the new merge gets a new file name (1.2.62 reused the old one and got the old video back)",
        SplitLogic.mergedName(redoneAfter, "G") == "Phoenix_merged_G_2.mp4", SplitLogic.mergedName(redoneAfter, "G"))
    check("the first merge keeps 1.2.62's name", SplitLogic.mergedName(g, "G") == "Phoenix_merged_G.mp4")

    // 5. a missing part (lost, not redone yet; or its file gone - Missing) holds the merge and is named
    val lost = listOf(part("a1", "K", 0, 4, "Done", 1), part("a2", "K", 1, 4, "Lost", 2, null), part("a3", "K", 2, 4, "Missing", 3, "content://x"), part("a4", "K", 3, 4, "Done", 4))
    val pk = SplitLogic.plan(lost, "K")
    check("a lost part and a part whose file is gone hold the merge", !SplitLogic.mergeable(lost, "K") && pk.missing == listOf(1, 2), pk.missing.toString())

    // 6. History: 1.2.62 kept the newest 50 entries - the first parts of a long split, then 60 other jobs, then the redone part
    val early = listOf(part("e1", "S", 0, 3, "Done", 10), part("e2", "S", 1, 3, "Lost", 11, null), part("e3", "S", 2, 3, "Done", 12))
    val oldMerged = listOf(part("o1", "M", 0, 2, "Done", 1), part("o2", "M", 1, 2, "Done", 2), merged("om", "M", 2, 3))
    val jobs = (0 until 60).map { Part("j$it", "Video", "Done", "content://j$it", null, 0, 1, 1000L + it) }.reversed()
    val hist = listOf(part("e2b", "S", 1, 3, "Done", 2000)) + jobs + early + oldMerged
    val kept = SplitLogic.trim(hist).map { it.id }.toSet()
    check("History keeps every part of a split not merged yet beyond the newest 50", listOf("e1", "e2", "e3", "e2b").all { it in kept })
    check("...so that split can still be merged", SplitLogic.mergeable(hist.filter { it.id in kept }, "S"))
    check("...entries beyond 50 of a merged split and of single jobs still go", "o1" !in kept && "om" !in kept && "j0" !in kept, (hist.size - kept.size).toString() + " dropped")
    check("...and at most 50 + the open split's entries are kept", kept.size == 50 + 3, kept.size.toString())
    val many = (0 until 400).map { part("x$it", "X", it % 300, 300, "Done", it.toLong()) }
    check("a hard cap of 300 entries in all", SplitLogic.trim(many).size == 300)

    // 7. a single-video job is never a split
    check("no split: nothing to merge", SplitLogic.newestMergeable(jobs) == null)
    println(if (fails == 0) "\nall checks passed" else "\n$fails check(s) failed")
}
