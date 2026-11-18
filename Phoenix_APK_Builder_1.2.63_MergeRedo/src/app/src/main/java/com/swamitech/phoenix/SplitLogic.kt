package com.swamitech.phoenix

/**
 * 1.2.63: which pieces of a split video are joined, and which History entries are kept. Plain Kotlin (no Android), so it is
 * tested on its own (SplitLogicTest in the builder's verification folder).
 *
 * Asked: "when the video processing stops, stating it is not visible, and you redo it, after redo the merge doesn't happen".
 * Causes found in 1.2.62:
 *  - History kept the newest 50 entries only. A split of several parts, its redone part and the other jobs in between pushed
 *    the oldest parts of the same video out, and the merge then waited for parts it could no longer see.
 *  - The automatic merge after a part finished asked for "the first group that can be merged" in History and merged only when
 *    that was this part's group; any other group left complete but unmerged (its merge had failed once) blocked it.
 *  - A group merged once was never merged again: a part redone afterwards was left out, and the new merge would have been
 *    saved under the old file name, which hands back the OLD merged video.
 *  - Refresh removed a finished part whose file had been moved or deleted, so its card (and its Redo part button) vanished.
 */
object SplitLogic {

    /** The fields of a History entry the merge needs. */
    data class Part(
        val id: String,
        val kind: String,
        val status: String,
        val uri: String?,
        val groupId: String?,
        val partIndex: Int,
        val partCount: Int,
        val createdAt: Long
    )

    /** What to join for one group: the chosen entry for each part (in order), the part numbers (0-based) still missing. */
    data class Plan(val gid: String, val count: Int, val chosen: List<Part>, val missing: List<Int>, val merged: Boolean)

    private fun isFinishedPart(p: Part) = p.kind == "Video" && p.status == "Done" && !p.uri.isNullOrBlank()

    /**
     * For each part number the NEWEST finished entry (a redone part replaces the first one; an old entry of the same part that
     * finished as well is not joined twice). [merged] is true when a merge of this group is newer than every chosen part -
     * a part redone after a merge makes the group mergeable again.
     */
    fun plan(items: List<Part>, gid: String): Plan {
        val group = items.filter { it.groupId == gid }
        val count = group.filter { it.kind == "Video" }.maxOfOrNull { it.partCount } ?: 0
        val chosen = mutableListOf<Part>()
        val missing = mutableListOf<Int>()
        for (i in 0 until count) {
            val best = group.filter { isFinishedPart(it) && it.partIndex == i }.maxByOrNull { it.createdAt }
            if (best == null) missing.add(i) else chosen.add(best)
        }
        val newestPart = chosen.maxOfOrNull { it.createdAt } ?: Long.MAX_VALUE
        val merged = group.any { it.kind == "Merged" && it.status == "Done" && it.createdAt >= newestPart }
        return Plan(gid, count, chosen, missing, merged)
    }

    fun mergeable(items: List<Part>, gid: String): Boolean {
        val p = plan(items, gid)
        return p.count >= 2 && p.missing.isEmpty() && !p.merged
    }

    /** The newest group in History that can be merged now (History is newest first), or null. */
    fun newestMergeable(items: List<Part>): String? =
        items.mapNotNull { it.groupId }.distinct().firstOrNull { mergeable(items, it) }

    /** File name for a merge of [gid]: the first one keeps 1.2.62's name, a later one gets _2, _3 ... (a new file, never the old one). */
    fun mergedName(items: List<Part>, gid: String): String {
        val before = items.count { it.groupId == gid && it.kind == "Merged" && it.status == "Done" }
        return if (before == 0) "Phoenix_merged_${gid}.mp4" else "Phoenix_merged_${gid}_${before + 1}.mp4"
    }

    /**
     * History after a new entry: the newest [keep] entries as before, and besides them every entry of a split that is not merged
     * yet (up to [hardCap] entries in all), so no part of a video still to be joined is dropped.
     */
    fun trim(items: List<Part>, keep: Int = 50, hardCap: Int = 300): List<Part> {
        if (items.size <= keep) return items
        val open = items.mapNotNull { it.groupId }.distinct().filter { gid ->
            val p = plan(items, gid)
            p.count >= 2 && !p.merged
        }.toSet()
        val out = items.filterIndexed { i, it -> i < keep || (it.groupId != null && it.groupId in open) }
        return out.take(hardCap)
    }
}
