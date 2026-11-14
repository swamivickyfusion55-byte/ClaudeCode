import androidx.compose.ui.graphics.Color
private fun spaceStatusColor(stage: String?, busy: Boolean): Color = when {
    stage == "PAUSED" -> Color(0xFFD69E2E)
    stage != null && (stage.endsWith("ERROR") || stage == "NO_APP_FILE") -> Color(0xFFD92D20)
    busy -> Color(0xFF2886DE)
    stage == "RUNNING" -> Color(0xFF1E9E5A)
    stage in setOf("BUILDING", "RUNNING_BUILDING", "APP_STARTING", "RUNNING_APP_STARTING") -> Color(0xFF7CB9F0)
    else -> Color(0xFF94A3B8)
}


object Vm {
    fun spaceStageText(stage: String?): String = when (stage) {
        null -> "Not checked yet"
        "RUNNING" -> "Running"
        "PAUSED" -> "Paused"
        "SLEEPING" -> "Asleep (wakes on the next job)"
        "BUILDING", "RUNNING_BUILDING" -> "Building"
        "APP_STARTING", "RUNNING_APP_STARTING" -> "Starting"
        "STOPPED" -> "Stopped"
        "BUILD_ERROR" -> "Build error"
        "RUNTIME_ERROR" -> "Runtime error"
        "CONFIG_ERROR" -> "Configuration error"
        "NO_APP_FILE" -> "No app file"
        "DELETING" -> "Being deleted"
        "UNKNOWN" -> "Status not read"
        else -> stage.lowercase().replace('_', ' ').replaceFirstChar { it.uppercase() }
    }


}
fun main() {
    var fails = 0
    fun check(n: String, ok: Boolean, x: String = "") { println((if (ok) "PASS " else "FAIL ") + n + (if (x.isNotEmpty()) "  [$x]" else "")); if (!ok) fails++ }
    val green = Color(0xFF1E9E5A); val amber = Color(0xFFD69E2E); val red = Color(0xFFD92D20); val blue = Color(0xFF2886DE); val light = Color(0xFF7CB9F0); val grey = Color(0xFF94A3B8)
    val cases = listOf(
        Triple("RUNNING", false, green to "Running"), Triple("RUNNING", true, blue to "Running"),
        Triple("PAUSED", false, amber to "Paused"), Triple("PAUSED", true, amber to "Paused"),
        Triple("RUNTIME_ERROR", false, red to "Runtime error"), Triple("BUILD_ERROR", true, red to "Build error"),
        Triple("CONFIG_ERROR", false, red to "Configuration error"), Triple("NO_APP_FILE", false, red to "No app file"),
        Triple("BUILDING", false, light to "Building"), Triple("RUNNING_APP_STARTING", false, light to "Starting"),
        Triple("SLEEPING", false, grey to "Asleep (wakes on the next job)"), Triple(null, false, grey to "Not checked yet"),
        Triple("UNKNOWN", false, grey to "Status not read"), Triple("STOPPED", false, grey to "Stopped"))
    for ((st, busy, exp) in cases) {
        val c = spaceStatusColor(st, busy); val t = Vm.spaceStageText(st)
        check("$st${if (busy) " + job" else ""}: colour and words", c == exp.first && t == exp.second, t)
    }
    println(if (fails == 0) "all checks passed" else "$fails check(s) failed")
}
