package com.swamitech.phoenix

import android.app.Activity
import android.app.Application
import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.os.Process
import java.io.File
import kotlin.system.exitProcess

/**
 * Pure decisions behind the crash guard (no Android classes, so they are unit-tested on the JVM).
 */
object GuardLogic {
    /** Crashes closer together than this count as one storm. */
    const val WINDOW_MS = 120_000L

    /** Automatic restarts allowed inside the window; the next crash shows Android's normal behaviour. */
    const val MAX_AUTO_RELAUNCH = 2

    /** This many crashes inside the window put the next launch into safe mode. */
    const val SAFE_MODE_AT = 3

    fun parseTimes(raw: String?): List<Long> =
        raw.orEmpty().split(',').mapNotNull { it.trim().toLongOrNull() }

    fun joinTimes(times: List<Long>): String = times.takeLast(10).joinToString(",")

    fun recent(times: List<Long>, now: Long): List<Long> =
        times.filter { now - it in 0..WINDOW_MS }

    /** [crashesInWindow] already includes the crash being handled. */
    fun shouldRelaunch(appVisible: Boolean, crashesInWindow: Int): Boolean =
        appVisible && crashesInWindow <= MAX_AUTO_RELAUNCH

    fun safeMode(crashesInWindow: Int): Boolean = crashesInWindow >= SAFE_MODE_AT

    /** First lines of an error, with the root cause, short enough to show on a phone. */
    fun describe(e: Throwable): String {
        val sb = StringBuilder()
        sb.append(e.javaClass.name).append(": ").append(e.message ?: "")
        e.stackTrace.take(6).forEach { sb.append("\n  at ").append(it) }
        var root: Throwable = e
        var guard = 0
        while (root.cause != null && root.cause !== root && guard++ < 8) root = root.cause!!
        if (root !== e) sb.append("\nCaused by: ").append(root.javaClass.name).append(": ").append(root.message ?: "")
        return sb.toString().take(1200)
    }

    /** Appends one entry and keeps only the newest [keep] entries / [maxChars] characters. */
    fun appendLog(old: String, entry: String, keep: Int = 6, maxChars: Int = 7000): String {
        val entries = old.split("\n=====\n").filter { it.isNotBlank() } + entry
        var out = entries.takeLast(keep).joinToString("\n=====\n")
        if (out.length > maxChars) out = out.takeLast(maxChars)
        return out
    }

    /**
     * History is read from saved text. One damaged entry must not lose the others, and an unreadable file
     * must be kept aside (never silently overwritten by an empty list).
     */
    fun <T> loadList(raw: String?, parseAll: (String) -> List<T?>, backup: (String) -> Unit): List<T> {
        val text = raw ?: "[]"
        return try {
            parseAll(text).filterNotNull()
        } catch (e: Throwable) {
            runCatching { backup(text) }
            emptyList()
        }
    }
}

/**
 * Keeps the app from vanishing.
 *
 *  - Records every uncaught error (ring log in the app's own folder, shown on the next launch).
 *  - If the app was on screen when it crashed, starts it again straight away so it reconnects to the
 *    running jobs, at most twice in two minutes.
 *  - Three crashes in two minutes put the next launch into safe mode: the app opens and shows what
 *    happened but does not touch the running jobs until you tap "Resume jobs".
 *  - Background work started with [handler] cannot take the app down.
 */
object Guard {
    private const val PREFS = "phoenix_guard"
    private const val LOG = "crash_log.txt"
    private const val LAST = "last_crash.txt"

    @Volatile private var started = 0
    @Volatile private var installed = false

    fun install(app: Application) {
        if (installed) return
        installed = true
        runCatching {
            app.registerActivityLifecycleCallbacks(object : Application.ActivityLifecycleCallbacks {
                override fun onActivityStarted(activity: Activity) { started++ }
                override fun onActivityStopped(activity: Activity) { if (started > 0) started-- }
                override fun onActivityCreated(activity: Activity, savedInstanceState: Bundle?) {}
                override fun onActivityResumed(activity: Activity) {}
                override fun onActivityPaused(activity: Activity) {}
                override fun onActivitySaveInstanceState(activity: Activity, outState: Bundle) {}
                override fun onActivityDestroyed(activity: Activity) {}
            })
        }
        val previous = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { thread, error ->
            var relaunched = false
            try {
                record(app, "crash", error)
                runCatching { File(app.filesDir, LAST).writeText(GuardLogic.describe(error)) }
                val now = System.currentTimeMillis()
                val prefs = app.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
                val times = GuardLogic.recent(GuardLogic.parseTimes(prefs.getString("crash_times", null)), now) + now
                prefs.edit().putString("crash_times", GuardLogic.joinTimes(times)).commit()
                if (GuardLogic.shouldRelaunch(started > 0, times.size)) {
                    prefs.edit().putLong("recovered_at", now).commit()
                    app.startActivity(
                        Intent(app, MainActivity::class.java)
                            .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK)
                    )
                    relaunched = true
                }
            } catch (_: Throwable) {
                // The guard itself must never make things worse.
            }
            if (relaunched) {
                Process.killProcess(Process.myPid())
                exitProcess(10)
            } else {
                previous?.uncaughtException(thread, error)
            }
        }
    }

    /** Add to every launch(...) so an error in background work is recorded instead of ending the app. */
    fun handler(app: Application, onError: (Throwable) -> Unit): kotlinx.coroutines.CoroutineExceptionHandler =
        kotlinx.coroutines.CoroutineExceptionHandler { _, e ->
            record(app, "background task", e)
            runCatching { onError(e) }
        }

    fun record(app: Application, kind: String, e: Throwable) {
        runCatching {
            val f = File(app.filesDir, LOG)
            val stamp = java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss", java.util.Locale.US).format(java.util.Date())
            val old = if (f.isFile) f.readText() else ""
            f.writeText(GuardLogic.appendLog(old, "$stamp  [$kind]\n${GuardLogic.describe(e)}"))
        }
    }

    fun inSafeMode(app: Application): Boolean = runCatching {
        val now = System.currentTimeMillis()
        val times = GuardLogic.parseTimes(app.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getString("crash_times", null))
        GuardLogic.safeMode(GuardLogic.recent(times, now).size)
    }.getOrDefault(false)

    fun clearCrashTimes(app: Application) {
        runCatching { app.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit().remove("crash_times").commit() }
    }

    /** True once after the guard restarted the app by itself. */
    fun takeRecovered(app: Application): Boolean = runCatching {
        val prefs = app.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        val at = prefs.getLong("recovered_at", 0L)
        if (at != 0L) prefs.edit().remove("recovered_at").commit()
        at != 0L && System.currentTimeMillis() - at < 10 * 60_000L
    }.getOrDefault(false)
}
