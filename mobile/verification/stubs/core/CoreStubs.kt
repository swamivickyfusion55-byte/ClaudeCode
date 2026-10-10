@file:Suppress("unused", "UNUSED_PARAMETER")
package androidx.core.content

import android.content.Context
import android.content.Intent

object ContextCompat {
    @JvmStatic fun startForegroundService(context: Context, intent: Intent) {}
    @JvmStatic fun checkSelfPermission(context: Context, permission: String): Int = 0
}
