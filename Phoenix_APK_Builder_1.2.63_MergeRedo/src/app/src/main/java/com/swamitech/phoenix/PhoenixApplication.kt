package com.swamitech.phoenix

import android.app.Application
import android.content.ComponentCallbacks2

/** Installs the crash guard before anything else in the process runs. */
class PhoenixApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        Guard.install(this)
    }

    override fun onTrimMemory(level: Int) {
        super.onTrimMemory(level)
        if (level >= ComponentCallbacks2.TRIM_MEMORY_RUNNING_LOW || level == ComponentCallbacks2.TRIM_MEMORY_UI_HIDDEN) {
            runCatching { onLowMemory?.invoke() }
        }
    }

    companion object {
        @JvmField
        var onLowMemory: (() -> Unit)? = null
    }
}