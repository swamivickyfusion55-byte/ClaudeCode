package com.swamitech.phoenix

import android.app.Application

/** Installs the crash guard before anything else in the process runs. */
class PhoenixApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        Guard.install(this)
    }
}
