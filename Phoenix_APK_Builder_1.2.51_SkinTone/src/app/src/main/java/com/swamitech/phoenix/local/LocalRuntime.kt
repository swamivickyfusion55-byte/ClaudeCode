package com.swamitech.phoenix.local

import android.content.Context

/**
 * Local model-pack gate. Do NOT touch ONNX Runtime here: OrtEnvironment.getEnvironment()
 * loads native .so files at class init, which crashes cold start on 16 KB-page
 * Android 15 devices and is unused in Cloud mode (the licensed swap graph is
 * not shipped). Load ORT only if a future local graph is actually installed.
 */
class LocalRuntime(context: Context) {
    private val manager = ModelPackManager(context)

    fun isReady(): Boolean = manager.isComplete()

    fun initialize(): String {
        require(isReady()) { "Install a verified Phoenix model pack first" }
        return "Local model pack present; licensed face-swap graph not installed"
    }

    fun processingReady(): Boolean = false

    fun engineStatus(): String =
        if (processingReady()) "Local face engine ready" else
            "Local model runtime ready; licensed face-swap graph not installed"

    fun close() {}
}
