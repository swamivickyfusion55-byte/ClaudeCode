package com.swamitech.phoenix

import android.content.Context
import android.net.Uri
import java.io.File

/**
 * Native/local processing facade.
 *
 * The UI never decides which face belongs to which person.
 * Identity and occlusion decisions live below this layer.
 */
class PhoenixEngine(private val context: Context) {
    companion object {
        init {
            System.loadLibrary("phoenix_engine")
        }
    }

    val backendInfo: String
        get() = nativeBackendInfo()

    val modelPackReady: Boolean
        get() = ModelPack(context).isComplete()

    suspend fun process(
        uri: Uri,
        mode: String,
        progress: (Float, String, Long) -> Unit,
        completed: (String) -> Unit,
        failed: (String) -> Unit
    ) {
        try {
            val input = LocalFileBridge.copyToCache(context, uri)
            val output = File(
                context.cacheDir,
                "phoenix_${System.currentTimeMillis()}.mp4"
            )

            // M2 currently validates the local pipeline boundary.
            // M3 will connect detector/tracker/swapper inference here.
            val result = nativeProcessVideo(
                input.absolutePath,
                output.absolutePath,
                mode
            ) { p, msg, eta ->
                progress(p, msg, eta)
            }

            if (result == 0) completed(output.absolutePath)
            else failed("Native engine returned $result")
        } catch (t: Throwable) {
            failed(t.message ?: t.javaClass.simpleName)
        }
    }

    private external fun nativeBackendInfo(): String

    private external fun nativeProcessVideo(
        inputPath: String,
        outputPath: String,
        mode: String,
        callback: (Float, String, Long) -> Unit
    ): Int
}
