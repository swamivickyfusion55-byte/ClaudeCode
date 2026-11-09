package com.swamitech.phoenix

import android.content.Context
import java.io.File

class ModelPack(private val context: Context) {
    private val root = File(context.filesDir, "phoenix/models")

    /**
     * Final model pack will be hash-verified before loading.
     * The names are logical slots, not commitments to a specific vendor model.
     */
    fun isComplete(): Boolean {
        val required = listOf(
            "detector.onnx",
            "recognizer.onnx",
            "landmarks.onnx",
            "swapper.onnx"
        )
        return required.all { File(root, it).isFile && File(root, it).length() > 0 }
    }

    fun directory(): File = root.apply { mkdirs() }
}
