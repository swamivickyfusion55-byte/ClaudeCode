package com.swamitech.phoenix

import android.content.Context
import android.net.Uri
import java.io.File

object LocalFileBridge {
    fun copyToCache(context: Context, uri: Uri, prefix: String = "input"): File {
        val mime = context.contentResolver.getType(uri)
        val suffix = when {
            mime == "video/mp4" -> ".mp4"
            mime?.startsWith("video/") == true -> ".mp4"
            mime?.startsWith("image/") == true -> ".jpg"
            else -> ""
        }
        val out = File(context.cacheDir, "${prefix}_${System.currentTimeMillis()}$suffix")
        context.contentResolver.openInputStream(uri).use { input ->
            requireNotNull(input) { "Unable to open selected file" }
            out.outputStream().use { output -> input.copyTo(output) }
        }
        return out
    }
}
