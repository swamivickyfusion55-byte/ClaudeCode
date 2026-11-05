package com.swamitech.phoenix.ui

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.Build
import android.provider.OpenableColumns
import androidx.compose.runtime.Composable
import androidx.compose.runtime.State
import androidx.compose.runtime.produceState
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Everything the UI needs to render a real preview tile for a picked media item.
 * The previous build only handled "file://" URIs, so every document-picker
 * result (which is always "content://") silently produced no thumbnail.
 */
data class ThumbState(
    val bitmap: ImageBitmap? = null,
    val displayName: String? = null,
    val sizeBytes: Long = -1L,
    val durationMs: Long = -1L,
    val pixelWidth: Int = 0,
    val pixelHeight: Int = 0,
    val loading: Boolean = false,
    val error: String? = null
) {
    val hasImage: Boolean get() = bitmap != null

    fun sizeLabel(): String? {
        if (sizeBytes <= 0L) return null
        val mb = sizeBytes / 1_048_576.0
        return if (mb >= 1.0) String.format("%.1f MB", mb) else String.format("%d KB", sizeBytes / 1024)
    }

    fun durationLabel(): String? {
        if (durationMs <= 0L) return null
        val total = durationMs / 1000
        val m = total / 60
        val s = total % 60
        return String.format("%d:%02d", m, s)
    }

    fun dimensionLabel(): String? =
        if (pixelWidth > 0 && pixelHeight > 0) "${pixelWidth}×${pixelHeight}" else null
}

/**
 * Loads a preview off the main thread. Re-runs whenever the URI or the
 * requested video frame position changes.
 */
@Composable
fun rememberThumb(
    uri: Uri?,
    isVideo: Boolean,
    framePercent: Int = 0,
    maxPx: Int = 640
): State<ThumbState> {
    val context = LocalContext.current
    return produceState(
        initialValue = ThumbState(loading = uri != null),
        uri,
        isVideo,
        framePercent,
        maxPx
    ) {
        val target = uri
        if (target == null) {
            value = ThumbState()
            return@produceState
        }
        value = ThumbState(loading = true)
        value = withContext(Dispatchers.IO) {
            loadThumb(context, target, isVideo, framePercent, maxPx)
        }
    }
}

private fun loadThumb(
    context: Context,
    uri: Uri,
    isVideo: Boolean,
    framePercent: Int,
    maxPx: Int
): ThumbState {
    val meta = readOpenableMeta(context, uri)
    return try {
        if (isVideo) loadVideoThumb(context, uri, framePercent, maxPx, meta)
        else loadImageThumb(context, uri, maxPx, meta)
    } catch (t: Throwable) {
        ThumbState(
            displayName = meta.first,
            sizeBytes = meta.second,
            error = t.message ?: t.javaClass.simpleName
        )
    }
}

/** DISPLAY_NAME + SIZE work for content:// providers; file:// falls back to the path. */
private fun readOpenableMeta(context: Context, uri: Uri): Pair<String?, Long> {
    if (uri.scheme == "file") {
        val f = uri.path?.let { java.io.File(it) }
        return (f?.name) to (f?.length() ?: -1L)
    }
    return runCatching {
        context.contentResolver.query(uri, null, null, null, null)?.use { c ->
            if (!c.moveToFirst()) return@use null to -1L
            val nameIdx = c.getColumnIndex(OpenableColumns.DISPLAY_NAME)
            val sizeIdx = c.getColumnIndex(OpenableColumns.SIZE)
            val name = if (nameIdx >= 0 && !c.isNull(nameIdx)) c.getString(nameIdx) else null
            val size = if (sizeIdx >= 0 && !c.isNull(sizeIdx)) c.getLong(sizeIdx) else -1L
            name to size
        } ?: (null to -1L)
    }.getOrDefault(null to -1L)
}

private fun loadImageThumb(
    context: Context,
    uri: Uri,
    maxPx: Int,
    meta: Pair<String?, Long>
): ThumbState {
    val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
    context.contentResolver.openInputStream(uri)?.use {
        BitmapFactory.decodeStream(it, null, bounds)
    }
    if (bounds.outWidth <= 0 || bounds.outHeight <= 0) {
        return ThumbState(
            displayName = meta.first,
            sizeBytes = meta.second,
            error = "Unsupported image format"
        )
    }

    var sample = 1
    while (bounds.outWidth / sample > maxPx || bounds.outHeight / sample > maxPx) sample *= 2

    val decodeOptions = BitmapFactory.Options().apply {
        inSampleSize = sample
        inPreferredConfig = Bitmap.Config.ARGB_8888
    }
    val raw = context.contentResolver.openInputStream(uri)?.use {
        BitmapFactory.decodeStream(it, null, decodeOptions)
    } ?: return ThumbState(
        displayName = meta.first,
        sizeBytes = meta.second,
        error = "Could not read image"
    )

    val rotated = applyExifRotation(context, uri, raw)
    return ThumbState(
        bitmap = rotated.asImageBitmap(),
        displayName = meta.first,
        sizeBytes = meta.second,
        pixelWidth = bounds.outWidth,
        pixelHeight = bounds.outHeight
    )
}

private fun applyExifRotation(context: Context, uri: Uri, source: Bitmap): Bitmap {
    val degrees = runCatching {
        context.contentResolver.openInputStream(uri)?.use { stream ->
            val exif = android.media.ExifInterface(stream)
            when (exif.getAttributeInt(
                android.media.ExifInterface.TAG_ORIENTATION,
                android.media.ExifInterface.ORIENTATION_NORMAL
            )) {
                android.media.ExifInterface.ORIENTATION_ROTATE_90 -> 90
                android.media.ExifInterface.ORIENTATION_ROTATE_180 -> 180
                android.media.ExifInterface.ORIENTATION_ROTATE_270 -> 270
                else -> 0
            }
        } ?: 0
    }.getOrDefault(0)

    if (degrees == 0) return source
    return runCatching {
        Bitmap.createBitmap(
            source, 0, 0, source.width, source.height,
            Matrix().apply { postRotate(degrees.toFloat()) }, true
        )
    }.getOrDefault(source)
}

private fun loadVideoThumb(
    context: Context,
    uri: Uri,
    framePercent: Int,
    maxPx: Int,
    meta: Pair<String?, Long>
): ThumbState {
    val retriever = MediaMetadataRetriever()
    try {
        retriever.setDataSource(context, uri)
        val durationMs = retriever
            .extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
            ?.toLongOrNull() ?: -1L
        val width = retriever
            .extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)
            ?.toIntOrNull() ?: 0
        val height = retriever
            .extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)
            ?.toIntOrNull() ?: 0

        val safeDuration = if (durationMs > 0) durationMs else 0L
        val timeUs = (safeDuration * 1000L * framePercent.coerceIn(0, 100) / 100L)
            .coerceAtLeast(0L)

        val frame: Bitmap? = if (Build.VERSION.SDK_INT >= 27 && width > 0 && height > 0) {
            val scale = maxOf(width, height).toFloat() / maxPx.toFloat()
            val targetW = if (scale > 1f) (width / scale).toInt() else width
            val targetH = if (scale > 1f) (height / scale).toInt() else height
            retriever.getScaledFrameAtTime(
                timeUs,
                MediaMetadataRetriever.OPTION_CLOSEST_SYNC,
                targetW.coerceAtLeast(1),
                targetH.coerceAtLeast(1)
            ) ?: retriever.getFrameAtTime(timeUs, MediaMetadataRetriever.OPTION_CLOSEST_SYNC)
        } else {
            retriever.getFrameAtTime(timeUs, MediaMetadataRetriever.OPTION_CLOSEST_SYNC)
        }

        return ThumbState(
            bitmap = frame?.asImageBitmap(),
            displayName = meta.first,
            sizeBytes = meta.second,
            durationMs = durationMs,
            pixelWidth = width,
            pixelHeight = height,
            error = if (frame == null) "No decodable frame at this position" else null
        )
    } finally {
        runCatching { retriever.release() }
    }
}
