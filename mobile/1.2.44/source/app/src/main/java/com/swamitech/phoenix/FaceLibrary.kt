package com.swamitech.phoenix

import android.content.ContentResolver
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.util.UUID

/** One saved replacement face: a name, the picture, and a small thumbnail for the drop-down. */
data class SavedFace(
    val id: String,
    val name: String,
    val image: File,
    val thumb: File,
    val createdMs: Long
) {
    /** The Uri the job code already understands (it reads faces through the ContentResolver). */
    val uri: Uri get() = Uri.fromFile(image)
}

/**
 * The phone's own library of replacement faces (1.2.23).
 *
 * Kept in the app's private folder (files/faces): the pictures never leave the phone except as the
 * replacement face of a job, exactly as a face picked from the gallery does, and they are removed with the
 * app. index.json holds id / name / created; each face is a JPEG (longest side at most 1280 px, so the
 * Space gets the same detail as before) plus a 160-px square thumbnail for the drop-down.
 *
 * Every method is synchronous file work: call it off the main thread. A damaged index is kept aside, never
 * overwritten, and the faces whose files still exist are recovered from the folder.
 */
class FaceLibrary(context: Context) {
    private val dir = File(context.filesDir, "faces").apply { mkdirs() }
    private val index = File(dir, "index.json")
    private val lock = Any()

    fun list(): List<SavedFace> = synchronized(lock) { readIndex() }

    fun byUri(uri: Uri?): SavedFace? {
        if (uri == null || uri.scheme != "file") return null
        val path = uri.path ?: return null
        return list().firstOrNull { it.image.absolutePath == path }
    }

    /** Copy the picture behind [source] into the library under [name]. Returns the new face. */
    fun add(resolver: ContentResolver, source: Uri, name: String): SavedFace = synchronized(lock) {
        val clean = cleanName(name)
        val bmp = decode(resolver, source, 1280) ?: throw IllegalStateException("Could not read that picture")
        val id = UUID.randomUUID().toString().replace("-", "").take(16)
        val image = File(dir, "$id.jpg")
        val thumb = File(dir, "${id}_t.jpg")
        try {
            writeJpeg(bmp, image, 95)
            writeJpeg(squareThumb(bmp, 160), thumb, 88)
        } catch (e: Throwable) {
            image.delete(); thumb.delete()
            throw e
        } finally {
            bmp.recycle()
        }
        val face = SavedFace(id, uniqueName(clean, readIndex()), image, thumb, System.currentTimeMillis())
        writeIndex(readIndex() + face)
        face
    }

    fun rename(id: String, name: String): Boolean = synchronized(lock) {
        val all = readIndex()
        if (all.none { it.id == id }) return false
        val others = all.filter { it.id != id }
        writeIndex(all.map { if (it.id == id) it.copy(name = uniqueName(cleanName(name), others)) else it })
        true
    }

    fun delete(id: String): Boolean = synchronized(lock) {
        val all = readIndex()
        val gone = all.firstOrNull { it.id == id } ?: return false
        writeIndex(all.filter { it.id != id })
        gone.image.delete(); gone.thumb.delete()
        true
    }

    /** The thumbnail as a bitmap, or null. Small file, decoded directly. */
    fun thumbnail(face: SavedFace): Bitmap? = runCatching { BitmapFactory.decodeFile(face.thumb.absolutePath) }.getOrNull()

    // ------------------------------------------------------------------ storage

    private fun readIndex(): List<SavedFace> {
        if (!index.isFile) return emptyList()
        val text = runCatching { index.readText() }.getOrNull() ?: return emptyList()
        val arr = runCatching { JSONArray(text) }.getOrNull()
        if (arr == null) {
            // keep the damaged file for inspection and rebuild from the pictures that are there
            runCatching { index.copyTo(File(dir, "index_unreadable_${System.currentTimeMillis()}.json"), overwrite = true) }
            val rebuilt = dir.listFiles { f -> f.name.endsWith(".jpg") && !f.name.endsWith("_t.jpg") }.orEmpty()
                .sortedBy { it.lastModified() }
                .mapIndexed { i, f ->
                    val id = f.nameWithoutExtension
                    SavedFace(id, "Face ${i + 1}", f, File(dir, "${id}_t.jpg"), f.lastModified())
                }
            writeIndex(rebuilt)
            return rebuilt
        }
        val out = ArrayList<SavedFace>()
        for (i in 0 until arr.length()) {
            val o = arr.optJSONObject(i) ?: continue
            val id = o.optString("id").filter { it.isLetterOrDigit() }
            if (id.isEmpty()) continue
            val image = File(dir, "$id.jpg")
            if (!image.isFile) continue
            out += SavedFace(id, o.optString("name").ifBlank { "Face" }, image, File(dir, "${id}_t.jpg"), o.optLong("created"))
        }
        return out
    }

    private fun writeIndex(faces: List<SavedFace>) {
        val arr = JSONArray()
        faces.forEach { arr.put(JSONObject().put("id", it.id).put("name", it.name).put("created", it.createdMs)) }
        val tmp = File(dir, "index.json.tmp")
        tmp.writeText(arr.toString())
        if (!tmp.renameTo(index)) {
            index.delete()
            if (!tmp.renameTo(index)) throw IllegalStateException("Could not save the face list")
        }
    }

    companion object {
        fun cleanName(name: String): String =
            name.replace(Regex("[\\r\\n\\t]"), " ").trim().take(40).ifBlank { "Face" }

        /** "Priya" twice becomes "Priya" and "Priya 2", so the drop-down never shows two identical rows. */
        fun uniqueName(name: String, existing: List<SavedFace>): String {
            val taken = existing.map { it.name.lowercase() }.toSet()
            if (name.lowercase() !in taken) return name
            var n = 2
            while ("$name $n".lowercase() in taken) n++
            return "$name $n"
        }

        private fun decode(resolver: ContentResolver, uri: Uri, maxSide: Int): Bitmap? {
            val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            resolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, bounds) }
            if (bounds.outWidth <= 0 || bounds.outHeight <= 0) return null
            var sample = 1
            while (maxOf(bounds.outWidth, bounds.outHeight) / (sample * 2) >= maxSide) sample *= 2
            val opts = BitmapFactory.Options().apply { inSampleSize = sample }
            var bmp = resolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, opts) } ?: return null
            val rot = exifRotation(resolver, uri)
            if (rot != 0) {
                val m = Matrix().apply { postRotate(rot.toFloat()) }
                val r = Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)
                if (r != bmp) bmp.recycle()
                bmp = r
            }
            val longest = maxOf(bmp.width, bmp.height)
            if (longest > maxSide) {
                val s = maxSide.toFloat() / longest
                val r = Bitmap.createScaledBitmap(bmp, (bmp.width * s).toInt().coerceAtLeast(1), (bmp.height * s).toInt().coerceAtLeast(1), true)
                if (r != bmp) bmp.recycle()
                bmp = r
            }
            return bmp
        }

        private fun exifRotation(resolver: ContentResolver, uri: Uri): Int = runCatching {
            resolver.openInputStream(uri)?.use { input ->
                when (android.media.ExifInterface(input).getAttributeInt(
                    android.media.ExifInterface.TAG_ORIENTATION, android.media.ExifInterface.ORIENTATION_NORMAL)) {
                    android.media.ExifInterface.ORIENTATION_ROTATE_90 -> 90
                    android.media.ExifInterface.ORIENTATION_ROTATE_180 -> 180
                    android.media.ExifInterface.ORIENTATION_ROTATE_270 -> 270
                    else -> 0
                }
            } ?: 0
        }.getOrDefault(0)

        private fun squareThumb(src: Bitmap, side: Int): Bitmap {
            val s = minOf(src.width, src.height)
            val crop = Bitmap.createBitmap(src, (src.width - s) / 2, (src.height - s) / 2, s, s)
            val out = Bitmap.createScaledBitmap(crop, side, side, true)
            if (crop != out && crop != src) crop.recycle()
            return out
        }

        private fun writeJpeg(bmp: Bitmap, file: File, quality: Int) {
            FileOutputStream(file).use { out ->
                if (!bmp.compress(Bitmap.CompressFormat.JPEG, quality, out)) throw IllegalStateException("Could not save the picture")
            }
        }
    }
}
