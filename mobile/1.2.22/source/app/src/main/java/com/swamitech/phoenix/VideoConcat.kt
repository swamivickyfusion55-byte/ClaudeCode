package com.swamitech.phoenix

import android.content.Context
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaMuxer
import android.net.Uri
import java.io.File
import java.nio.ByteBuffer

/**
 * Joins Phoenix parts the same way FFmpeg does `ffmpeg -f concat -c copy`:
 * compressed samples are copied, never decoded. Instant, lossless.
 *
 * Requirements: every part used the same resolution / fps / codec
 * (same Phoenix quality preset).
 */
object VideoConcat {

    fun concat(
        context: Context,
        parts: List<Uri>,
        dest: File,
        onProgress: ((done: Int, total: Int) -> Unit)? = null
    ): Boolean {
        if (parts.size < 2) return false
        var muxer: MediaMuxer? = null
        var started = false
        try {
            val first = open(context, parts[0]) ?: return false
            val vMime = mimeOf(first, videoTrack(first))
            val aMime = audioTrack(first).takeIf { it >= 0 }?.let { mimeOf(first, it) }
            val vDur = frameDurationUs(first, videoTrack(first))
            first.release()

            muxer = MediaMuxer(dest.absolutePath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
            var videoOut = -1
            var audioOut = -1
            var videoCursor = 0L
            var audioCursor = 0L

            parts.forEachIndexed { idx, uri ->
                onProgress?.invoke(idx, parts.size)
                val ex = open(context, uri) ?: return false
                val vIdx = videoTrack(ex)
                if (vIdx < 0) {
                    ex.release()
                    return false
                }
                if (mimeOf(ex, vIdx) != vMime) {
                    ex.release()
                    return false
                }
                val aIdx = audioTrack(ex)
                if (!started) {
                    videoOut = muxer.addTrack(ex.getTrackFormat(vIdx))
                    if (aIdx >= 0 && aMime != null && mimeOf(ex, aIdx) == aMime) {
                        audioOut = muxer.addTrack(ex.getTrackFormat(aIdx))
                    }
                    muxer.start()
                    started = true
                }
                val maxIn = runCatching {
                    ex.getTrackFormat(vIdx).getInteger(MediaFormat.KEY_MAX_INPUT_SIZE)
                }.getOrDefault(0).coerceIn(256 * 1024, 4 * 1024 * 1024)
                val buffer = ByteBuffer.allocateDirect(maxIn)
                val info = MediaCodec.BufferInfo()

                val writtenV = copyTrack(ex, vIdx, muxer, videoOut, videoCursor, buffer, info)
                videoCursor = writtenV + vDur

                if (aIdx >= 0 && audioOut >= 0 && mimeOf(ex, aIdx) == aMime) {
                    val writtenA = copyTrack(ex, aIdx, muxer, audioOut, audioCursor, buffer, info)
                    audioCursor = writtenA + vDur
                } else {
                    audioCursor = videoCursor
                }
                ex.release()
                onProgress?.invoke(idx + 1, parts.size)
            }
            return dest.exists() && dest.length() > 1024L
        } catch (_: Throwable) {
            runCatching { dest.delete() }
            return false
        } finally {
            runCatching { if (started) muxer?.stop() }
            runCatching { muxer?.release() }
        }
    }

    private fun open(context: Context, uri: Uri): MediaExtractor? {
        val ex = MediaExtractor()
        val pfd = context.contentResolver.openFileDescriptor(uri, "r") ?: run {
            ex.release()
            return null
        }
        return try {
            pfd.use { ex.setDataSource(it.fileDescriptor) }
            ex
        } catch (_: Throwable) {
            runCatching { ex.release() }
            null
        }
    }

    private fun videoTrack(ex: MediaExtractor): Int {
        for (i in 0 until ex.trackCount) {
            if ((ex.getTrackFormat(i).getString(MediaFormat.KEY_MIME) ?: "").startsWith("video/")) return i
        }
        return -1
    }

    private fun audioTrack(ex: MediaExtractor): Int {
        for (i in 0 until ex.trackCount) {
            if ((ex.getTrackFormat(i).getString(MediaFormat.KEY_MIME) ?: "").startsWith("audio/")) return i
        }
        return -1
    }

    private fun mimeOf(ex: MediaExtractor, track: Int): String =
        ex.getTrackFormat(track).getString(MediaFormat.KEY_MIME) ?: ""

    private fun frameDurationUs(ex: MediaExtractor, track: Int): Long {
        if (track < 0) return 33_333L
        val fmt = ex.getTrackFormat(track)
        val fps = when {
            fmt.containsKey(MediaFormat.KEY_FRAME_RATE) ->
                runCatching { fmt.getInteger(MediaFormat.KEY_FRAME_RATE).toFloat() }.getOrNull()
            else -> null
        } ?: 30f
        return (1_000_000f / fps.coerceAtLeast(1f)).toLong().coerceIn(8_000L, 100_000L)
    }

    /** Copy one track. Returns the last written presentation time. */
    private fun copyTrack(
        ex: MediaExtractor,
        track: Int,
        muxer: MediaMuxer,
        outTrack: Int,
        cursor: Long,
        buffer: ByteBuffer,
        info: MediaCodec.BufferInfo
    ): Long {
        ex.selectTrack(track)
        // Land on a keyframe so the join is decodable, same as concat demuxer.
        runCatching { ex.seekTo(0, MediaExtractor.SEEK_TO_CLOSEST_SYNC) }
        var origin = Long.MIN_VALUE
        var last = cursor
        while (true) {
            buffer.clear()
            val n = ex.readSampleData(buffer, 0)
            if (n < 0) break
            val raw = ex.sampleTime
            if (raw >= 0 && origin == Long.MIN_VALUE) origin = raw
            val rel = if (raw >= 0 && origin != Long.MIN_VALUE) (raw - origin).coerceAtLeast(0L) else 0L
            info.offset = 0
            info.size = n
            info.flags = ex.sampleFlags
            info.presentationTimeUs = cursor + rel
            last = info.presentationTimeUs
            muxer.writeSampleData(outTrack, buffer, info)
            if (!ex.advance()) break
        }
        ex.unselectTrack(track)
        return last
    }
}
