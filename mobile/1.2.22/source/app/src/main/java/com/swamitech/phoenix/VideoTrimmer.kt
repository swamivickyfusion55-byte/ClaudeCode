package com.swamitech.phoenix

import android.content.Context
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaMetadataRetriever
import android.media.MediaMuxer
import android.net.Uri
import java.io.File
import java.nio.ByteBuffer

/**
 * Cuts a video down to the selected range BEFORE it is uploaded.
 *
 * Previously the entire source file went to the Space and the server threw
 * away everything outside the trim range - so trimming a 4-minute clip down
 * to 10 seconds still uploaded all 4 minutes. On a phone connection that is
 * the slowest part of the whole job, and it is the exact transfer that has
 * been dropping mid-flight when several Spaces are started together.
 *
 * This is a remux, not a re-encode: compressed samples are copied straight
 * through MediaMuxer. It takes seconds, uses little CPU, and cannot degrade
 * quality because nothing is ever decoded.
 *
 * The one real constraint is that a video track can only START at a sync
 * (key) frame - cutting anywhere else leaves the opening frames
 * undecodable. So the cut lands on the sync frame at or BEFORE the requested
 * start, and [Result.residualStartUs] reports the extra footage that had to
 * be kept. The caller passes that residue to the server as a small trim on
 * the now far smaller file, keeping the final output frame-exact while still
 * saving nearly all of the upload.
 */
object VideoTrimmer {

    data class Result(
        val file: File,
        /** Extra leading footage kept because the cut had to land on a sync frame. */
        val residualStartUs: Long,
        /** Duration of the produced file. */
        val durationUs: Long
    )

    fun durationUs(context: Context, uri: Uri): Long? {
        val mmr = MediaMetadataRetriever()
        return try {
            mmr.setDataSource(context, uri)
            mmr.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                ?.toLongOrNull()
                ?.times(1000L)
        } catch (t: Throwable) {
            null
        } finally {
            runCatching { mmr.release() }
        }
    }

    /**
     * Remuxes [uri] between [startUs] and [endUs] into [dest].
     * Returns null on ANY failure so the caller falls back to uploading the
     * original file unchanged - a trim that cannot be done must never fail
     * the job.
     */
    fun trim(context: Context, uri: Uri, startUs: Long, endUs: Long, dest: File): Result? {
        var extractorRef: MediaExtractor? = null
        var muxerRef: MediaMuxer? = null
        var started = false
        try {
            val extractor = MediaExtractor()
            extractorRef = extractor
            val pfd = context.contentResolver.openFileDescriptor(uri, "r") ?: return null
            pfd.use { extractor.setDataSource(it.fileDescriptor) }

            val muxer = MediaMuxer(dest.absolutePath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
            muxerRef = muxer

            val indexMap = HashMap<Int, Int>()
            var maxInputSize = 0
            var hasVideo = false

            for (i in 0 until extractor.trackCount) {
                val format = extractor.getTrackFormat(i)
                val mime = format.getString(MediaFormat.KEY_MIME) ?: continue
                if (!mime.startsWith("video/") && !mime.startsWith("audio/")) continue
                extractor.selectTrack(i)
                indexMap[i] = muxer.addTrack(format)
                if (format.containsKey(MediaFormat.KEY_MAX_INPUT_SIZE)) {
                    maxInputSize = maxOf(maxInputSize, format.getInteger(MediaFormat.KEY_MAX_INPUT_SIZE))
                }
                if (mime.startsWith("video/")) hasVideo = true
            }
            if (!hasVideo || indexMap.isEmpty()) return null
            if (maxInputSize <= 0) maxInputSize = 2 * 1024 * 1024

            // Land on the sync frame at or before the requested start.
            extractor.seekTo(startUs, MediaExtractor.SEEK_TO_PREVIOUS_SYNC)
            val firstSample = extractor.sampleTime
            val actualStartUs = if (firstSample < 0L) 0L else firstSample

            muxer.start()
            started = true

            val buffer = ByteBuffer.allocate(maxInputSize)
            val info = MediaCodec.BufferInfo()
            var wroteAny = false
            var lastPtsUs = 0L

            while (true) {
                val sampleTime = extractor.sampleTime
                if (sampleTime < 0L) break
                if (sampleTime > endUs) break
                val dstTrack = indexMap[extractor.sampleTrackIndex]
                if (dstTrack != null) {
                    buffer.clear()
                    val size = extractor.readSampleData(buffer, 0)
                    if (size < 0) break
                    info.offset = 0
                    info.size = size
                    info.presentationTimeUs = (sampleTime - actualStartUs).coerceAtLeast(0L)
                    info.flags = if (extractor.sampleFlags and MediaExtractor.SAMPLE_FLAG_SYNC != 0) {
                        MediaCodec.BUFFER_FLAG_KEY_FRAME
                    } else {
                        0
                    }
                    muxer.writeSampleData(dstTrack, buffer, info)
                    wroteAny = true
                    if (info.presentationTimeUs > lastPtsUs) lastPtsUs = info.presentationTimeUs
                }
                if (!extractor.advance()) break
            }

            if (!wroteAny) return null
            muxer.stop()
            started = false

            if (!dest.exists() || dest.length() <= 0L) return null
            return Result(
                file = dest,
                residualStartUs = (startUs - actualStartUs).coerceAtLeast(0L),
                durationUs = lastPtsUs
            )
        } catch (t: Throwable) {
            runCatching { if (started) muxerRef?.stop() }
            runCatching { dest.delete() }
            return null
        } finally {
            runCatching { muxerRef?.release() }
            runCatching { extractorRef?.release() }
        }
    }
}
