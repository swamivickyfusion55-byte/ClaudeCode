package com.swamitech.phoenix

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import androidx.core.app.NotificationCompat

/**
 * Keeps Phoenix running while a video is being UPLOADED (1.2.23).
 *
 * Android freezes or ends an app soon after it leaves the screen; an upload running in it stops with it.
 * While this service is in the foreground the process keeps running, and a partial wake lock keeps the CPU
 * (and so the upload) going with the screen off.
 *
 * How it is started matters, and is what went wrong in 1.2.19/1.2.20:
 *   * it is started with startService() while the app is ON SCREEN (the user just tapped Start), never from
 *     the background, and never with startForegroundService(): that call obliges the service to reach the
 *     foreground within seconds, and when Android then refused the promotion it ended the whole app with no
 *     message - the "app vanishes" of 1.2.19;
 *   * here, if Android refuses the promotion, the service simply stops itself and the upload carries on as
 *     before (and is resumable, so a pause costs one chunk);
 *   * it is stopped as soon as the last upload ends. The render runs on the Space; following a job after
 *     the upload needs nothing from this service.
 */
class PhoenixKeepAliveService : Service() {
    companion object {
        private const val CHANNEL_ID = "phoenix_processing"
        private const val NOTIFICATION_ID = 7301
        private const val WAKE_MS = 3L * 60 * 60 * 1000      // never hold the CPU longer than 3 hours
    }

    private var wakeLock: PowerManager.WakeLock? = null

    override fun onCreate() {
        super.onCreate()
        runCatching { createChannel() }
        val ok = try {
            val n = buildNotification()
            if (Build.VERSION.SDK_INT >= 29) {
                startForeground(NOTIFICATION_ID, n, ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC)
            } else {
                startForeground(NOTIFICATION_ID, n)
            }
            true
        } catch (_: Throwable) {
            false
        }
        if (!ok) {
            stopSelf()
            return
        }
        runCatching {
            val pm = getSystemService(PowerManager::class.java)
            wakeLock = pm?.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "Phoenix:upload")?.apply {
                setReferenceCounted(false)
                acquire(WAKE_MS)
            }
        }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int = START_NOT_STICKY

    override fun onTimeout(startId: Int) {
        stopQuietly()
    }

    override fun onTimeout(startId: Int, fgsType: Int) {
        stopQuietly()
    }

    override fun onDestroy() {
        runCatching { wakeLock?.let { if (it.isHeld) it.release() } }
        wakeLock = null
        super.onDestroy()
    }

    private fun stopQuietly() {
        runCatching { stopForeground(STOP_FOREGROUND_REMOVE) }
        stopSelf()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    private fun createChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val manager = getSystemService(NotificationManager::class.java)
            manager.createNotificationChannel(
                NotificationChannel(
                    CHANNEL_ID,
                    "Phoenix uploads",
                    NotificationManager.IMPORTANCE_LOW
                ).apply {
                    description = "Shown while a video is being uploaded to a Space"
                    setShowBadge(false)
                }
            )
        }
    }

    private fun buildNotification(): Notification {
        val openIntent = Intent(this, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_CLEAR_TOP
        }
        val pending = PendingIntent.getActivity(
            this, 7301, openIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.stat_sys_upload)
            .setContentTitle("Phoenix is uploading")
            .setContentText("The upload continues while the app is in the background")
            .setOngoing(true)
            .setCategory(NotificationCompat.CATEGORY_PROGRESS)
            .setContentIntent(pending)
            .setOnlyAlertOnce(true)
            .build()
    }
}
