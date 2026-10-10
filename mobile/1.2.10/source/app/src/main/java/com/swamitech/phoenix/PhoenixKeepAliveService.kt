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
import androidx.core.app.NotificationCompat

/**
 * Keeps the Phoenix process in the foreground while long-running remote jobs
 * are being monitored/downloaded. The actual render remains on the HF Space;
 * this service only protects the client-side polling/download coroutine from
 * Activity/background process eviction.
 */
class PhoenixKeepAliveService : Service() {
    companion object {
        private const val CHANNEL_ID = "phoenix_processing"
        private const val NOTIFICATION_ID = 7301
    }

    override fun onCreate() {
        super.onCreate()
        createChannel()
        val n = buildNotification()
        try {
            if (Build.VERSION.SDK_INT >= 34) {
                startForeground(
                    NOTIFICATION_ID, n,
                    ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC
                )
            } else {
                startForeground(NOTIFICATION_ID, n)
            }
        } catch (_: Throwable) {
            // Never take down the UI if the OS refuses a foreground service.
            stopSelf()
        }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        return START_NOT_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? = null

    private fun createChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val manager = getSystemService(NotificationManager::class.java)
            manager.createNotificationChannel(
                NotificationChannel(
                    CHANNEL_ID,
                    "Phoenix processing",
                    NotificationManager.IMPORTANCE_LOW
                ).apply {
                    description = "Keeps long-running Phoenix video jobs active in the background"
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
            .setSmallIcon(android.R.drawable.stat_notify_sync)
            .setContentTitle("Phoenix processing")
            .setContentText("Video processing will continue if you leave the app")
            .setOngoing(true)
            .setCategory(NotificationCompat.CATEGORY_PROGRESS)
            .setContentIntent(pending)
            .setOnlyAlertOnce(true)
            .build()
    }
}
