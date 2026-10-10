@file:Suppress("unused", "UNUSED_PARAMETER")
package androidx.core.app

import android.app.Notification
import android.app.PendingIntent
import android.content.Context

class NotificationCompat {
    class Builder(context: Context, channelId: String) {
        fun setSmallIcon(icon: Int): Builder = this
        fun setContentTitle(title: CharSequence?): Builder = this
        fun setContentText(text: CharSequence?): Builder = this
        fun setOngoing(ongoing: Boolean): Builder = this
        fun setCategory(category: String?): Builder = this
        fun setContentIntent(intent: PendingIntent?): Builder = this
        fun setOnlyAlertOnce(onlyAlertOnce: Boolean): Builder = this
        fun build(): Notification = throw UnsupportedOperationException()
    }
    companion object {
        const val CATEGORY_PROGRESS = "progress"
    }
}
