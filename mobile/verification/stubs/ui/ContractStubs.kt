@file:Suppress("unused", "UNUSED_PARAMETER")
package androidx.activity.result.contract

import android.net.Uri

abstract class ActivityResultContract<I, O>

class ActivityResultContracts {
    open class OpenDocument : ActivityResultContract<Array<String>, Uri?>()
    open class OpenDocumentTree : ActivityResultContract<Uri?, Uri?>()
    open class OpenMultipleDocuments : ActivityResultContract<Array<String>, List<@JvmSuppressWildcards Uri>>()
}
