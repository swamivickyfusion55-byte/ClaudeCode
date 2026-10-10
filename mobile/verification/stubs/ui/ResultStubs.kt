@file:Suppress("unused", "UNUSED_PARAMETER")
package androidx.activity.result

abstract class ActivityResultLauncher<I> {
    abstract fun launch(input: I)
}
