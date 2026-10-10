// Compile-only stand-ins for the AndroidX classes that live solely on Google's
// Maven (blocked here). Signatures match the real ones for what the app uses.
@file:Suppress("unused", "UNUSED_PARAMETER")
package androidx.lifecycle

import android.app.Application
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob

abstract class ViewModel {
    protected open fun onCleared() {}
}

open class AndroidViewModel(private val application: Application) : ViewModel() {
    @Suppress("UNCHECKED_CAST")
    fun <T : Application> getApplication(): T = application as T
}

val ViewModel.viewModelScope: CoroutineScope
    get() = CoroutineScope(SupervisorJob() + Dispatchers.Main)
