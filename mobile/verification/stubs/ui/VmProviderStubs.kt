// Compile-only stand-in for androidx.lifecycle.ViewModelProvider (used by MainActivity.onStart/onStop since 1.2.21).
package androidx.lifecycle

class ViewModelProvider(owner: Any) {
    operator fun <T : ViewModel> get(modelClass: Class<T>): T = error("stub")
}
