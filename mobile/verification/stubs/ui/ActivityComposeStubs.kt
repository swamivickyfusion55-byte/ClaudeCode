@file:Suppress("unused", "UNUSED_PARAMETER")
package androidx.activity.compose

import androidx.activity.ComponentActivity
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContract
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionContext

fun ComponentActivity.setContent(parent: CompositionContext? = null, content: @Composable () -> Unit) {}

class ManagedActivityResultLauncher<I, O> : ActivityResultLauncher<I>() {
    override fun launch(input: I) {}
}

@Composable
fun <I, O> rememberLauncherForActivityResult(
    contract: ActivityResultContract<I, O>,
    onResult: (O) -> Unit
): ManagedActivityResultLauncher<I, O> = ManagedActivityResultLauncher()
