# Kotlin / Compose compile checks without the Android SDK

The sandbox has no Android SDK and cannot reach Google's Maven (dl.google.com). Everything here uses Maven Central only:
Kotlin 2.0.21 (the project's version) with its Compose compiler plugin, and JetBrains Compose Multiplatform 1.7.3 jars, which carry the
same `androidx.compose.*` API surface as the project's Compose 1.7.8 / Material 3 1.3.1.

1. `fetch2.sh` downloads the compiler and jars into `libc/`, `libcmp/` (add OkHttp 4.12, okio-jvm 3.6, org.json:json 20231013 into
   `libnet/` the same way, with `dependency:copy -Dartifact=...`). `pom_compiler.xml` is the helper POM.
2. `kc.sh OUTDIR file.kt ...` compiles with the Compose plugin. A deliberate mistake (a wrong parameter name) is reported, so it is a real check.
3. `build_harness.py SRC OUTDIR` builds a harness from the REAL text of the restyled composables in `MainActivity.kt` / `PatchRemoval.kt`
   (palette, 13 composables, the belly card, the theme and root layout), with stubs only for Android-only types, then `kc.sh out OUTDIR/*.kt`.
4. `build_vm_harness.py PhoenixViewModel.kt OUTDIR` lifts the real `sendVideo`, `resumableToken` and `hasInternet` into a stand-in class;
   compile it together with the real `net/HfApi.kt`, `HfSettings.kt` and small Android stubs (ContentResolver, Uri, Base64, ...).
5. `diffcompile.py OLD.kt NEW.kt` compiles two versions of a whole file and prints only the errors that are new (a file that needs the Android
   libraries still has hundreds of "unresolved reference" errors; a mistake in new code shows up as a new, different one).

What it cannot do: run the app, a Gradle build, or draw the screens (the renderer needs `androidx.collection` classes that only Google's
Maven has). The APK Builder remains the real build.
