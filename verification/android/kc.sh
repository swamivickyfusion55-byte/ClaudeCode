#!/bin/bash
# kc.sh OUTDIR file.kt ...   : compile Kotlin + Compose (JetBrains Compose Multiplatform 1.7.3 jars = the same androidx.compose API surface)
D="$(cd "$(dirname "$0")" && pwd)"; OUT="$1"; shift; mkdir -p "$OUT"
CP="$(ls $D/libcmp/*.jar $D/libnet/*.jar $D/librun/*.jar 2>/dev/null | tr '\n' ':')$D/libc/kotlin-stdlib-2.0.21.jar:$D/libc/kotlinx-coroutines-core-jvm-1.9.0.jar"
java -cp "$D/libc/*" org.jetbrains.kotlin.cli.jvm.K2JVMCompiler -no-stdlib -no-reflect -jvm-target 17 -nowarn \
  -cp "$CP" -Xplugin="$D/libc/kotlin-compose-compiler-plugin-embeddable-2.0.21.jar" -d "$OUT" "$@" 2>&1 | grep -v "^Picked up"
