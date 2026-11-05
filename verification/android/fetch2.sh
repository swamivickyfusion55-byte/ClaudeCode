#!/bin/bash
cd "$(dirname "$0")"
mkdir -p libc libcmp
try() { for i in 1 2 3 4 5; do "$@" >> mvn2.log 2>&1 && return 0; sleep $((i*10)); done; return 1; }
try /opt/maven/bin/mvn -q -B -f pom_compiler.xml dependency:copy-dependencies -DoutputDirectory=libc || echo "COMPILER FAILED" >> mvn2.log
V=1.7.3
for a in \
  org.jetbrains.compose.material3:material3-desktop:$V \
  org.jetbrains.compose.material:material-ripple-desktop:$V \
  org.jetbrains.compose.foundation:foundation-desktop:$V \
  org.jetbrains.compose.foundation:foundation-layout-desktop:$V \
  org.jetbrains.compose.ui:ui-desktop:$V \
  org.jetbrains.compose.ui:ui-graphics-desktop:$V \
  org.jetbrains.compose.ui:ui-text-desktop:$V \
  org.jetbrains.compose.ui:ui-unit-desktop:$V \
  org.jetbrains.compose.ui:ui-geometry-desktop:$V \
  org.jetbrains.compose.ui:ui-util-desktop:$V \
  org.jetbrains.compose.runtime:runtime-desktop:$V \
  org.jetbrains.compose.runtime:runtime-saveable-desktop:$V \
  org.jetbrains.compose.animation:animation-desktop:$V \
  org.jetbrains.compose.animation:animation-core-desktop:$V \
  org.jetbrains.compose.annotation-internal:annotation:$V \
  org.jetbrains.compose.collection-internal:collection:$V ; do
  try /opt/maven/bin/mvn -q -B -f pom_compiler.xml dependency:copy -Dartifact=$a -DoutputDirectory=libcmp || echo "MISSING $a" >> mvn2.log
done
echo done > fetch2.done
