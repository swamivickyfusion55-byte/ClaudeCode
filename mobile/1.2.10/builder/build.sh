#!/usr/bin/env bash
set -euo pipefail

ROOT="${PHOENIX_ROOT:-/workspace}"
cd "$ROOT/source"

echo "=== Phoenix v1.2.10 static release audit ==="
python3 -u tools/audit/m3_audit.py .

echo
echo "=== Toolchain ==="
java -version 2>&1 | head -n 1
gradle --version 2>&1 | grep -i '^Gradle' || true
echo "GRADLE_USER_HOME=${GRADLE_USER_HOME:-default}"
echo "ANDROID_HOME=${ANDROID_HOME:-unset}"

echo
echo "=== Gradle incremental build ==="
# --console=plain keeps the streamed log readable (no ANSI progress redraws).
gradle --no-daemon --stacktrace --console=plain --build-cache --configuration-cache --max-workers=2 assembleDebug

echo
echo "=== Collecting artefact ==="
# NOTE: a "find ... | head -n1" pipeline trips `set -o pipefail` when find is
# killed by SIGPIPE, which used to fail builds that had actually succeeded.
mapfile -t APKS < <(find app/build/outputs/apk -type f -name '*.apk' | sort)
if [ "${#APKS[@]}" -eq 0 ]; then
    echo "ERROR: assembleDebug completed but no .apk was found under app/build/outputs/apk"
    exit 2
fi
APK="${APKS[0]}"
echo "APK=$APK"

sha256sum "$APK" | tee "$ROOT/PhoenixMobile-1.2.10-debug.sha256"
cp "$APK" "$ROOT/PhoenixMobile-1.2.10-debug.apk"
ls -lh "$ROOT/PhoenixMobile-1.2.10-debug.apk"

echo
echo "=== BUILD PASS ==="
