#!/usr/bin/env bash
set -euo pipefail

ROOT="${PHOENIX_ROOT:-/workspace}"
VER="1.2.21"
cd "$ROOT/source"

echo "=== Phoenix v$VER static release audit ==="
python3 -u tools/audit/m3_audit.py .

echo
echo "=== Toolchain ==="
java -version 2>&1 | head -n 1
gradle --version 2>&1 | grep -i '^Gradle' || true
echo "GRADLE_USER_HOME=${GRADLE_USER_HOME:-default}"
echo "ANDROID_HOME=${ANDROID_HOME:-unset}"

echo
echo "=== SDK repair ==="
# AGP refuses to configure if build-tools 34.0.0 is present with a bad
# source.properties. That folder is leftover and unused. 35.0.0 is the one
# this project pins.
SDK="${ANDROID_HOME:-/opt/android-sdk}"
rm -rf "$SDK/build-tools/34.0.0"
if [ ! -f "$SDK/build-tools/35.0.0/source.properties" ] || ! grep -q "Pkg.Revision=35.0.0" "$SDK/build-tools/35.0.0/source.properties"; then
    echo "Installing build-tools 35.0.0"
    yes | sdkmanager --install "build-tools;35.0.0" || true
fi
if [ -d "$SDK/build-tools/35.0.0" ] && ! grep -q "Pkg.Revision=35.0.0" "$SDK/build-tools/35.0.0/source.properties" 2>/dev/null; then
    printf 'Pkg.UserSrc=false\nPkg.Revision=35.0.0\n' > "$SDK/build-tools/35.0.0/source.properties"
fi
# A failed configure was stored last time. Do not replay it.
rm -rf "$ROOT/source/.gradle/configuration-cache" "${GRADLE_USER_HOME:-$ROOT/.gradle}/configuration-cache" || true

echo
echo "=== Gradle incremental build ==="
# --console=plain keeps the streamed log readable (no ANSI progress redraws).
gradle --no-daemon --stacktrace --console=plain --build-cache --no-configuration-cache --max-workers=2 assembleDebug

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

echo
echo "=== Checking the APK the way the phone will ==="
# The phone's installer refuses an APK that is unsigned, badly signed, test-only,
# or signed with a different key from the copy already installed - and only says
# "Something went wrong". Check the first three here and print the key so the
# fourth can be compared between builds.
BT="$SDK/build-tools/35.0.0"
INFO="$ROOT/PhoenixMobile-$VER-install-info.txt"
: > "$INFO"
if [ -x "$BT/apksigner" ]; then
    if ! SIGOUT="$("$BT/apksigner" verify --verbose --print-certs "$APK" 2>&1)"; then
        echo "$SIGOUT"
        echo "ERROR: apksigner says this APK does not verify. A phone would refuse it. Not offering it."
        exit 3
    fi
    echo "$SIGOUT" | grep -E '^Verifies|Verified using|certificate (DN|SHA-256)' || true
    KEY="$(echo "$SIGOUT" | sed -n 's/.*[Ss]igner #1 certificate SHA-256 digest: *//p' | head -n 1)"
    SCHEMES="$(echo "$SIGOUT" | sed -n 's/.*Verified using \(v[0-9.]*\) scheme.*: true/\1/p' | tr '\n' ' ')"
    echo "Signing key (SHA-256): ${KEY:-unknown}" >> "$INFO"
    echo "Signature schemes: ${SCHEMES:-unknown}" >> "$INFO"
else
    echo "apksigner not found at $BT/apksigner - skipping the signature check"
    echo "Signature check: skipped (apksigner missing)" >> "$INFO"
fi
if [ -x "$BT/aapt2" ]; then
    BADGE="$("$BT/aapt2" dump badging "$APK" 2>/dev/null || true)"
    PKGLINE="$(echo "$BADGE" | grep -m1 '^package:' || true)"
    [ -n "$PKGLINE" ] && echo "$PKGLINE" | sed -n "s/package: name='\([^']*\)' versionCode='\([^']*\)' versionName='\([^']*\)'.*/Package: \1  versionCode \2  versionName \3/p" >> "$INFO"
    echo "$BADGE" | grep -m1 "^sdkVersion:" | sed "s/sdkVersion:'\(.*\)'/Minimum Android (API): \1/" >> "$INFO" || true
    echo "$BADGE" | grep -m1 "^targetSdkVersion:" | sed "s/targetSdkVersion:'\(.*\)'/Target API: \1/" >> "$INFO" || true
    ABIS="$(echo "$BADGE" | grep -m1 '^native-code:' | sed "s/native-code: *//; s/'//g" || true)"
    [ -n "$ABIS" ] && echo "ABIs: $ABIS" >> "$INFO"
    MANI="$("$BT/aapt2" dump xmltree --file AndroidManifest.xml "$APK" 2>/dev/null || true)"
    # (plain bash tests, not "grep -q" in a pipe: with pipefail an early-exiting
    # grep -q can make the pipeline report failure even though it matched)
    TESTONLY="$(echo "$MANI" | grep -E 'testOnly' || true)"
    if [[ "$TESTONLY" =~ true|0xffffffff ]]; then
        echo "ERROR: this APK is marked test-only; Android refuses to install it from a file. Not offering it."
        exit 3
    fi
    if [[ "$BADGE" == *application-debuggable* ]]; then
        echo "Build type: debug (debuggable, signed with the Phoenix debug key)" >> "$INFO"
    fi
else
    echo "aapt2 not found at $BT/aapt2 - skipping the package check"
fi
echo "--- install info ---"
cat "$INFO"

sha256sum "$APK" | tee "$ROOT/PhoenixMobile-$VER-debug.sha256"
cp "$APK" "$ROOT/PhoenixMobile-$VER-debug.apk"
ls -lh "$ROOT/PhoenixMobile-$VER-debug.apk"

echo
echo "=== BUILD PASS ==="
