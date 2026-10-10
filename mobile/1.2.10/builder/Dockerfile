FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV ANDROID_SDK_ROOT=/opt/android-sdk
ENV ANDROID_HOME=/opt/android-sdk
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV GRADLE_VERSION=8.10.2
ENV PATH=$JAVA_HOME/bin:$ANDROID_SDK_ROOT/cmdline-tools/latest/bin:$ANDROID_SDK_ROOT/platform-tools:/opt/gradle/bin:$PATH

# openjdk-17-jdk-headless (not the full -jdk) and no git/file: the full JDK
# pulls in AWT/Swing's GUI stack (GTK, X11, Cairo, Pango, Mesa, LLVM - over
# 150MB of packages this build never touches, since only javac/jar/jarsigner/
# keytool run here, all headless-safe). git and file were never actually
# invoked anywhere in build.sh/app.py. That combined package weight is what
# got this build OOM-killed (exit 137) partway through a single apt install
# layer on a memory-constrained Space build tier.
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jdk-headless wget unzip zip python3 ca-certificates \
    python3-pip && \
    rm -rf /var/lib/apt/lists/*

# The Docker Space UI is a Gradio app; install it explicitly because the base
# Ubuntu image does not provide Python packages such as gradio.
RUN python3 -m pip install --no-cache-dir --break-system-packages "gradio>=4,<6"

RUN mkdir -p $ANDROID_SDK_ROOT/cmdline-tools && \
    wget -q https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip -O /tmp/cmdline.zip && \
    unzip -q /tmp/cmdline.zip -d /tmp/cmdline && \
    mv /tmp/cmdline/cmdline-tools $ANDROID_SDK_ROOT/cmdline-tools/latest && \
    rm -rf /tmp/cmdline /tmp/cmdline.zip && \
    (yes | sdkmanager --licenses > /dev/null || true) && \
    sdkmanager "platform-tools" "platforms;android-35" "build-tools;35.0.0" "cmake;3.22.1"

RUN wget -q https://services.gradle.org/distributions/gradle-${GRADLE_VERSION}-bin.zip -O /tmp/gradle.zip && \
    unzip -q /tmp/gradle.zip -d /opt && \
    ln -s /opt/gradle-${GRADLE_VERSION} /opt/gradle && \
    rm /tmp/gradle.zip

WORKDIR /workspace

# Mobile-upload layout: all builder files are at Space root and the Android
# project is carried as source.zip. Do not use builder/ or source/ COPY paths.
COPY source.zip /tmp/source.zip
RUN mkdir -p /workspace/source && \
    unzip -q /tmp/source.zip -d /workspace/_unpack && \
    # Tolerate a zip that wraps everything in a single top-level folder.
    if [ ! -f /workspace/_unpack/settings.gradle.kts ]; then \
        inner="$(find /workspace/_unpack -maxdepth 2 -name settings.gradle.kts -printf '%h\n' | head -n 1)"; \
        test -n "$inner" && cp -a "$inner"/. /workspace/source/; \
    else \
        cp -a /workspace/_unpack/. /workspace/source/; \
    fi && \
    rm -rf /workspace/_unpack /tmp/source.zip && \
    test -f /workspace/source/tools/audit/m3_audit.py && \
    test -f /workspace/source/settings.gradle.kts

COPY build.sh /workspace/build.sh
COPY app.py /workspace/app.py
COPY README.md /workspace/README.md

RUN chmod +x /workspace/build.sh && \
    python3 /workspace/source/tools/audit/m3_audit.py /workspace/source

# Hugging Face Spaces execute the container as a non-root user (UID 1000).
# Without this block, Gradle cannot write to /workspace or to its own home
# directory and the build fails with an opaque permission error.
RUN useradd -m -u 1000 -s /bin/bash builder || true && \
    mkdir -p /workspace/.gradle /workspace/.android && \
    chown -R 1000:1000 /workspace && \
    chmod -R 777 /opt/android-sdk

ENV HOME=/home/builder
ENV GRADLE_USER_HOME=/workspace/.gradle
ENV ANDROID_USER_HOME=/workspace/.android
ENV GRADLE_OPTS="-Dorg.gradle.daemon=false -Dorg.gradle.jvmargs=-Xmx2g -Dorg.gradle.workers.max=2"

USER 1000

# No Gradle "warmup" build here anymore. It used to run a full
# :app:assembleDebug during the image build purely to pre-cache Gradle/AGP/
# Kotlin dependencies so the first real build (triggered later by "Build APK",
# which runs build.sh at RUNTIME) would be faster. On Phoenix Local's sibling
# builder this exact step was confirmed to be what was getting the build
# OOM-killed - the image-build phase appears to have a much tighter memory
# ceiling than the running container, and a cold Gradle build (JVM daemon +
# AGP's own forked compiler/dex processes) is the single most memory-hungry
# step in this pipeline. Removed here too, proactively, before it fails here
# the same way. Costs nothing functionally: build.sh performs the incremental
# "assembleDebug" build every time "Build APK" is pressed, using the persistent
# Gradle cache. The first build is still colder than a pre-warmed image build.

CMD ["python3", "/workspace/app.py"]
