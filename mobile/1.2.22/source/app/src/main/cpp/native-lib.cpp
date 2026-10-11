#include <jni.h>
#include <string>

namespace {
constexpr int PHOENIX_ENGINE_NOT_READY = 1001;
constexpr int PHOENIX_INVALID_ARGUMENT = 1002;
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_swamitech_phoenix_PhoenixEngine_nativeBackendInfo(JNIEnv* env, jobject) {
    const std::string info =
        "Phoenix M3 native boundary: CPU/NNAPI/XNNPACK-ready; licensed swap graph required";
    return env->NewStringUTF(info.c_str());
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_swamitech_phoenix_PhoenixEngine_nativeProcessVideo(
        JNIEnv* env, jobject, jstring inputPath, jstring outputPath,
        jstring mode, jobject callback) {
    // M3 deliberately refuses to claim processing until the validated, licensed
    // detector/recognizer/landmark/swapper graph is installed. Returning an explicit
    // status is safer than producing an apparently successful but incorrect video.
    if (!inputPath || !outputPath || !mode) return PHOENIX_INVALID_ARGUMENT;
    (void)callback;
    return PHOENIX_ENGINE_NOT_READY;
}
