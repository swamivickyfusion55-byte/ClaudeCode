package com.swamitech.phoenix

data class HfSettings(
    val spaceUrl: String = "",
    val endpoint: String = "",
    val modelRepo: String = "",
    val modelRevision: String = "main"
)

data class PhoenixOptions(
    val durationSec: Int = 360,
    val fps: Int = 30,
    val resolution: String = "720p (HD)",
    val quality: String = "Ultra",
    val enhancer: String = "Cinematic (clarity + smooth)",
    val swapEvery: String = "1",
    val detectEvery: String = "1",
    val detectInterval: Int = 4,
    val faceMode: String = "2 faces",
    val enhanceScope: String = "All faces",
    val deviceMode: String = "CPU only",
    val primarySlot: String = "Auto",
    val smoothMotion: String = "Fast (blend)",
    val trimStart: Int = 0,
    val trimEnd: Int = 100,
    val password: String = "",
    val splitParts: String = "Off",
    val splitSharedSettings: Boolean = true,
    val splitFaceMode1: String = "2 faces",
    val splitFaceMode2: String = "2 faces",
    val splitFaceMode3: String = "2 faces",
    val splitFaceMode4: String = "2 faces",
    val splitFaceMode5: String = "2 faces",
    val splitFaceMode6: String = "2 faces",
    val splitFaceMode7: String = "2 faces",
    val splitFaceMode8: String = "2 faces",
    val splitFaceMode9: String = "2 faces",
    val splitFaceMode10: String = "2 faces",
    val autoMergeSplits: Boolean = true
)
