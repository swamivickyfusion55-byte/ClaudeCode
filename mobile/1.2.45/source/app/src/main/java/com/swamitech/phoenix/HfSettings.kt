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
    val enhancer: String = "GPEN + Cinematic",
    val swapEvery: String = "1",
    val detectEvery: String = "1",
    val detectInterval: Int = 4,
    val faceMode: String = "2 faces",
    val enhanceScope: String = "All faces",
    val deviceMode: String = "CPU only",
    val primarySlot: String = "Auto",
    val smoothMotion: String = "Fast (blend)",
    // 1.2.30: lower-belly clear skin (Space v11.2.110+): every tattoo or mark from below the navel to the waist line removed, skin evened
    val bellyClear: Boolean = false,
    // 1.2.31: the band's length and breadth (Space v11.2.111+), x0.5 - x1.5
    val bellyLength: Float = 1f,
    val bellyWidth: Float = 1f,
    // 1.2.45 (Space v11.2.142+): the belly pentagon in body coordinates ([u, t] x 5, JSON) as the last Preview belly returned it
    val bellyPentagon: String = "",
    // ...and the marks that preview found, each with its place on the body and whether it was taken ([u, t, 1|0, px] x n, JSON): the job
    // takes exactly those (a mark on the pentagon's line gets the preview's decision, not a new one)
    val bellyMarks: String = "",
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
