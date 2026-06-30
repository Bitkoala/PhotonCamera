package com.hinnka.mycamera.camera

import kotlin.math.log2
import kotlin.math.round
import kotlin.math.sqrt

internal object HdrBracketConfig {
    const val SIDE_EV = 2f
    const val RAW_REFERENCE_FRAME_COUNT = 4

    fun rawReferenceFrameCount(mfnrFrameCount: Int): Int {
        return RAW_REFERENCE_FRAME_COUNT + mfnrFrameCount.coerceAtLeast(0)
    }

    fun rawReferenceEv(mfnrFrameCount: Int): Float {
        return -rawReferenceBaselineEvForFrameCount(rawReferenceFrameCount(mfnrFrameCount))
    }

    fun rawReferenceEvForFrameCount(frameCount: Int): Float {
        return -rawReferenceBaselineEvForFrameCount(frameCount)
    }

    fun rawReferenceBaselineEvForFrameCount(frameCount: Int): Float {
        val safeFrameCount = frameCount.coerceAtLeast(RAW_REFERENCE_FRAME_COUNT)
        val ev = log2(sqrt(safeFrameCount.toDouble()) * 0.8)
        return (round(ev * 10.0) / 10.0).toFloat()
    }
}
