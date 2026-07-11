package com.hinnka.mycamera.processor

import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.sin
import kotlin.math.sqrt

data class RawGlobalAlignment(
    val dxProxyPx: Float,
    val dyProxyPx: Float,
    val rotationDegrees: Float,
    val score: Float,
    val margin: Float,
    val coverage: Float,
)

object RawGlobalPreAligner {
    fun align(reference: RawAlignmentProxy, current: RawAlignmentProxy): RawGlobalAlignment? {
        if (reference.width != current.width || reference.height != current.height) return null
        if (reference.width < 32 || reference.height < 32) return null

        var best = Candidate.INVALID
        var second = Candidate.INVALID
        for (rotation in COARSE_ROTATIONS) {
            var dy = -COARSE_RADIUS
            while (dy <= COARSE_RADIUS) {
                var dx = -COARSE_RADIUS
                while (dx <= COARSE_RADIUS) {
                    val candidate = score(reference, current, dx.toFloat(), dy.toFloat(), rotation, COARSE_SAMPLE_STEP)
                    if (candidate.score < best.score) {
                        second = best
                        best = candidate
                    } else if (candidate.score < second.score) {
                        second = candidate
                    }
                    dx += COARSE_SHIFT_STEP
                }
                dy += COARSE_SHIFT_STEP
            }
        }
        if (!best.score.isFinite()) return null

        val refinedCandidates = ArrayList<Candidate>()
        for (rotationOffset in FINE_ROTATION_OFFSETS) {
            for (dyOffset in -FINE_RADIUS..FINE_RADIUS) {
                for (dxOffset in -FINE_RADIUS..FINE_RADIUS) {
                    refinedCandidates += score(
                        reference,
                        current,
                        best.dx + dxOffset,
                        best.dy + dyOffset,
                        best.rotationDegrees + rotationOffset,
                        FINE_SAMPLE_STEP,
                    )
                }
            }
        }
        val sorted = refinedCandidates.filter { it.score.isFinite() }.sortedBy { it.score }
        if (sorted.isEmpty()) return null
        best = sorted.first()
        second = sorted.firstOrNull { candidate ->
            abs(candidate.dx - best.dx) >= 1f || abs(candidate.dy - best.dy) >= 1f ||
                abs(candidate.rotationDegrees - best.rotationDegrees) >= 0.25f
        } ?: second
        return RawGlobalAlignment(
            dxProxyPx = best.dx,
            dyProxyPx = best.dy,
            rotationDegrees = best.rotationDegrees,
            score = best.score,
            margin = (second.score - best.score).coerceAtLeast(0f),
            coverage = best.coverage,
        )
    }

    private fun score(
        reference: RawAlignmentProxy,
        current: RawAlignmentProxy,
        dx: Float,
        dy: Float,
        rotationDegrees: Float,
        sampleStep: Int,
    ): Candidate {
        val width = reference.width
        val height = reference.height
        val radians = Math.toRadians(rotationDegrees.toDouble())
        val cosTheta = cos(radians).toFloat()
        val sinTheta = sin(radians).toFloat()
        val centerX = (width - 1) * 0.5f
        val centerY = (height - 1) * 0.5f
        val border = max(4, minOf(width, height) / 12)
        var sumReference2 = 0.0
        var sumCurrent2 = 0.0
        var sumCross = 0.0
        var count = 0
        var possible = 0
        var y = border
        while (y < height - border) {
            var x = border
            while (x < width - border) {
                possible++
                val centeredX = x - centerX
                val centeredY = y - centerY
                val sourceX = cosTheta * centeredX - sinTheta * centeredY + centerX + dx
                val sourceY = sinTheta * centeredX + cosTheta * centeredY + centerY + dy
                if (sourceX >= 1f && sourceY >= 1f && sourceX < width - 2f && sourceY < height - 2f) {
                    val refLow = reference.lowPass[y * width + x]
                    val curLow = bilinear(current.lowPass, width, height, sourceX, sourceY)
                    val refBand = reference.bandPass[y * width + x]
                    val curBand = bilinear(current.bandPass, width, height, sourceX, sourceY)
                    val ref = refLow + BAND_WEIGHT * refBand
                    val cur = curLow + BAND_WEIGHT * curBand
                    sumReference2 += ref * ref
                    sumCurrent2 += cur * cur
                    sumCross += ref * cur
                    count++
                }
                x += sampleStep
            }
            y += sampleStep
        }
        if (count < MIN_SAMPLES || possible == 0) return Candidate.INVALID
        val denominator = sqrt(max(sumReference2 * sumCurrent2, 1.0e-12))
        val correlation = (sumCross / denominator).coerceIn(-1.0, 1.0)
        val coverage = count.toFloat() / possible
        val cost = (1.0 - correlation).toFloat() + COVERAGE_PENALTY * (1f - coverage)
        return Candidate(dx, dy, rotationDegrees, cost, coverage)
    }

    private fun bilinear(values: FloatArray, width: Int, height: Int, x: Float, y: Float): Float {
        val x0 = x.toInt().coerceIn(0, width - 1)
        val y0 = y.toInt().coerceIn(0, height - 1)
        val x1 = (x0 + 1).coerceAtMost(width - 1)
        val y1 = (y0 + 1).coerceAtMost(height - 1)
        val fx = x - x0
        val fy = y - y0
        val a = values[y0 * width + x0] * (1f - fx) + values[y0 * width + x1] * fx
        val b = values[y1 * width + x0] * (1f - fx) + values[y1 * width + x1] * fx
        return a * (1f - fy) + b * fy
    }

    private data class Candidate(
        val dx: Float,
        val dy: Float,
        val rotationDegrees: Float,
        val score: Float,
        val coverage: Float,
    ) {
        companion object {
            val INVALID = Candidate(0f, 0f, 0f, Float.POSITIVE_INFINITY, 0f)
        }
    }

    private val COARSE_ROTATIONS = floatArrayOf(-1.0f, -0.5f, 0f, 0.5f, 1.0f)
    private val FINE_ROTATION_OFFSETS = floatArrayOf(-0.25f, 0f, 0.25f)
    private const val COARSE_RADIUS = 10
    private const val COARSE_SHIFT_STEP = 2
    private const val FINE_RADIUS = 2
    private const val COARSE_SAMPLE_STEP = 4
    private const val FINE_SAMPLE_STEP = 3
    private const val BAND_WEIGHT = 0.75f
    private const val COVERAGE_PENALTY = 0.15f
    private const val MIN_SAMPLES = 256
}
