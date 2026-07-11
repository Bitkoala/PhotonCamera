package com.hinnka.mycamera.processor

import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max

data class RawBurstReferencePlan(
    val referenceOriginalIndex: Int,
    val orderedIndices: IntArray,
    val alignmentsToReference: List<RawGlobalAlignment?>,
    val preAlignmentsToReference: List<RawFramePreAlignment?>,
    val referenceCost: Float,
)

object RawBurstReferencePlanner {
    fun plan(
        frames: List<RawStackFrame>,
        proxyConfig: RawAlignmentProxyConfig,
    ): RawBurstReferencePlan {
        val proxies = frames.map { RawAlignmentProxyBuilder.build(it.image, proxyConfig) }
        val gyroRisks = RawBurstGyroSelector.select(frames).blurRiskByIndex
        return planProxies(proxies, gyroRisks, frames.map { it.sensorTimestampNs })
    }

    internal fun planProxies(
        proxies: List<RawAlignmentProxy?>,
        gyroRisks: FloatArray = FloatArray(proxies.size) { Float.NaN },
        sensorTimestampsNs: List<Long> = proxies.indices.map(Int::toLong),
    ): RawBurstReferencePlan {
        if (proxies.isEmpty()) {
            return RawBurstReferencePlan(-1, IntArray(0), emptyList(), emptyList(), Float.POSITIVE_INFINITY)
        }
        val valid = proxies.indices.filter { proxies[it] != null }
        if (valid.size < 2) return fallback(proxies.size, valid.firstOrNull() ?: 0)

        val candidates = referenceCandidates(proxies, gyroRisks, valid)
        var bestReference = candidates.first()
        var bestCost = Float.POSITIVE_INFINITY
        var bestAlignments: List<RawGlobalAlignment?> = List(proxies.size) { null }
        val maxSharpness = valid.maxOf { proxies[it]!!.sharpness }.coerceAtLeast(1.0e-6f)
        val medianGyro = gyroRisks
            .filter(Float::isFinite)
            .sorted()
            .let { values ->
                if (values.isEmpty()) Float.NaN else values[values.size / 2].coerceAtLeast(1.0e-6f)
            }

        for (candidate in candidates) {
            val reference = proxies[candidate] ?: continue
            val alignments = MutableList<RawGlobalAlignment?>(proxies.size) { null }
            alignments[candidate] = RawGlobalAlignment(0f, 0f, 0f, 0f, 1f, 1f)
            var geometryCost = 0f
            var failed = 0
            for (index in valid) {
                if (index == candidate) continue
                val alignment = RawGlobalPreAligner.align(reference, proxies[index]!!)
                alignments[index] = alignment
                if (alignment == null) {
                    failed++
                    geometryCost += FAILED_ALIGNMENT_COST
                    continue
                }
                geometryCost += GEOMETRY_TRANSLATION_WEIGHT * hypot(
                    alignment.dxProxyPx / reference.width,
                    alignment.dyProxyPx / reference.height,
                )
                geometryCost += GEOMETRY_ROTATION_WEIGHT * abs(alignment.rotationDegrees)
                geometryCost += ALIGNMENT_SCORE_WEIGHT * alignment.score
                geometryCost += COVERAGE_WEIGHT * (1f - alignment.coverage)
                geometryCost += MARGIN_WEIGHT / max(alignment.margin, MIN_MARGIN)
            }
            val sharpnessPenalty = SHARPNESS_WEIGHT *
                (1f - reference.sharpness / maxSharpness).coerceIn(0f, 1f)
            val gyroPenalty = if (medianGyro.isFinite() && gyroRisks.getOrNull(candidate)?.isFinite() == true) {
                GYRO_WEIGHT * (gyroRisks[candidate] / medianGyro).coerceIn(0f, 4f)
            } else {
                UNKNOWN_GYRO_PENALTY
            }
            val cost = geometryCost / max(1, valid.size - 1) + sharpnessPenalty + gyroPenalty +
                failed * FAILED_REFERENCE_PENALTY
            if (cost < bestCost) {
                bestCost = cost
                bestReference = candidate
                bestAlignments = alignments
            }
        }

        val remainder = proxies.indices
            .asSequence()
            .filter { it != bestReference }
            .sortedWith(
                compareBy<Int> { bestAlignments[it] == null }
                    .thenBy { index ->
                        bestAlignments[index]?.let { hypot(it.dxProxyPx, it.dyProxyPx) }
                            ?: Float.POSITIVE_INFINITY
                    }
                    .thenBy { it }
            )
            .toList()
        val directPreAlignments = bestAlignments.mapIndexed { index, alignment ->
            val proxy = proxies[index]
            if (index == bestReference) {
                RawFramePreAlignment.Identity
            } else if (alignment != null && proxy != null) {
                alignment.toFramePreAlignment(proxy)
            } else {
                null
            }
        }
        val graphPreAlignments = solveGlobalTemporalGraph(
            proxies = proxies,
            sensorTimestampsNs = sensorTimestampsNs,
            referenceIndex = bestReference,
            directAlignments = bestAlignments,
            directPreAlignments = directPreAlignments,
        )
        return RawBurstReferencePlan(
            referenceOriginalIndex = bestReference,
            orderedIndices = intArrayOf(bestReference, *remainder.toIntArray()),
            alignmentsToReference = bestAlignments,
            preAlignmentsToReference = graphPreAlignments ?: directPreAlignments,
            referenceCost = bestCost,
        )
    }

    private fun referenceCandidates(
        proxies: List<RawAlignmentProxy?>,
        gyroRisks: FloatArray,
        valid: List<Int>,
    ): List<Int> {
        val candidates = LinkedHashSet<Int>()
        candidates += valid[valid.size / 2]
        valid.sortedByDescending { proxies[it]!!.sharpness }.take(2).forEach(candidates::add)
        valid.asSequence()
            .filter { gyroRisks.getOrNull(it)?.isFinite() == true }
            .sortedBy { gyroRisks[it] }
            .take(2)
            .forEach(candidates::add)
        if (candidates.size < MAX_REFERENCE_CANDIDATES) {
            candidates += valid.first()
            candidates += valid.last()
        }
        return candidates.take(MAX_REFERENCE_CANDIDATES)
    }

    private fun fallback(frameCount: Int, reference: Int): RawBurstReferencePlan {
        val remainder = (0 until frameCount).filter { it != reference }
        return RawBurstReferencePlan(
            referenceOriginalIndex = reference,
            orderedIndices = intArrayOf(reference, *remainder.toIntArray()),
            alignmentsToReference = List(frameCount) { null },
            preAlignmentsToReference = List(frameCount) { index ->
                if (index == reference) RawFramePreAlignment.Identity else null
            },
            referenceCost = Float.POSITIVE_INFINITY,
        )
    }

    private fun RawGlobalAlignment.toFramePreAlignment(proxy: RawAlignmentProxy): RawFramePreAlignment {
        return RawFramePreAlignment(
            translationXPlanePx = dxProxyPx * proxy.sourcePlaneWidth / proxy.width.coerceAtLeast(1),
            translationYPlanePx = dyProxyPx * proxy.sourcePlaneHeight / proxy.height.coerceAtLeast(1),
            rotationDegrees = rotationDegrees,
            confidence = confidence(),
        )
    }

    private fun RawGlobalAlignment.confidence(): Float {
        val scoreConfidence = (1f - score / MAX_CONFIDENT_SCORE).coerceIn(0f, 1f)
        val marginConfidence = (margin / FULL_CONFIDENCE_MARGIN).coerceIn(0f, 1f)
        return coverage.coerceIn(0f, 1f) * scoreConfidence * marginConfidence
    }

    private fun solveGlobalTemporalGraph(
        proxies: List<RawAlignmentProxy?>,
        sensorTimestampsNs: List<Long>,
        referenceIndex: Int,
        directAlignments: List<RawGlobalAlignment?>,
        directPreAlignments: List<RawFramePreAlignment?>,
    ): List<RawFramePreAlignment?>? {
        if (sensorTimestampsNs.size != proxies.size || proxies.size < 3) return null
        val graph = RawTemporalAlignmentGraphBuilder.build(
            sensorTimestampsNs = sensorTimestampsNs,
            referenceFrameIndex = referenceIndex,
            includeSkipOne = proxies.size <= MAX_SKIP_ONE_FRAME_COUNT,
        )
        val observations = graph.edges.mapNotNull { edge ->
            val fromProxy = proxies[edge.fromFrameIndex] ?: return@mapNotNull null
            val toProxy = proxies[edge.toFrameIndex] ?: return@mapNotNull null
            val alignment = if (
                edge.type == RawTemporalEdgeType.REFERENCE && edge.fromFrameIndex == referenceIndex
            ) {
                directAlignments[edge.toFrameIndex]
            } else {
                RawGlobalPreAligner.align(fromProxy, toProxy)
            } ?: return@mapNotNull null
            val preAlignment = alignment.toFramePreAlignment(fromProxy)
            if (preAlignment.confidence < MIN_GRAPH_EDGE_CONFIDENCE) return@mapNotNull null
            RawTemporalEdgeObservation(
                edge = edge,
                dxPlanePx = preAlignment.translationXPlanePx,
                dyPlanePx = preAlignment.translationYPlanePx,
                confidence = preAlignment.confidence,
            )
        }
        val solution = RawTemporalAlignmentSolver.solve(graph, observations) ?: return null
        if (solution.acceptedObservationIndices.size < proxies.size - 1) return null
        val accepted = solution.acceptedObservationIndices.toSet()
        return List(proxies.size) { frameIndex ->
            if (frameIndex == referenceIndex) return@List RawFramePreAlignment.Identity
            val incidentConfidence = observations.indices.asSequence()
                .filter { it in accepted }
                .map { observations[it] }
                .filter { observation ->
                    observation.edge.fromFrameIndex == frameIndex || observation.edge.toFrameIndex == frameIndex
                }
                .map { it.confidence }
                .toList()
                .average()
                .takeIf(Double::isFinite)
                ?.toFloat()
                ?: return@List directPreAlignments[frameIndex]
            val position = solution.positions[frameIndex]
            RawFramePreAlignment(
                translationXPlanePx = position.xPlanePx,
                translationYPlanePx = position.yPlanePx,
                rotationDegrees = directPreAlignments[frameIndex]?.rotationDegrees ?: 0f,
                confidence = incidentConfidence,
            )
        }
    }

    private const val MAX_REFERENCE_CANDIDATES = 4
    private const val GEOMETRY_TRANSLATION_WEIGHT = 8f
    private const val GEOMETRY_ROTATION_WEIGHT = 0.08f
    private const val ALIGNMENT_SCORE_WEIGHT = 0.7f
    private const val COVERAGE_WEIGHT = 0.3f
    private const val MARGIN_WEIGHT = 0.0005f
    private const val MIN_MARGIN = 0.002f
    private const val SHARPNESS_WEIGHT = 0.4f
    private const val GYRO_WEIGHT = 0.15f
    private const val UNKNOWN_GYRO_PENALTY = 0.08f
    private const val FAILED_ALIGNMENT_COST = 2f
    private const val FAILED_REFERENCE_PENALTY = 0.5f
    private const val MAX_CONFIDENT_SCORE = 0.8f
    private const val FULL_CONFIDENCE_MARGIN = 0.02f
    private const val MIN_GRAPH_EDGE_CONFIDENCE = 0.02f
    private const val MAX_SKIP_ONE_FRAME_COUNT = 8
}
