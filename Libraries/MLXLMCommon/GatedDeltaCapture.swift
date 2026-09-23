// Copyright © 2026 Apple Inc.

import Foundation
import MLX

/// What one gated-delta layer consumed during a speculative verification pass.
///
/// The pass computes every candidate position in one batched forward, but only
/// a prefix of those positions is finally accepted. A recurrent state cannot be
/// trimmed by dropping rows, so the layer keeps the projections that produced
/// the pass and replays them once the accepted count is known. The replay
/// carries no model weights; it only re-runs the elementwise gated-delta scan.
public struct GatedDeltaCapture {
    /// `concat([convState, qkv])`: the conv input, `[1, K - 1 + S, convDim]`.
    public var convInput: MLXArray
    /// Post-norm, post-scale `q`/`k` (`[1, S, Hk, Dk]`) and `v` (`[1, S, Hv, Dv]`).
    public var q: MLXArray
    public var k: MLXArray
    public var v: MLXArray
    /// Raw gate projections, `[1, S, Hv]`.
    public var a: MLXArray
    public var b: MLXArray
    public var aLog: MLXArray
    public var dtBias: MLXArray
    /// Recurrent state before the pass.
    public var initialState: MLXArray

    public init(
        convInput: MLXArray,
        q: MLXArray,
        k: MLXArray,
        v: MLXArray,
        a: MLXArray,
        b: MLXArray,
        aLog: MLXArray,
        dtBias: MLXArray,
        initialState: MLXArray
    ) {
        self.convInput = convInput
        self.q = q
        self.k = k
        self.v = v
        self.a = a
        self.b = b
        self.aLog = aLog
        self.dtBias = dtBias
        self.initialState = initialState
    }

    /// The layer's state after the first `validCount` positions of the pass.
    ///
    /// Positions past `validCount` are masked out, so one trace serves every
    /// possible accepted prefix. `validCount` may be a lazy int32 array.
    public func replay(validCount: MLXArray) -> (recurrent: MLXArray, conv: MLXArray) {
        let outputs = compiledGatedDeltaReplay([
            q, k, v, a, b, aLog, dtBias, initialState, convInput, validCount,
        ])
        return (outputs[0], outputs[1])
    }
}

/// One trace serves every layer: the shapes match across a model's gated-delta
/// layers, and elementwise work around the scan fuses into a few launches.
private let compiledGatedDeltaReplay: @Sendable ([MLXArray]) -> [MLXArray] = compile { inputs in
    let (q, k, v) = (inputs[0], inputs[1], inputs[2])
    let (a, b, aLog, dtBias) = (inputs[3], inputs[4], inputs[5], inputs[6])
    let (state, convInput, validCount) = (inputs[7], inputs[8], inputs[9])
    let s = q.dim(1)
    let mask =
        (MLXArray(Int32(0) ..< Int32(s)) .< validCount.asType(.int32))
        .expandedDimensions(axis: 0)
    let (_, newState) = gatedDeltaUpdate(
        q: q,
        k: k,
        v: v,
        a: a,
        b: b,
        aLog: aLog,
        dtBias: dtBias,
        state: state,
        mask: mask)

    // Conv state after `validCount` positions: the last K - 1 rows ending at
    // that position in the pass's conv input.
    let kernelRows = convInput.dim(1) - s
    let rows =
        (validCount.asType(.int32) + MLXArray(Int32(0) ..< Int32(kernelRows)))
        .reshaped([1, kernelRows, 1])
    let conv = contiguous(takeAlong(convInput, rows, axis: 1))
    return [newState, conv]
}
