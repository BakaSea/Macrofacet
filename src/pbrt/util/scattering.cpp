// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include <pbrt/util/scattering.h>

namespace pbrt {

// BSSRDF Utility Functions
Float FresnelMoment1(Float eta) {
    Float eta2 = eta * eta, eta3 = eta2 * eta, eta4 = eta3 * eta, eta5 = eta4 * eta;
    if (eta < 1)
        return 0.45966f - 1.73965f * eta + 3.37668f * eta2 - 3.904945 * eta3 +
               2.49277f * eta4 - 0.68441f * eta5;
    else
        return -4.61686f + 11.1136f * eta - 10.4646f * eta2 + 5.11455f * eta3 -
               1.27198f * eta4 + 0.12746f * eta5;
}

Float FresnelMoment2(Float eta) {
    Float eta2 = eta * eta, eta3 = eta2 * eta, eta4 = eta3 * eta, eta5 = eta4 * eta;
    if (eta < 1) {
        return 0.27614f - 0.87350f * eta + 1.12077f * eta2 - 0.65095f * eta3 +
               0.07883f * eta4 + 0.04860f * eta5;
    } else {
        Float r_eta = 1 / eta, r_eta2 = r_eta * r_eta, r_eta3 = r_eta2 * r_eta;
        return -547.033f + 45.3087f * r_eta3 - 218.725f * r_eta2 + 458.843f * r_eta +
               404.557f * eta - 189.519f * eta2 + 54.9327f * eta3 - 9.00603f * eta4 +
               0.63942f * eta5;
    }
}

std::string TrowbridgeReitzDistribution::ToString() const {
    return StringPrintf("[ TrowbridgeReitzDistribution alpha_x: %f alpha_y: %f ]",
                        alpha_x, alpha_y);
}

std::string BeckmannDistribution::ToString() const {
    return StringPrintf("[ BeckmannDistribution alpha_x: %f alpha_y: %f ]", alpha_x,
                        alpha_y);
}

PBRT_CPU_GPU
std::pair<Vector3f, Float> GPDistribution::Sample_wm(Vector3f wi, Point2f u) const {
    RNG rng;
    const Float t = 0.1f;
    if (rng.Uniform<Float>() < t) {
        Frame frame = Frame::FromZ(wi);
        Vector3f wm = SampleUniformHemisphere(u);
        wm = frame.FromLocal(wm);
        return {wm, t * Inv2Pi + (1 - t) * beckmann.D(wi, wm)};
    } else {
        Vector3f wm = beckmann.Sample_wm(wi, u);
        Float pdf = beckmann.D(wi, wm);
        return {wm, t * Inv2Pi + (1 - t) * pdf};
    }
}

PBRT_CPU_GPU
Float GPDistribution::D(Vector3f wm) const {
    //double C = 1.f / (alpha_z * alpha_z);
    //double A = wm.x * wm.x / (alpha_x * alpha_x) + wm.y * wm.y / (alpha_y * alpha_y) +
    //          wm.z * wm.z * C;
    //double B = wm.z * C;
    //double sqrtPi = std::sqrt(Pi);
    //double sqrtA = std::sqrt(A);
    //return 1.0 / (Pi * sqrtPi * alpha_x * alpha_y * alpha_z) * exp(-C) *
    //       (B / (2.0 * A * A) + sqrtPi / (4.0 * A * sqrtA) * exp(B * B / A) *
    //                                (2.0 * B * B / A + 1.f) * erfc(-B / sqrtA));
    double C = 1.f / (alpha_z * alpha_z);
    double A = wm.x * wm.x / (beckmann.alpha_x * beckmann.alpha_x) +
               wm.y * wm.y / (beckmann.alpha_y * beckmann.alpha_y) + wm.z * wm.z * C;
    double B = wm.z * C;
    const double sqrtPi = std::sqrt(Pi);
    const double sqrtA = std::sqrt(A);

    double d =
        1.0 / (Pi * sqrtPi * beckmann.alpha_x * beckmann.alpha_y * alpha_z) *
        exp(-C + B * B / A) *
        ((B * B / A + 1.0) / (2 * A * A) * exp(-B * B / A) +
         B * sqrtPi / (2.0 * A * A * sqrtA) * (1.5 + B * B / A) * erfc(-B / sqrtA));

    return std::abs(wm.z) * d;
}

PBRT_CPU_GPU
Float GPDistribution::D(Vector3f wi, Vector3f wm) const {
    double projectedarea = projectedArea(wi);
    if (projectedarea == 0)
        return 0;

    double C = 1.f / (alpha_z * alpha_z);
    double A = wm.x * wm.x / (beckmann.alpha_x * beckmann.alpha_x) + wm.y * wm.y / (beckmann.alpha_y * beckmann.alpha_y) +
               wm.z * wm.z * C;
    double B = wm.z * C;
    const double sqrtPi = std::sqrt(Pi);
    const double sqrtA = std::sqrt(A);

    double d =
        1.0 / (Pi * sqrtPi * beckmann.alpha_x * beckmann.alpha_y * alpha_z) * exp(-C + B * B / A) *
        ((B * B / A + 1.0) / (2 * A * A) * exp(-B * B / A) +
         B * sqrtPi / (2.0 * A * A * sqrtA) * (1.5 + B * B / A) * erfc(-B / sqrtA));

    return std::max(Dot(wi, wm), 0.f) * d / projectedarea;
}

std::string GPDistribution::ToString() const {
    return StringPrintf("[ GPDistribution alpha_x: %f alpha_y: %f ]", beckmann.alpha_x,
                        beckmann.alpha_y, alpha_z);
}

}  // namespace pbrt
