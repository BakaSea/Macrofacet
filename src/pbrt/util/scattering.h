// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_UTIL_SCATTERING_H
#define PBRT_UTIL_SCATTERING_H

#include <pbrt/pbrt.h>

#include <pbrt/util/math.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/taggedptr.h>
#include <pbrt/util/vecmath.h>

namespace pbrt {

// Scattering Inline Functions
PBRT_CPU_GPU inline Vector3f Reflect(Vector3f wo, Vector3f n) {
    return -wo + 2 * Dot(wo, n) * n;
}

PBRT_CPU_GPU inline bool Refract(Vector3f wi, Normal3f n, Float eta, Float *etap,
                                 Vector3f *wt) {
    Float cosTheta_i = Dot(n, wi);
    // Potentially flip interface orientation for Snell's law
    if (cosTheta_i < 0) {
        eta = 1 / eta;
        cosTheta_i = -cosTheta_i;
        n = -n;
    }

    // Compute $\cos\,\theta_\roman{t}$ using Snell's law
    Float sin2Theta_i = std::max<Float>(0, 1 - Sqr(cosTheta_i));
    Float sin2Theta_t = sin2Theta_i / Sqr(eta);
    // Handle total internal reflection case
    if (sin2Theta_t >= 1)
        return false;

    Float cosTheta_t = std::sqrt(1 - sin2Theta_t);

    *wt = -wi / eta + (cosTheta_i / eta - cosTheta_t) * Vector3f(n);
    // Provide relative IOR along ray to caller
    if (etap)
        *etap = eta;

    return true;
}

PBRT_CPU_GPU inline Float HenyeyGreenstein(Float cosTheta, Float g) {
    // The Henyey-Greenstein phase function isn't suitable for |g| \approx
    // 1 so we clamp it before it becomes numerically instable. (It's an
    // analogous situation to BSDFs: if the BSDF is perfectly specular, one
    // should use one based on a Dirac delta distribution rather than a
    // very smooth microfacet distribution...)
    g = Clamp(g, -.99, .99);
    Float denom = 1 + Sqr(g) + 2 * g * cosTheta;
    return Inv4Pi * (1 - Sqr(g)) / (denom * SafeSqrt(denom));
}

// Fresnel Inline Functions
PBRT_CPU_GPU inline Float FrDielectric(Float cosTheta_i, Float eta) {
    cosTheta_i = Clamp(cosTheta_i, -1, 1);
    // Potentially flip interface orientation for Fresnel equations
    if (cosTheta_i < 0) {
        eta = 1 / eta;
        cosTheta_i = -cosTheta_i;
    }

    // Compute $\cos\,\theta_\roman{t}$ for Fresnel equations using Snell's law
    Float sin2Theta_i = 1 - Sqr(cosTheta_i);
    Float sin2Theta_t = sin2Theta_i / Sqr(eta);
    if (sin2Theta_t >= 1)
        return 1.f;
    Float cosTheta_t = SafeSqrt(1 - sin2Theta_t);

    Float r_parl = (eta * cosTheta_i - cosTheta_t) / (eta * cosTheta_i + cosTheta_t);
    Float r_perp = (cosTheta_i - eta * cosTheta_t) / (cosTheta_i + eta * cosTheta_t);
    return (Sqr(r_parl) + Sqr(r_perp)) / 2;
}

PBRT_CPU_GPU inline Float FrComplex(Float cosTheta_i, pstd::complex<Float> eta) {
    using Complex = pstd::complex<Float>;
    cosTheta_i = Clamp(cosTheta_i, 0, 1);
    // Compute complex $\cos\,\theta_\roman{t}$ for Fresnel equations using Snell's law
    Float sin2Theta_i = 1 - Sqr(cosTheta_i);
    Complex sin2Theta_t = sin2Theta_i / Sqr(eta);
    Complex cosTheta_t = pstd::sqrt(1 - sin2Theta_t);

    Complex r_parl = (eta * cosTheta_i - cosTheta_t) / (eta * cosTheta_i + cosTheta_t);
    Complex r_perp = (cosTheta_i - eta * cosTheta_t) / (cosTheta_i + eta * cosTheta_t);
    return (pstd::norm(r_parl) + pstd::norm(r_perp)) / 2;
}

PBRT_CPU_GPU inline SampledSpectrum FrComplex(Float cosTheta_i, SampledSpectrum eta,
                                              SampledSpectrum k) {
    SampledSpectrum result;
    for (int i = 0; i < NSpectrumSamples; ++i)
        result[i] = FrComplex(cosTheta_i, pstd::complex<Float>(eta[i], k[i]));
    return result;
}

// BSSRDF Utility Declarations
PBRT_CPU_GPU
Float FresnelMoment1(Float invEta);
PBRT_CPU_GPU
Float FresnelMoment2(Float invEta);

enum NormalDistributionType {
    GGX,
    Beckmann,
    GP
};

// TrowbridgeReitzDistribution Definition
class TrowbridgeReitzDistribution {
  public:
    // TrowbridgeReitzDistribution Public Methods
    TrowbridgeReitzDistribution() = default;
    PBRT_CPU_GPU
    TrowbridgeReitzDistribution(Float ax, Float ay)
        : alpha_x(ax), alpha_y(ay) {
        if (!EffectivelySmooth()) {
            // If one direction has some roughness, then the other can't
            // have zero (or very low) roughness; the computation of |e| in
            // D() blows up in that case.
            alpha_x = std::max<Float>(alpha_x, 1e-4f);
            alpha_y = std::max<Float>(alpha_y, 1e-4f);
        }
    }

    PBRT_CPU_GPU inline Float D(Vector3f wm) const {
        Float tan2Theta = Tan2Theta(wm);
        if (IsInf(tan2Theta))
            return 0;
        Float cos4Theta = Sqr(Cos2Theta(wm));
        if (cos4Theta < 1e-16f)
            return 0;
        Float e = tan2Theta * (Sqr(CosPhi(wm) / alpha_x) + Sqr(SinPhi(wm) / alpha_y));
        return 1 / (Pi * alpha_x * alpha_y * cos4Theta * Sqr(1 + e));
    }

    PBRT_CPU_GPU
    bool EffectivelySmooth() const { return std::max(alpha_x, alpha_y) < 1e-3f; }

    PBRT_CPU_GPU
    Float G1(Vector3f w) const { return 1 / (1 + Lambda(w)); }

    PBRT_CPU_GPU
    Float Lambda(Vector3f w) const {
        Float tan2Theta = Tan2Theta(w);
        if (IsInf(tan2Theta))
            return 0;
        Float alpha2 = Sqr(CosPhi(w) * alpha_x) + Sqr(SinPhi(w) * alpha_y);
        return (std::sqrt(1 + alpha2 * tan2Theta) - 1) / 2;
    }

    PBRT_CPU_GPU
    Float G(Vector3f wo, Vector3f wi) const { return 1 / (1 + Lambda(wo) + Lambda(wi)); }

    PBRT_CPU_GPU
    Float D(Vector3f w, Vector3f wm) const {
        return G1(w) / AbsCosTheta(w) * D(wm) * AbsDot(w, wm);
    }

    PBRT_CPU_GPU
    Float PDF(Vector3f w, Vector3f wm) const { return D(w, wm); }

    PBRT_CPU_GPU
    Vector3f Sample_wm(Vector3f w, Point2f u) const {
        // Transform _w_ to hemispherical configuration
        Vector3f wh = Normalize(Vector3f(alpha_x * w.x, alpha_y * w.y, w.z));
        if (wh.z < 0)
            wh = -wh;

        // Find orthonormal basis for visible normal sampling
        Vector3f T1 = (wh.z < 0.99999f) ? Normalize(Cross(Vector3f(0, 0, 1), wh))
                                        : Vector3f(1, 0, 0);
        Vector3f T2 = Cross(wh, T1);

        // Generate uniformly distributed points on the unit disk
        Point2f p = SampleUniformDiskPolar(u);

        // Warp hemispherical projection for visible normal sampling
        Float h = std::sqrt(1 - Sqr(p.x));
        p.y = Lerp((1 + wh.z) / 2, h, p.y);

        // Reproject to hemisphere and transform normal to ellipsoid configuration
        Float pz = std::sqrt(std::max<Float>(0, 1 - LengthSquared(Vector2f(p))));
        Vector3f nh = p.x * T1 + p.y * T2 + pz * wh;
        CHECK_RARE(1e-5f, nh.z == 0);
        return Normalize(
            Vector3f(alpha_x * nh.x, alpha_y * nh.y, std::max<Float>(1e-6f, nh.z)));
    }

    PBRT_CPU_GPU
    Float projectedArea(Vector3f wi) const { return (1.f + Lambda(wi)) * CosTheta(wi); }

    std::string ToString() const;

    // Note that this should probably instead be "return Sqr(roughness)" to
    // be more perceptually uniform, though this wasn't noticed until some
    // time after pbrt-v4 shipped: https://github.com/mmp/pbrt-v4/issues/479.
    // therefore, we will leave it as is so that the rendered results with
    // existing pbrt-v4 scenes doesn't change unexpectedly.
    PBRT_CPU_GPU
    static Float RoughnessToAlpha(Float roughness) { return std::sqrt(roughness); }

    PBRT_CPU_GPU
    void Regularize() {
        if (alpha_x < 0.3f)
            alpha_x = Clamp(2 * alpha_x, 0.1f, 0.3f);
        if (alpha_y < 0.3f)
            alpha_y = Clamp(2 * alpha_y, 0.1f, 0.3f);
    }

  private:
    // TrowbridgeReitzDistribution Private Members
    Float alpha_x, alpha_y;
};

class BeckmannDistribution {
  public:
    BeckmannDistribution() = default;
    PBRT_CPU_GPU
    BeckmannDistribution(Float ax, Float ay) : alpha_x(ax), alpha_y(ay) {
        if (!EffectivelySmooth()) {
            alpha_x = std::max<Float>(alpha_x, 1e-4f);
            alpha_y = std::max<Float>(alpha_y, 1e-4f);
        }
    }

    PBRT_CPU_GPU
    bool EffectivelySmooth() const { return std::max(alpha_x, alpha_y) < 1e-3f; }

    PBRT_CPU_GPU
    Float G1(Vector3f w) const { return 1 / (1 + Lambda(w)); }

    PBRT_CPU_GPU
    Float G(Vector3f wo, Vector3f wi) const { return 1 / (1 + Lambda(wo) + Lambda(wi)); }

    PBRT_CPU_GPU
    Float PDF(Vector3f w, Vector3f wm) const { return D(w, wm); }

    PBRT_CPU_GPU
    inline Float D(Vector3f wm) const {
        if (wm.z <= 0.0f)
            return 0.0f;

        // slope of wm
        const Float slope_x = -wm.x / wm.z;
        const Float slope_y = -wm.y / wm.z;

        // value
        const Float value = P22(slope_x, slope_y) / (wm.z * wm.z * wm.z * wm.z);
        return value;
    }

    PBRT_CPU_GPU
    inline Float D(Vector3f wi, Vector3f wm) const {
        if (wm.z <= 0.0f)
            return 0.0f;

        // normalization coefficient
        const Float projectedarea = projectedArea(wi);
        if (projectedarea < FLT_EPSILON)
            return 0;
        const Float c = 1.0f / projectedarea;

        // value
        const Float value = c * std::max(0.0f, Dot(wi, wm)) * D(wm);
        return value;
    }

    PBRT_CPU_GPU
    inline Float P22(Float slope_x, Float slope_y) const {
        const Float value = 1.0f / (Pi * alpha_x * alpha_y) *
                            expf(-slope_x * slope_x / (alpha_x * alpha_x) -
                                 slope_y * slope_y / (alpha_y * alpha_y));
        return value;
    }

    PBRT_CPU_GPU
    Float alpha_i(Vector3f wi) const {
        const Float invSinTheta2 = 1.0f / (1.0f - wi.z * wi.z);
        const Float cosPhi2 = wi.x * wi.x * invSinTheta2;
        const Float sinPhi2 = wi.y * wi.y * invSinTheta2;
        const Float alpha_i =
            sqrtf(cosPhi2 * alpha_x * alpha_x + sinPhi2 * alpha_y * alpha_y);
        return alpha_i;
    }

    PBRT_CPU_GPU
    Float Lambda(Vector3f wi) const {
        if (wi.z > 0.9999f)
            return 0.0f;
        if (wi.z < -0.9999f)
            return -1.0f;

        // a
        const Float theta_i = acosf(wi.z);
        const Float a = 1.0f / tanf(theta_i) / alpha_i(wi);

        // value
        const Float value =
            0.5f * ((float)erf(a) - 1.0f) + 1.0f / (2.f * a * sqrtf(Pi)) * expf(-a * a);

        return value;
    }

    PBRT_CPU_GPU
    Float projectedArea(Vector3f wi) const {
        if (wi.z > 0.9999f)
            return 1.0f;
        if (wi.z < -0.9999f)
            return 0.0f;

        // a
        const Float alphai = alpha_i(wi);
        const Float theta_i = acosf(wi.z);
        const Float a = 1.0f / tanf(theta_i) / alphai;

        // value
        const Float value =
            0.5f * ((float)erf(a) + 1.0f) * wi.z +
            1.f / (2.f * sqrtf(Pi)) * alphai * sinf(theta_i) * expf(-a * a);

        return value;
    }

    PBRT_CPU_GPU
    Vector3f Sample_wm(Vector3f wi, Point2f u) const {
        // stretch to match configuration with alpha=1.0
        const Vector3f wi_11 = Normalize(Vector3f(alpha_x * wi.x, alpha_y * wi.y, wi.z));

        // sample visible slope with alpha=1.0
        Vector2f slope_11 = sampleP22_11(acosf(wi_11.z), u.x, u.y);

        // align with view direction
        const float phi = atan2(wi_11.y, wi_11.x);
        Vector2f slope(cosf(phi) * slope_11.x - sinf(phi) * slope_11.y,
                       sinf(phi) * slope_11.x + cos(phi) * slope_11.y);

        // stretch back
        slope.x *= alpha_x;
        slope.y *= alpha_y;

        // if numerical instability
        if ((slope.x != slope.x) || !IsFinite(slope.x)) {
            if (wi.z > 0)
                return Vector3f(0.0f, 0.0f, 1.0f);
            else
                return Normalize(Vector3f(wi.x, wi.y, 0.0f));
        }

        // compute normal
        const Vector3f wm = Normalize(Vector3f(-slope.x, -slope.y, 1.0f));
        return wm;
    }

    PBRT_CPU_GPU
    Vector2f sampleP22_11(Float theta_i, Float U, Float U_2) const {
        Vector2f slope;

        if (theta_i < 0.0001f) {
            const float r = sqrtf(-logf(U));
            const float phi = 6.28318530718f * U_2;
            slope.x = r * cosf(phi);
            slope.y = r * sinf(phi);
            return slope;
        }

        // constant
        const float sin_theta_i = sinf(theta_i);
        const float cos_theta_i = cosf(theta_i);

        // slope associated to theta_i
        const float slope_i = cos_theta_i / sin_theta_i;

        // projected area
        const float a = cos_theta_i / sin_theta_i;
        const float projectedarea = 0.5f * ((float)erf(a) + 1.0f) * cos_theta_i +
                                    1.f / (2.f * sqrtf(Pi)) * sin_theta_i * expf(-a * a);
        if (projectedarea < 0.0001f || projectedarea != projectedarea)
            return Vector2f(0, 0);
        // VNDF normalization factor
        const float c = 1.0f / projectedarea;

        // search
        float erf_min = -0.9999f;
        float erf_max = std::max(erf_min, (float)erf(slope_i));
        float erf_current = 0.5f * (erf_min + erf_max);

        while (erf_max - erf_min > 0.00001f) {
            if (!(erf_current >= erf_min && erf_current <= erf_max))
                erf_current = 0.5f * (erf_min + erf_max);

            // evaluate slope
            const float slope = ErfInv(erf_current);

            // CDF
            const float CDF =
                (slope >= slope_i)
                    ? 1.0f
                    : c * (1.f / (2.f * sqrtf(Pi)) * sin_theta_i * expf(-slope * slope) +
                           cos_theta_i * (0.5f + 0.5f * (float)erf(slope)));
            const float diff = CDF - U;

            // test estimate
            if (std::abs(diff) < 0.00001f)
                break;

            // update bounds
            if (diff > 0.0f) {
                if (erf_max == erf_current)
                    break;
                erf_max = erf_current;
            } else {
                if (erf_min == erf_current)
                    break;
                erf_min = erf_current;
            }

            // update estimate
            const float derivative =
                0.5f * c * cos_theta_i - 0.5f * c * sin_theta_i * slope;
            erf_current -= diff / derivative;
        }

        slope.x = ErfInv(std::min(erf_max, std::max(erf_min, erf_current)));
        slope.y = ErfInv(2.0f * U_2 - 1.0f);
        return slope;
    }

    PBRT_CPU_GPU
    static Float RoughnessToAlpha(Float roughness) { return std::sqrt(roughness); }

    PBRT_CPU_GPU
    void Regularize() {
        if (alpha_x < 0.3f)
            alpha_x = Clamp(2 * alpha_x, 0.1f, 0.3f);
        if (alpha_y < 0.3f)
            alpha_y = Clamp(2 * alpha_y, 0.1f, 0.3f);
    }

    std::string ToString() const;

    Float alpha_x, alpha_y;
};

class GPDistribution {
  public:
    GPDistribution() = default;
    PBRT_CPU_GPU
    GPDistribution(Float ax, Float ay, Float az) : beckmann(ax, ay), alpha_z(az) {
        if (!EffectivelySmooth()) {
            beckmann.alpha_x = std::max<Float>(beckmann.alpha_x, 1e-4f);
            beckmann.alpha_y = std::max<Float>(beckmann.alpha_y, 1e-4f);
            alpha_z = std::max<Float>(alpha_z, 1e-4f);
        }
    }

    PBRT_CPU_GPU
    bool EffectivelySmooth() const {
        return std::max(beckmann.alpha_x, std::max(beckmann.alpha_y, alpha_z)) < 1e-3f;
    }

    PBRT_CPU_GPU
    Float G1(Vector3f w) const { return 1 / (1 + Lambda(w)); }

    PBRT_CPU_GPU
    Float G(Vector3f wo, Vector3f wi) const { return 1 / (1 + Lambda(wo) + Lambda(wi)); }

    PBRT_CPU_GPU
    Float PDF(Vector3f w, Vector3f wm) const { return D(w, wm); }

    PBRT_CPU_GPU
    Float D(Vector3f wm) const;

    PBRT_CPU_GPU
    Float D(Vector3f wi, Vector3f wm) const;

    PBRT_CPU_GPU
    inline double alpha_i(Vector3f wi) const {
        return std::sqrt(beckmann.alpha_x * beckmann.alpha_x * wi.x * wi.x +
                         beckmann.alpha_y * beckmann.alpha_y * wi.y * wi.y +
                         alpha_z * alpha_z * wi.z * wi.z);
    }

    PBRT_CPU_GPU
    Float Lambda(Vector3f wi) const {
        if (wi.z > 0.9999f)
            return 0.0f;
        if (wi.z < -0.9999f)
            return -1.0f;

        const double a = wi.z / alpha_i(wi);

        const double value =
            0.5 * (erf(a) - 1.0) + 1.0 / (2.0 * a * std::sqrt(Pi)) * exp(-a * a);

        return value;
    }

    PBRT_CPU_GPU
    Float projectedArea(Vector3f wi) const {
        if (beckmann.alpha_x == 0.f && beckmann.alpha_y == 0.f)
            return 0.f;
        // a
        const double alphai = alpha_i(wi);
        const double a = wi.z / alphai;

        // value
        const double value = 0.5 * (erf(a) + 1.0) * wi.z +
                             1.0 / (2.0 * std::sqrt(Pi)) * alphai * exp(-a * a);

        return value;
    }

    PBRT_CPU_GPU
    std::pair<Vector3f, Float> Sample_wm(Vector3f wi, Point2f u) const;

    PBRT_CPU_GPU
    static Float RoughnessToAlpha(Float roughness) { return std::sqrt(roughness); }

    PBRT_CPU_GPU
    void Regularize() { beckmann.Regularize();
    }

    std::string ToString() const;

    BeckmannDistribution beckmann;
    Float alpha_z;
};

class NormalDistribution {
  public:
    NormalDistribution() = default;
    NormalDistribution(NormalDistributionType type, Float ax, Float ay, Float az = 1e-4f) : type(type) {
        switch (type) {
        case pbrt::GGX:
            ggx = TrowbridgeReitzDistribution(ax, ay);
            break;
        case pbrt::Beckmann:
            beckmann = BeckmannDistribution(ax, ay);
            break;
        case pbrt::GP:
            gp = GPDistribution(ax, ay, az);
            break;
        default:
            break;
        }
    }

    PBRT_CPU_GPU inline Float D(Vector3f wm) const {
        switch (type) {
        case pbrt::GGX:
            return ggx.D(wm);
        case pbrt::Beckmann:
            return beckmann.D(wm);
        case pbrt::GP:
            return gp.D(wm);
        default:
            return 0.f;
        }
    }

    PBRT_CPU_GPU
    bool EffectivelySmooth() const {
        switch (type) {
        case pbrt::GGX:
            return ggx.EffectivelySmooth();
        case pbrt::Beckmann:
            return beckmann.EffectivelySmooth();
        case pbrt::GP:
            return gp.EffectivelySmooth();
        default:
            return false;
        }
    }

    PBRT_CPU_GPU
    Float G(Vector3f wo, Vector3f wi) const {
        switch (type) {
        case pbrt::GGX:
            return ggx.G(wo, wi);
        case pbrt::Beckmann:
            return beckmann.G(wo, wi);
        case pbrt::GP:
            return gp.G(wo, wi);
        default:
            return 0.f;
        }
    }

    PBRT_CPU_GPU
    Float D(Vector3f w, Vector3f wm) const {
        switch (type) {
        case pbrt::GGX:
            return ggx.D(w, wm);
        case pbrt::Beckmann:
            return beckmann.D(w, wm);
        case pbrt::GP:
            return gp.D(w, wm);
        default:
            return 0.f;
        }
    }

    PBRT_CPU_GPU
    Float PDF(Vector3f w, Vector3f wm) const {
        switch (type) {
        case pbrt::GGX:
            return ggx.PDF(w, wm);
        case pbrt::Beckmann:
            return beckmann.PDF(w, wm);
        case pbrt::GP:
            return gp.PDF(w, wm);
        default:
            return 0.f;
        }
    }

    PBRT_CPU_GPU
    Float projectedArea(Vector3f w) const {
        switch (type) {
        case pbrt::GGX:
            return ggx.projectedArea(w);
        case pbrt::Beckmann:
            return beckmann.projectedArea(w);
        case pbrt::GP:
            return gp.projectedArea(w);
        default:
            return 0.f;
        }
    }

    PBRT_CPU_GPU
    std::pair<Vector3f, Float> Sample_wm(Vector3f w, Point2f u) const {
        switch (type) {
        case pbrt::GGX:
            return {ggx.Sample_wm(w, u), 1.f};
        case pbrt::Beckmann:
            return {beckmann.Sample_wm(w, u), 1.f};
        case pbrt::GP:
            return gp.Sample_wm(w, u);
        default:
            return {Vector3f(0.f, 0.f, 0.f), 0.f};
        }
    }

    PBRT_CPU_GPU
    void Regularize() {
        switch (type) {
        case pbrt::GGX:
            return ggx.Regularize();
        case pbrt::Beckmann:
            return beckmann.Regularize();
        case pbrt::GP:
            return gp.Regularize();
        default:
            return;
        }
    }

    std::string ToString() const {
        switch (type) {
        case pbrt::GGX:
            return ggx.ToString();
        case pbrt::Beckmann:
            return beckmann.ToString();
        case pbrt::GP:
            return gp.ToString();
        default:
            return "";
        }
    }

    NormalDistributionType type;
    union {
        TrowbridgeReitzDistribution ggx;
        BeckmannDistribution beckmann;
        GPDistribution gp;
    };
};

}  // namespace pbrt

#endif  // PBRT_UTIL_SCATTERING_H
