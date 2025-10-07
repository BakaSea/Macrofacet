// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#ifndef PBRT_BASE_MEDIUM_H
#define PBRT_BASE_MEDIUM_H

#include <pbrt/pbrt.h>

#include <pbrt/util/pstd.h>
#include <pbrt/util/rng.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/taggedptr.h>

#include <string>
#include <vector>

namespace pbrt {

// PhaseFunctionSample Definition
struct PhaseFunctionSample {
    Float p;
    Vector3f wi;
    Float pdf;
};

// PhaseFunction Definition
class HGPhaseFunction;
class SGGXPhaseFunction;

class PhaseFunction : public TaggedPointer<HGPhaseFunction, SGGXPhaseFunction> {
  public:
    // PhaseFunction Interface
    using TaggedPointer::TaggedPointer;

    std::string ToString() const;

    PBRT_CPU_GPU inline Float p(Vector3f wo, Vector3f wi) const;

    PBRT_CPU_GPU inline pstd::optional<PhaseFunctionSample> Sample_p(Vector3f wo,
                                                                     Point2f u) const;

    PBRT_CPU_GPU inline Float PDF(Vector3f wo, Vector3f wi) const;
};

class FuzzyMedium;
class HomogeneousMedium;
class GridMedium;
class RGBGridMedium;
class CloudMedium;
class NanoVDBMedium;
class FuzzyNanoVDBMedium;

struct MediumProperties;

// RayMajorantSegment Definition
struct RayMajorantSegment {
    Float tMin, tMax;
    SampledSpectrum sigma_maj;
    std::string ToString() const;
};

// RayMajorantIterator Definition
class HomogeneousMajorantIterator;
class DDAMajorantIterator;

class RayMajorantIterator
    : public TaggedPointer<HomogeneousMajorantIterator, DDAMajorantIterator> {
  public:
    using TaggedPointer::TaggedPointer;

    PBRT_CPU_GPU
    pstd::optional<RayMajorantSegment> Next();

    std::string ToString() const;
};

struct MediumSample {
    MediumSample() = default;
    MediumSample(Point3f p, SampledSpectrum Tr, Float pdf, bool success) : p(p), Tr(Tr), pdf(pdf), success(success) {}
    SampledSpectrum Tr;
    Point3f p;
    Float pdf;
    bool success;
};

// Medium Definition
class Medium
    : public TaggedPointer<  // Medium Types
          FuzzyMedium, HomogeneousMedium, GridMedium, RGBGridMedium, CloudMedium, NanoVDBMedium, FuzzyNanoVDBMedium

          > {
  public:
    // Medium Interface
    using TaggedPointer::TaggedPointer;

    static Medium Create(const std::string &name, const ParameterDictionary &parameters,
                         const Transform &renderFromMedium, const FileLoc *loc,
                         Allocator alloc);

    std::string ToString() const;

    PBRT_CPU_GPU
    bool IsEmissive() const;

    PBRT_CPU_GPU
    MediumProperties SamplePoint(Point3f p, const SampledWavelengths &lambda) const;

    PBRT_CPU_GPU
    MediumProperties SamplePoint(Point3f p, Vector3f wo,
                                 const SampledWavelengths &lambda) const;

    // Medium Public Methods
    RayMajorantIterator SampleRay(Ray ray, Float tMax, const SampledWavelengths &lambda,
                                  ScratchBuffer &buf) const;

    PBRT_CPU_GPU
    MediumSample SampleDistance(Ray ray, Float tMax, Float u, const SampledWavelengths &lambda) const;

    PBRT_CPU_GPU
    SampledSpectrum EvalTransmittance(Point3f x, Point3f y, const SampledWavelengths &lambda) const;

};

// MediumInterface Definition
struct MediumInterface {
    // MediumInterface Public Methods
    std::string ToString() const;

    MediumInterface() = default;
    PBRT_CPU_GPU
    MediumInterface(Medium medium) : inside(medium), outside(medium) {}
    PBRT_CPU_GPU
    MediumInterface(Medium inside, Medium outside) : inside(inside), outside(outside) {}

    PBRT_CPU_GPU
    bool IsMediumTransition() const { return inside != outside; }

    // MediumInterface Public Members
    Medium inside, outside;
};

}  // namespace pbrt

#endif  // PBRT_BASE_MEDIUM_H
