#include <pbrt/pbrt.h>
#include <pbrt/options.h>
#include <pbrt/media.h>

using namespace pbrt;

Vector3f generateRandomDirection() {
    return Normalize(Vector3f(0.f, 0, -1.f));
}

int main() {
    PBRTOptions options;
    InitPBRT(options);

    NormalDistribution ndf(GP, 1.f, 1.f, 1.f);
    Frame frame;
    SpecularPhaseFunction phase(SampledSpectrum(0.f), ndf, frame);

    //Vector3f wo = generateRandomDirection();

    Vector3f wo(-0.201376423f, -0.582662702f, 0.787370026f);
    Vector3f wi(0.362200797f, -0.464358330f, -0.808196545f);
    Vector3f wh = Normalize(wo+wi);
    Float vndf = ndf.D(wo, wh);
    printf("%lf\n", Dot(wo, wh));
    printf("%lf\n", 0.25f * vndf / Dot(wo, wh));
    printf("%lf\n", ndf.D(wh));

    double result = 0;
    double stepTheta = 0.005;
    double stepPhi = 0.005;
    for (double theta = 0.f; theta <= Pi; theta += stepTheta) {
        for (double phi = 0.f; phi <= 2.f * Pi; phi += stepPhi) {
            Vector3f w(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
            result += stepTheta * stepPhi * sin(theta) * phase.PDF(wo, w);
        }
    }
    printf("p(wo, wi)=%lf\n", result);

    return 0;
}