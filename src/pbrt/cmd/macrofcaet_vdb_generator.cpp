#include <vector>
#include <string>

#include <pbrt/pbrt.h>
#include <pbrt/options.h>
#include <pbrt/util/args.h>
#include <pbrt/util/mesh.h>
#include <pbrt/util/transform.h>
#include <pbrt/shapes.h>
#include <pbrt/media.h>

#include <nanovdb/NanoVDB.h>
#define NANOVDB_USE_ZIP 1
#include <nanovdb/util/IO.h>
#include <nanovdb/util/GridHandle.h>
#include <nanovdb/util/GridBuilder.h>

#include <Eigen/Core>
#include <igl/fast_winding_number.h>
#include <igl/AABB.h>

using namespace pbrt;

float Density(float x, float sigma, float k) {
    float pdf = Gaussian(x, 0.f, sigma);
    float cdf = 0.5f * (1.f + erf(x / (sigma * std::sqrt(2.f))));
    return k * pdf / cdf;
}

int main(int argc, char* argv[]) {
    PBRTOptions options;
    InitPBRT(options);

    std::vector<std::string> args = GetCommandLineArguments(argv);

    auto onError = [](const std::string &err) {
        exit(1);
    };

    std::string filename;
    float sigma = 0.05f;
    float minSigma = 0.005f;
    float alpha = 0.5f;
    float minAlpha = 0.1f;
    int xRes = 32, yRes = 32, zRes = 32;
    for (auto iter = args.begin(); iter != args.end(); ++iter) {
        if ((*iter)[0] != '-') {
            if (filename.empty())
                filename = *iter;
            else {
                exit(1);
            }
        } else if (ParseArg(&iter, args.end(), "sigma", &sigma, onError) ||
                   ParseArg(&iter, args.end(), "alpha", &alpha, onError) ||
                   ParseArg(&iter, args.end(), "x", &xRes, onError) ||
                   ParseArg(&iter, args.end(), "y", &yRes, onError) ||
                   ParseArg(&iter, args.end(), "z", &zRes, onError)) {
        
        } else {
            exit(1);
        }
    }
    printf("sigma=%f\n", sigma);
    printf("alpha=%f\n", alpha);
    printf("x=%d\n", xRes);
    printf("y=%d\n", yRes);
    printf("z=%d\n", zRes);
    //minSigma = sigma;

    Allocator alloc;

    Transform renderFromObject;

    TriQuadMesh plyMesh = TriQuadMesh::ReadPLY(filename);
    TriangleMesh *mesh = alloc.new_object<TriangleMesh>(
        renderFromObject, false, plyMesh.triIndices, plyMesh.p,
        std::vector<Vector3f>(), plyMesh.n, plyMesh.uv, plyMesh.faceIndices, alloc);

    pstd::vector<Shape> shapes = Triangle::CreateTriangles(mesh, alloc);
    
    Bounds3f bbox;
    for (Shape shape : shapes) {
        Triangle *triangle = shape.Cast<Triangle>();
        bbox = Union(bbox, triangle->Bounds());
    }

    float minX = bbox.pMin.x;
    float minY = bbox.pMin.y;
    float invXInterval = 1.f / (bbox.pMax.x - bbox.pMin.x);
    float invYInterval = 1.f / (bbox.pMax.y - bbox.pMin.y);

    Vector3f sigmaOffset(3.f * sigma, 3.f * sigma, 3.f * sigma);
    bbox.pMin -= sigmaOffset;
    bbox.pMax += sigmaOffset;
    Vector3f gap = bbox.Diagonal();
    float dx = std::min(gap.x / xRes, std::min(gap.y / yRes, gap.z / zRes));
    printf("dx=%f\n", dx);

    nanovdb::GridBuilder<float> densityBuilder = nanovdb::GridBuilder(0.f);
    auto &densityAccessor = densityBuilder.getAccessor();
    auto densityHandle = densityBuilder.getHandle(dx, nanovdb::Vec3d(0), "density");
    auto *grid = densityHandle.grid<float>();
    
    nanovdb::GridBuilder<float> alphaBuilder = nanovdb::GridBuilder(0.f);
    auto &alphaAccessor = alphaBuilder.getAccessor();
    nanovdb::GridBuilder<float> sdfBuilder = nanovdb::GridBuilder(0.f);
    auto &sdfAccessor = sdfBuilder.getAccessor();

    {
        // for (int i = 0; i < shapes.size(); ++i) {
        //     printf("%d/%d\n", i + 1, shapes.size());

        //    Triangle *triangle = shapes[i].Cast<Triangle>();
        //
        //    Bounds3f box = triangle->Bounds();
        //    box.pMin -= sigmaOffset;
        //    box.pMax += sigmaOffset;
        //    nanovdb::Coord minCoord = nanovdb::Coord::Floor(
        //        grid->worldToIndexF(nanovdb::Vec3f(box.pMin.x, box.pMin.y,
        //        box.pMin.z)));
        //    nanovdb::Coord maxCoord = nanovdb::Coord::Floor(
        //        grid->worldToIndexF(nanovdb::Vec3f(box.pMax.x, box.pMax.y,
        //        box.pMax.z)));

        //    pstd::optional<ShapeSample> shapeSample = triangle->Sample(Point2f(0, 0));
        //    Vector3f n = Vector3f(shapeSample->intr.n);

        //    for (int i = minCoord.x(); i <= maxCoord.x(); ++i) {
        //        for (int j = minCoord.y(); j <= maxCoord.y(); ++j) {
        //            for (int k = minCoord.z(); k <= maxCoord.z(); ++k) {
        //                nanovdb::Vec3f xyz = grid->indexToWorld(nanovdb::Vec3f(i, j,
        //                k));
        //                //printf("index: %d %d %d\n", i, j, k);
        //                //printf("world pos: %lf %lf %lf\n", xyz[0], xyz[1], xyz[2]);
        //                Point3f origin(xyz[0], xyz[1], xyz[2]);
        //
        //                Ray ray(origin, -n);
        //                pstd::optional<ShapeIntersection> shapeIntr =
        //                triangle->Intersect(ray); Float t = 0.f; Float intrSigma = 0.f;
        //                if (shapeIntr) {
        //                    t = shapeIntr->tHit;
        //                    //intrSigma = sigma;
        //                    intrSigma = Lerp((shapeIntr->intr.p().x - minX) *
        //                    invXInterval,
        //                                     minSigma, sigma);
        //                } else {
        //                    ray.d = n;
        //                    shapeIntr = triangle->Intersect(ray);
        //                    if (shapeIntr) {
        //                        t = -shapeIntr->tHit;
        //                        //intrSigma = sigma;
        //                        intrSigma =
        //                            Lerp((shapeIntr->intr.p().x - minX) * invXInterval,
        //                                 minSigma, sigma);
        //                    } else {
        //                        continue;
        //                    }
        //                }

        //                if (-3.f * intrSigma < t && t < 3.f * intrSigma) {
        //                    Float d = Density(t, intrSigma, 1.f);
        //                    nanovdb::Coord ijk = nanovdb::Coord(i, j, k);
        //                    float v = accessor.getValue(ijk);
        //                    accessor.setValue(ijk, v + d);
        //                    //printf("%lf\n", v + d);
        //                }
        //                //accessor.setValue(nanovdb::Coord(i, j, k), 1.f);
        //            }
        //        }
        //    }
        //}
    }

    igl::FastWindingNumberBVH bvh;
    igl::AABB<Eigen::MatrixXd, 3> tree;
    Eigen::MatrixXd V;
    Eigen::MatrixXi T, F;

    V.resize(mesh->nVertices, 3);
    for (int i = 0; i < mesh->nVertices; ++i) {
        V(i, 0) = mesh->p[i].x;
        V(i, 1) = mesh->p[i].y;
        V(i, 2) = mesh->p[i].z;
    }

    F.resize(mesh->nTriangles, 3);
    for (int i = 0; i < mesh->nTriangles; ++i) {
        const int *v = &mesh->vertexIndices[3 * i];
        F(i, 0) = v[0];
        F(i, 1) = v[1];
        F(i, 2) = v[2];
    }

    tree.init(V, F);
    igl::fast_winding_number(V, F, 2, bvh);

    nanovdb::Coord minCoord = nanovdb::Coord::Floor(
        grid->worldToIndexF(nanovdb::Vec3f(bbox.pMin.x, bbox.pMin.y, bbox.pMin.z)));
    nanovdb::Coord maxCoord = nanovdb::Coord::Floor(
        grid->worldToIndexF(nanovdb::Vec3f(bbox.pMax.x, bbox.pMax.y, bbox.pMax.z)));

    for (int i = minCoord.x(); i <= maxCoord.x(); ++i) {
        printf("%d/%d\n", i-minCoord.x(), maxCoord.x()-minCoord.x());
        for (int j = minCoord.y(); j <= maxCoord.y(); ++j) {
            for (int k = minCoord.z(); k <= maxCoord.z(); ++k) {
                //printf("%d %d %d\n", i, j, k);
                nanovdb::Vec3f xyz = grid->indexToWorldF(nanovdb::Vec3f(i, j, k));
                Eigen::RowVector3d P(xyz[0], xyz[1], xyz[2]);
                Eigen::VectorXd sqrD;
                Eigen::VectorXi I;
                Eigen::RowVector3d closestPoint;
                tree.squared_distance(V, F, P, sqrD, I, closestPoint);
                float w = igl::fast_winding_number(bvh, 2, P);
                float dist = std::sqrt(sqrD(0)) * (1.f - 2.f * std::abs(w));
                //float intrSigma = sigma;
                float intrSigma = Lerp((closestPoint.y() - minY) * invYInterval, sigma, minSigma);
                //float intrAlpha = alpha;
                float intrAlpha = Lerp((closestPoint.x() - minX) * invXInterval, minAlpha, alpha);
                if (-3.f * intrSigma < dist && dist < 3.f * intrSigma) {
                    float d = Density(dist, intrSigma, 1.f);
                    nanovdb::Coord ijk(i, j, k);
                    densityAccessor.setValue(ijk, d);
                    if (IsInf(d)) {
                        printf("%d %d %d\n", i, j, k);
                    }
                    alphaAccessor.setValue(ijk, intrAlpha);
                    sdfAccessor.setValue(ijk, dist);
                }
            }
        }
    }

    densityHandle = densityBuilder.getHandle(dx, nanovdb::Vec3d(0), "density");
    auto alphaHandle = alphaBuilder.getHandle(dx, nanovdb::Vec3d(0), "alpha");
    auto sdfHandle = sdfBuilder.getHandle(dx, nanovdb::Vec3d(0), "sdf");
    std::vector<nanovdb::GridHandle<nanovdb::HostBuffer>> handles(3);
    handles[0] = std::move(densityHandle);
    handles[1] = std::move(alphaHandle);
    handles[2] = std::move(sdfHandle);
    nanovdb::io::writeGrids<nanovdb::HostBuffer, std::vector>("out.nvdb", handles);

    return 0;
}