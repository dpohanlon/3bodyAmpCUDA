#include <iostream>
#include <vector>

class Managed {

public:

    size_t size;

    void *operator new(size_t len) {

        void *ptr;

        cudaMallocManaged(&ptr, len);
        cudaDeviceSynchronize();

        return ptr;
    }

    void operator delete(void *ptr) {

        cudaDeviceSynchronize();

        cudaFree(ptr);
    }

    void sync() {
        cudaDeviceSynchronize();
    }
};

class FloatArr : public Managed
{

public:

    int size;
    float * data;

    FloatArr() : size(0), data(0)
    {

    }

    FloatArr(std::vector<float> * a) : size(a->size())
    {
        // Allocate unified memory
        realloc_(a->size());

        // Copy C array from vector
        memcpy(data, a->data(), a->size() * sizeof(float));
    }

    FloatArr(const FloatArr & a) : size(a.size)
    {
        realloc_(a.size);
        memcpy(data, a.data, a.size * sizeof(float));
    }

    ~FloatArr() { cudaFree(data); }

    FloatArr& operator=(std::vector<float> * a)
    {
        size = a->size();
        realloc_(a->size());
        memcpy(data, a->data(), size * sizeof(float));
        return *this;
    }

    void prefetch()
    {
        int device = -1;
        cudaGetDevice(&device);

        cudaMemPrefetchAsync(data, size * sizeof(float), device, NULL);
        cudaMemPrefetchAsync(&size, sizeof(int), device, NULL);
    }

    __host__ __device__
    float& operator[](int pos) { return data[pos]; }

private:

    void realloc_(int s)
    {
        // cudaFree(data);
        cudaMallocManaged(&data, s * sizeof(float));
        cudaDeviceSynchronize();
    }

};

class SplineParams : public Managed
{

public:

    FloatArr knotsX;
    FloatArr knotsY;
    FloatArr dydxs;

    KernelParamsL() {}

    KernelParamsL(FloatArr knotsX_, FloatArr knotsY_, FloatArr dydxs_)
                  : knotsX(knotsX_), knotsY(knotsY_), dydxs(dydxs_) {}

    void prefetch()
    {
        knotsX.prefetch();
        knotsY.prefetch();
        dydxs_.prefetch();
    }

};

__global__
void evalSplineKern(const int n, const FloatArr * xs, const SplineParams * params, FloatArr * splineVals)
{
    // All threads handle blockDim.x * gridDim.x
    // consecutive elements (interleaved partitioning)

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride){

        int cell(0);

        while( xs->at(i) > params->knotsX->at(cell+1) ) {
            ++cell;
        }

        float xLow  = params->knotsX->at(cell);
        float xHigh = params->knotsX->at(cell+1);
        float yLow  = params->knotsY->at(cell);
        float yHigh = params->knotsY->at(cell+1);

        float t = (xs->at(i) - xLow) / (xHigh - xLow);
        float a = params->dydxs->at(cell) * (xHigh - xLow) - (yHigh - yLow);
        float b = -1. * params->dydxs->at(cell+1) * (xHigh - xLow) + (yHigh - yLow);

        splineVals->at(i) = (1 - t) * yLow + t * yHigh + t * (1 - t) * ( a * (1 - t) + b * t );

    }

}

void calcSplineGPU(std::vector<float> & knotsX,
                   std::vector<float> & knotsY,
                   std::vector<float> & dydxs,
                   std::vector<float> & masses)
{
    int n = masses.size();
    int nKnots = knotsX.size();

    SplineParams * splineParams = new SplineParams();

    splineParams->knotsX = knotsX;
    splineParams->knotsY = knotsY;
    splineParams->dydxs = dydxs;

    FloatArr xs = masses;

    std::vector<float> splineValsV;
    std::fill(splineValsV.begin(), splineValsV.end(), 0.0);

    FloatArr splineVals = splineValsV;

    splineParams->prefetch();
    xs.prefetch();
    splineVals.prefetch();

    int blockSize = 128;
    int numBlocks = (n + blockSize - 1) / blockSize;

    for (int i = 0; i < 100; i++){
        evalSplineKern<<<numBlocks, blockSize>>>(n, &xs, splineParams, &splineVals);
    }

    splineParams->sync();
    xs.sync();
    splineVals.sync();

    delete splineParams;
}

int main(int argc, char const *argv[]) {

    // Do stuff.

    return 0;
}
