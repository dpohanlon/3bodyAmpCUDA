#include <iostream>
#include <vector>

#include <random>
#include <cmath>

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
    float& operator[](int pos) const { return data[pos]; }

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

    SplineParams() {}

    SplineParams(FloatArr knotsX_, FloatArr knotsY_, FloatArr dydxs_)
                 : knotsX(knotsX_), knotsY(knotsY_), dydxs(dydxs_) {}

    void prefetch()
    {
        knotsX.prefetch();
        knotsY.prefetch();
        dydxs.prefetch();
    }

};

// Basically straight from the Laura++ implementation
// Only a loop with trip count nKnots, probably not worth putting it on the GPU
// Fit for y eventually, here just assume they are given.
std::vector<float> calculateGrads(std::vector<float> x, std::vector<float> y)
{
    int nKnots = x.size();

    std::vector<float> a(nKnots);
    std::vector<float> b(nKnots);
    std::vector<float> c(nKnots);
    std::vector<float> d(nKnots);

    std::vector<float> grad(nKnots);

    a[0] = 0.;
    c[nKnots - 1] = 0.;

    // Left

    float xD10 = x[1] - x[0];

    b[0] = 2. / xD10;
    c[0] = 1. / xD10;
    d[0] = 3. * (y[1] - y[0]) / ( xD10 * xD10 );

    // Right

    float xk12 = x[nKnots - 1] - x[nKnots - 2];

    a[nKnots - 1] = 1. / xk12;
    b[nKnots - 1] = 2. / xk12;
    d[nKnots - 1] = 3. * (y[nKnots - 1] - y[nKnots - 2]) / ( xk12 * xk12 );

    // Internal

    for(uint i = 1; i < nKnots - 1; i++) {

        float xDi = x[i] - x[i - 1];
        float xD1i = x[i + 1] - x[i];

        a[i] = 1. / xDi;
        b[i] = 2. / xDi + 2. / xD1i;
        c[i] = 1./ xD1i;
        d[i] = 3. * (y[i] - y[i - 1]) / ( xDi * xDi ) + 3. * (y[i + 1] - y[i]) / ( xD1i * xD1i );

    }

    c[0] /= b[0];
    d[0] /= b[0];

    for(uint i = 1; i < nKnots - 1; i++) {

        c[i] = c[i] / (b[i] - a[i] * c[i - 1]);
        d[i] = (d[i] - a[i] * d[i - 1]) / (b[i] - a[i] * c[i - 1]);

    }

    d[nKnots - 1] = (d[nKnots - 1] - a[nKnots - 1] * d[nKnots - 2]) / (b[nKnots - 1] - a[nKnots - 1] * c[nKnots - 2]);

    grad[nKnots - 1] = d[nKnots - 1];

    for(int i = nKnots - 2; i >= 0; i--) {
        grad[i] = d[i] - c[i] * grad[i + 1];
    }

    return grad;
}

__global__
void evalSplineKern(const int n, const FloatArr * xs, const SplineParams * params, FloatArr * splineVals)
{
    // All threads handle blockDim.x * gridDim.x
    // consecutive elements (interleaved partitioning)

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride){

        int cell(0);

        // Try and avoid this on GPU if possible -> precompute and save?
        while( (*xs)[i] > params->knotsX[cell+1] ) {
            ++cell;
        }

        // Might be a slow memory access....

        float xLow  = params->knotsX[cell];
        float xHigh = params->knotsX[cell+1];
        float yLow  = params->knotsY[cell];
        float yHigh = params->knotsY[cell+1];

        float t = ((*xs)[i] - xLow) / (xHigh - xLow);
        float a = params->dydxs[cell] * (xHigh - xLow) - (yHigh - yLow);
        float b = -1. * params->dydxs[cell+1] * (xHigh - xLow) + (yHigh - yLow);

        (*splineVals)[i] = (1 - t) * yLow + t * yHigh + t * (1 - t) * ( a * (1 - t) + b * t );

    }

}

void calcSplineGPU(std::vector<float> * knotsX,
                   std::vector<float> * knotsY,
                   std::vector<float> * dydxs,
                   std::vector<float> * masses)
{
    int n = masses->size();
    int nKnots = knotsX->size();

    SplineParams * splineParams = new SplineParams();

    splineParams->knotsX = knotsX;
    splineParams->knotsY = knotsY;
    splineParams->dydxs = dydxs;

    FloatArr * xs = new FloatArr(masses);

    std::vector<float> * splineValsV = new std::vector<float>;
    splineValsV->resize(n);

    std::fill(splineValsV->begin(), splineValsV->end(), 0.0);

    FloatArr * splineVals = new FloatArr(splineValsV);

    splineParams->prefetch();
    xs->prefetch();
    splineVals->prefetch();

    int blockSize = 128;
    int numBlocks = (n + blockSize - 1) / blockSize;

    for (int i = 0; i < 100; i++){
        evalSplineKern<<<numBlocks, blockSize>>>(n, xs, splineParams, splineVals);
    }

    splineParams->sync();
    xs->sync();
    splineVals->sync();

    std::cout << (*splineVals)[0] << std::endl;

    delete splineParams;
}

float normalDist(float mu, float sigma, float x)
{
	float norm = 1. / (sigma * std::sqrt(2 * M_PI));
	float z = (x - mu) / sigma;

	float arg = -0.5 * z * z;

	return norm * std::exp(arg);

}

int main(int argc, char const *argv[]) {

	std::default_random_engine generator;
	std::normal_distribution<float> normal(0.0, 1.0);

	int n = 10000;
    int nKnots = 10;

	std::vector<float> data(n);

	for (auto & d : data) d = normal(generator);

	std::cout << data[123] << std::endl;

    std::vector<float> * knotsX = new std::vector<float>(nKnots);
    std::vector<float> * knotsY = new std::vector<float>(nKnots);
    std::vector<float> * dydxs = new std::vector<float>(nKnots);
    std::vector<float> * masses = new std::vector<float>(n);

    float startKnot = -3.0;
    float endKnot = 3.0;
    float stepSize = (endKnot - startKnot) / nKnots;

    for (int i = 0; i < nKnots; i++) {
        (*knotsX)[i] = startKnot + i * stepSize;
        (*knotsY)[i] = (n / nKnots) * normalDist(0.0, 1.0, (*knotsX)[i]);
    }

    // knotsX->resize(100);
    // knotsY->resize(100);
    // dydxs->resize(100);
    // masses->resize(1000);

    // std::fill(knotsX->begin(), knotsX->end(), 0.05);
    // std::fill(knotsY->begin(), knotsY->end(), 0.05);
    // std::fill(dydxs->begin(), dydxs->end(), 0.05);
    // std::fill(masses->begin(), masses->end(), 0.1);

    // calcSplineGPU(knotsX, knotsY, dydxs, masses);

    return 0;
}
