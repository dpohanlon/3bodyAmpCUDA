#include <iostream>
#include <vector>

#include <random>
#include <cmath>

// #include <boost/timer/timer.hpp>

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

template <class T>
class CuArray : public Managed
{

public:

    int size;
    T * data;

    CuArray() : size(0), data(nullptr)
    {

    }

    CuArray(std::vector<T> * a) : size(a->size())
    {
        // Allocate unified memory
        realloc_(a->size());

        // Copy C array from vector
        memcpy(data, a->data(), a->size() * sizeof(T));
    }

    CuArray(const CuArray<T> & a) : size(a.size)
    {
        realloc_(a.size);
        memcpy(data, a.data, a.size * sizeof(T));
    }

    ~CuArray() { cudaFree(data); }

    CuArray& operator=(std::vector<T> * a)
    {
        size = a->size();
        realloc_(a->size());
        memcpy(data, a->data(), size * sizeof(T));
        return *this;
    }

    void prefetch()
    {
        int device = -1;
        cudaGetDevice(&device);

        cudaMemPrefetchAsync(data, size * sizeof(T), device, NULL);
        cudaMemPrefetchAsync(&size, sizeof(int), device, NULL);
    }

    __host__ __device__
    T& operator[](int pos) const { return data[pos]; }

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

    CuArray<float> knotsX;
    CuArray<float> knotsY;
    CuArray<float> dydxs;
    CuArray<int> cells;

    SplineParams() {}

    SplineParams(CuArray<float> knotsX_, CuArray<float> knotsY_, CuArray<float> dydxs_, CuArray<int> cells_)
                 : knotsX(knotsX_), knotsY(knotsY_), dydxs(dydxs_), cells(cells_) {}

    void prefetch()
    {
        knotsX.prefetch();
        knotsY.prefetch();
        dydxs.prefetch();
        cells.prefetch();
    }

};

// Basically straight from the Laura++ implementation
// Only a loop with trip count nKnots, probably not worth putting it on the GPU
// Fit for y eventually, here just assume they are given.
std::vector<float> calculateGrads(std::vector<float> &x, std::vector<float> &y)
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
void evalSplineKern(const int n, const CuArray<float> * xs, const SplineParams * params, CuArray<float> * splineVals)
{
    // All threads handle blockDim.x * gridDim.x
    // consecutive elements (interleaved partitioning)

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride){

        int cell = params->cells[i];

        // // Try and avoid this on GPU if possible -> precompute and save?
        // while( (*xs)[i] > params->knotsX[cell+1] ) {
        //     ++cell;
        // }

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
                   std::vector<float> * masses,
                   std::vector<int> * cells)
{
    int n = masses->size();
    int nKnots = knotsX->size();

    SplineParams * splineParams = new SplineParams();

    splineParams->knotsX = knotsX;
    splineParams->knotsY = knotsY;
    splineParams->dydxs = dydxs;
    splineParams->cells = cells;

    CuArray<float> * xs = new CuArray<float>(masses);

    std::vector<float> * splineValsV = new std::vector<float>;
    splineValsV->resize(n);

    std::fill(splineValsV->begin(), splineValsV->end(), 0.0);

    CuArray<float> * splineVals = new CuArray<float>(splineValsV);

    splineParams->prefetch();
    xs->prefetch();
    splineVals->prefetch();

    int blockSize = 512;
    int numBlocks = (n + blockSize - 1) / blockSize;

    {
    // boost::timer::auto_cpu_timer t;

    evalSplineKern<<<numBlocks, blockSize>>>(n, xs, splineParams, splineVals);

    splineParams->sync();
    xs->sync();
    splineVals->sync();

    } // Timer


    std::cout << (*splineVals)[0] << std::endl;

    delete splineParams;
}

std::vector<float> calcSplineCPU(std::vector<float> & knotsX,
                   std::vector<float> & knotsY,
                   std::vector<float> & dydxs,
                   std::vector<float> & masses,
                   std::vector<int> & cells)
{

    std::vector<float> splineVals(masses.size());

    std::cout << splineVals.size() << std::endl;

    for (int i = 0; i < splineVals.size(); i++) {

        int cell = cells[i];

        // while( masses[i] > knotsX[cell+1] ) {
        //     ++cell;
        // }

        float xLow  = knotsX[cell];
        float xHigh = knotsX[cell+1];
        float yLow  = knotsY[cell];
        float yHigh = knotsY[cell+1];

        float t = (masses[i] - xLow) / (xHigh - xLow);
        float a = dydxs[cell] * (xHigh - xLow) - (yHigh - yLow);
        float b = -1. * dydxs[cell+1] * (xHigh - xLow) + (yHigh - yLow);

        splineVals[i] = (1 - t) * yLow + t * yHigh + t * (1 - t) * ( a * (1 - t) + b * t );

    }

    return splineVals;
}

std::vector<int> calculateCells(std::vector<float> & masses, std::vector<float> & knotsX)
{
    std::vector<int> cells(masses.size());

    for (int i = 0; i < cells.size(); i++) {

        int cell = 0;
        while( masses[i] > knotsX[cell+1] ) {
            cell++;
        }

        cells[i] = cell;
    }

    return cells;
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

	int n = 100000;
    int nKnots = 30;

	std::vector<float> data(n);

	for (auto & d : data) d = normal(generator);

    std::vector<float> * knotsX = new std::vector<float>(nKnots);
    std::vector<float> * knotsY = new std::vector<float>(nKnots);
    std::vector<float> * masses = new std::vector<float>(n);

    float startKnot = -3.0;
    float endKnot = 3.0;
    float stepSize = (endKnot - startKnot) / nKnots;

    for (int i = 0; i < nKnots; i++) {
        (*knotsX)[i] = startKnot + i * stepSize;
        (*knotsY)[i] = (n / nKnots) * normalDist(0.0, 1.0, (*knotsX)[i]);
    }

    std::vector<int> cells = calculateCells(*masses, *knotsX);

    std::vector<float> dydxs = calculateGrads(*knotsX, *knotsY);

    {
    // boost::timer::auto_cpu_timer t;
    calcSplineGPU(knotsX, knotsY, &dydxs, &data, &cells);
    // calcSplineCPU(*knotsX, *knotsY, dydxs, data, cells);
    }

    return 0;
}
