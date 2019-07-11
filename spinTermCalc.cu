#include <iostream>
#include <vector>

template <int I>
struct Int2Type
{
  enum { value = I };
};

enum Spin { SPIN0, SPIN1, SPIN2, SPIN3, SPIN4, SPIN5 };
enum SpinFactor { ZEMACH, COVARIANT, LEGENDRE };

struct SpinTermParams
{
	std::vector<float> * p;
	std::vector<float> * q;
	std::vector<float> * erm;
	std::vector<float> * cosHel;
	std::vector<float> * leg;
	std::vector<float> * spinTerms;

	// Deal with these guys later...
	static const int spin = 0;
	static const int spinType = 0;
};

struct KernelParams
{
	float * p;
	float * q;
	float * erm;
	float * cosHel;
	float * leg;
	float * spinTerms;
};

// Spin functions

template<typename Spin>
__device__
float legFunc(float cosHel)
{
    return 1.0;
}

template<>
__device__
float legFunc<Int2Type<SPIN0>>(float cosHel)
{
    return 1.0;
}

template<>
__device__
float legFunc<Int2Type<SPIN1>>(float cosHel)
{
    return -2.0 * cosHel;
}

template<>
__device__
float legFunc<Int2Type<SPIN2>>(float cosHel)
{
    return 4.0*(3.0*cosHel*cosHel - 1.0)/3.0;
}

template<>
__device__
float legFunc<Int2Type<SPIN3>>(float cosHel)
{
    return -8.0*(5.0*cosHel*cosHel*cosHel - 3.0*cosHel)/5.0;
}

template<>
__device__
float legFunc<Int2Type<SPIN4>>(float cosHel)
{
    return 16.0*(35.0*cosHel*cosHel*cosHel*cosHel - 30.0*cosHel*cosHel + 3.0)/35.0;
}

template<>
__device__
float legFunc<Int2Type<SPIN5>>(float cosHel)
{
    return -32.0*(63.0*cosHel*cosHel*cosHel*cosHel*cosHel - 70.0*cosHel*cosHel*cosHel + 15.0*cosHel)/63.0;
}

// Cov factors

template<typename Spin>
__device__
float covFactor(float erm)
{
    return 1.0;
}

template<>
__device__
float covFactor<Int2Type<SPIN0>>(float erm)
{
    return 1.0;
}

template<>
__device__
float covFactor<Int2Type<SPIN1>>(float erm)
{
    return erm;
}

template<>
__device__
float covFactor<Int2Type<SPIN2>>(float erm)
{
    return erm*erm + 0.5;
}

template<>
__device__
float covFactor<Int2Type<SPIN3>>(float erm)
{
    return erm*(erm*erm + 1.5);
}

template<>
__device__
float covFactor<Int2Type<SPIN4>>(float erm)
{
    return (8.*erm*erm*erm*erm + 24.*erm*erm + 3.)/35.;
}


template<typename Spin>
__global__
void legKern(const int n, KernelParams params)
{

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride){
    	params.leg[i] = legFunc<Spin>(params.cosHel[i]);
    }

}

template<typename Spin>
__global__
void spinTermZemachKern(const int n, KernelParams params)
{

	// Get an instance of our Int2Type type, so that
    // s.value is out integer spin (the enum value)

    Spin s;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride){

    	params.leg[i] = legFunc<Spin>(params.cosHel[i]);

		float pProd = params.p[i] * params.q[i];

		params.spinTerms[i] = params.leg[i] * pow(pProd, s.value);
    }

}

template<typename Spin>
__global__
void spinTermCovKern(const int n, KernelParams params)
{

	// Get an instance of our Int2Type type, so that
    // s.value is out integer spin (the enum value)

    Spin s;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride){

    	params.leg[i] = legFunc<Spin>(params.cosHel[i]);

		float pProd = params.p[i] * params.q[i];

		params.spinTerms[i] = params.leg[i] * pow(pProd, s.value) * covFactor<Spin>(params.erm[i]);
    }

}

void calcLegendrePoly(SpinTermParams inParams)
{

	int n = inParams.cosHel->size();

	KernelParams kParams;

	cudaMallocManaged(&kParams.leg, n * sizeof(float));
    cudaMallocManaged(&kParams.cosHel, n * sizeof(float));

	kParams.cosHel = inParams.cosHel->data();

	int device = -1;

    cudaGetDevice(&device);

	cudaMemPrefetchAsync(kParams.leg, n * sizeof(float), device, NULL);
	cudaMemPrefetchAsync(kParams.cosHel, n * sizeof(float), device, NULL);

	int blockSize = 128;
    int numBlocks = (n + blockSize - 1) / blockSize;

    legKern<Int2Type<inParams.spin>><<<numBlocks, blockSize>>>(n, kParams);

    cudaError_t cudaStatus = cudaDeviceSynchronize();

    if (cudaStatus != cudaSuccess) {
        std::cout << "sync failed" << std::endl;
    }

	inParams.leg->insert(inParams.leg->end(), &kParams.leg[0], &kParams.leg[n]);

}

void calcSpinTerm(SpinTermParams inParams)
{

	int n = inParams.cosHel->size();

	bool covariant = inParams.spinType == COVARIANT;

	KernelParams kParams;

    cudaMallocManaged(&kParams.spinTerms, n * sizeof(float));
	cudaMallocManaged(&kParams.leg, n * sizeof(float));
    cudaMallocManaged(&kParams.cosHel, n * sizeof(float));
	cudaMallocManaged(&kParams.p, n * sizeof(float));
	cudaMallocManaged(&kParams.q, n * sizeof(float));
	if (covariant) cudaMallocManaged(&kParams.erm, n * sizeof(float));

	// Init on device (we can do that as memory is 'unified')

	kParams.cosHel = inParams.cosHel->data();
	kParams.p = inParams.p->data();
	kParams.q = inParams.q->data();
	if (covariant) kParams.erm = inParams.erm->data();

	int device = -1;

    cudaGetDevice(&device);

    cudaMemPrefetchAsync(kParams.spinTerms, n * sizeof(float), device, NULL);
	cudaMemPrefetchAsync(kParams.leg, n * sizeof(float), device, NULL);
	cudaMemPrefetchAsync(kParams.cosHel, n * sizeof(float), device, NULL);
	cudaMemPrefetchAsync(kParams.p, n * sizeof(float), device, NULL);
    cudaMemPrefetchAsync(kParams.q, n * sizeof(float), device, NULL);
	if (covariant) cudaMemPrefetchAsync(kParams.erm, n * sizeof(float), device, NULL);

	int blockSize = 128;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // spinTermKern<Int2Type<SPIN1>, Int2Type<ZEMACH>><<<numBlocks, blockSize>>>(n, cosHel, p, q, out, leg);
	// spinTermKern<Int2Type<spin>, Int2Type<spinType>><<<numBlocks, blockSize>>>(n, cosHel, p, q, out, leg);

	if (!covariant) {
		spinTermZemachKern<Int2Type<inParams.spin>><<<numBlocks, blockSize>>>(n, kParams);
	} else {
		spinTermCovKern<Int2Type<inParams.spin>><<<numBlocks, blockSize>>>(n, kParams);
	}

    cudaError_t cudaStatus = cudaDeviceSynchronize();

    if (cudaStatus != cudaSuccess) {
        std::cout << "sync failed" << std::endl;
    }

	inParams.leg->insert(inParams.leg->end(), &kParams.leg[0], &kParams.leg[n]);
	inParams.spinTerms->insert(inParams.spinTerms->end(), &kParams.spinTerms[0], &kParams.spinTerms[n]);
}

int main(int argc, char const *argv[]) {

	return 0;
}
