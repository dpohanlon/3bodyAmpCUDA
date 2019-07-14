#include <iostream>
#include <vector>

template <int I>
struct Int2Type
{
  enum { value = I };
};

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

enum Spin { SPIN0, SPIN1, SPIN2, SPIN3, SPIN4, SPIN5 };
enum SpinFactor { ZEMACH, COVARIANT, LEGENDRE };

struct SpinTermParams
{
public:

	std::vector<float> * p;
	std::vector<float> * q;
	std::vector<float> * erm;
	std::vector<float> * cosHel;
	std::vector<float> * leg;
	std::vector<float> * spinTerms;

	// Deal with these guys later...
	static const int spin = 4;
	static const int spinType = 1;

	SpinTermParams(int s)
	{
		p = new std::vector<float>(s);
		q = new std::vector<float>(s);
		erm = new std::vector<float>(s);
		cosHel = new std::vector<float>(s);
		leg = new std::vector<float>(s);
		spinTerms = new std::vector<float>(s);
	}

	~SpinTermParams()
	{
		delete p;
		delete q;
		delete erm;
		delete cosHel;
		delete leg;
		delete spinTerms;
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

class KernelParamsL : public Managed
{

public:

	FloatArr cosHel;
	FloatArr leg;

	KernelParamsL() {}

	KernelParamsL(FloatArr cosHel_, FloatArr leg_) : cosHel(cosHel_), leg(leg_) {}

	void prefetch()
	{
		cosHel.prefetch();
		leg.prefetch();
	}

};

class KernelParams : public Managed
{

public:

	FloatArr cosHel;
	FloatArr leg;
	FloatArr p;
	FloatArr q;
	FloatArr erm;
	FloatArr spinTerms;

	KernelParams() {}

	void prefetch()
	{
		cosHel.prefetch();
		leg.prefetch();
		p.prefetch();
		q.prefetch();
		spinTerms.prefetch();
        if (erm.size > 0) erm.prefetch();
	}

};

// Spin functions

template<typename Spin>
__host__ __device__
float legFunc(float cosHel)
{
    return 1.0;
}

template<>
__host__ __device__
float legFunc<Int2Type<SPIN0>>(float cosHel)
{
    return 1.0;
}

template<>
__host__ __device__
float legFunc<Int2Type<SPIN1>>(float cosHel)
{
    return -2.0 * cosHel;
}

template<>
__host__ __device__
float legFunc<Int2Type<SPIN2>>(float cosHel)
{
    return 4.0*(3.0*cosHel*cosHel - 1.0)/3.0;
}

template<>
__host__ __device__
float legFunc<Int2Type<SPIN3>>(float cosHel)
{
    return -8.0*(5.0*cosHel*cosHel*cosHel - 3.0*cosHel)/5.0;
}

template<>
__host__ __device__
float legFunc<Int2Type<SPIN4>>(float cosHel)
{
    return 16.0*(35.0*cosHel*cosHel*cosHel*cosHel - 30.0*cosHel*cosHel + 3.0)/35.0;
}

template<>
__host__ __device__
float legFunc<Int2Type<SPIN5>>(float cosHel)
{
    return -32.0*(63.0*cosHel*cosHel*cosHel*cosHel*cosHel - 70.0*cosHel*cosHel*cosHel + 15.0*cosHel)/63.0;
}

// Cov factors

template<typename Spin>
__host__ __device__
float covFactor(float erm)
{
    return 1.0;
}

template<>
__host__ __device__
float covFactor<Int2Type<SPIN0>>(float erm)
{
    return 1.0;
}

template<>
__host__ __device__
float covFactor<Int2Type<SPIN1>>(float erm)
{
    return erm;
}

template<>
__host__ __device__
float covFactor<Int2Type<SPIN2>>(float erm)
{
    return erm*erm + 0.5;
}

template<>
__host__ __device__
float covFactor<Int2Type<SPIN3>>(float erm)
{
    return erm*(erm*erm + 1.5);
}

template<>
__host__ __device__
float covFactor<Int2Type<SPIN4>>(float erm)
{
    return (8.*erm*erm*erm*erm + 24.*erm*erm + 3.)/35.;
}


template<typename Spin>
__global__
void legKern(const int n, KernelParamsL * params)
{

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride){
        params->leg[i] = legFunc<Spin>(params->cosHel[i]);
    }

}

template<typename Spin>
__global__
void spinTermZemachKern(const int n, KernelParams * params)
{

	// Get an instance of our Int2Type type, so that
    // s.value is out integer spin (the enum value)

    Spin s;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride){

        params->leg[i] = legFunc<Spin>(params->cosHel[i]);

        float pProd = params->p[i] * params->q[i];

        params->spinTerms[i] = params->leg[i] * pow(pProd, s.value);
    }

}

template<typename Spin>
__global__
void spinTermCovKern(const int n, KernelParams * params)
{

	// Get an instance of our Int2Type type, so that
    // s.value is out integer spin (the enum value)

    Spin s;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride){

        params->leg[i] = legFunc<Spin>(params->cosHel[i]);

        float pProd = params->p[i] * params->q[i];

        params->spinTerms[i] = params->leg[i] * pow(pProd, s.value) * covFactor<Spin>(params->erm[i]);
    }

}

void calcLegendrePolyManaged(const SpinTermParams & inParams)
{

	int n = inParams.cosHel->size();

	KernelParamsL * kParams = new KernelParamsL();

	kParams->cosHel = inParams.cosHel;
	kParams->leg = inParams.leg;

	kParams->prefetch();

	int blockSize = 128;
	int numBlocks = (n + blockSize - 1) / blockSize;

	legKern<Int2Type<inParams.spin>><<<numBlocks, blockSize>>>(n, kParams);

	kParams->sync();

	inParams.leg->insert(inParams.leg->begin(), &kParams->leg[0], &kParams->leg[n]);

	delete kParams;
}

void calcSpinTerm(const SpinTermParams & inParams)
{

	int n = inParams.cosHel->size();

	bool covariant = inParams.spinType == COVARIANT;

    std::cout << covariant << std::endl;

	KernelParams * kParams = new KernelParams();

	kParams->cosHel = inParams.cosHel;
	kParams->leg = inParams.leg;
	kParams->p = inParams.p;
	kParams->q = inParams.q;
	kParams->spinTerms = inParams.spinTerms;
	if (covariant) kParams->erm = inParams.erm;

	kParams->prefetch();

	int blockSize = 128;
	int numBlocks = (n + blockSize - 1) / blockSize;

	if (!covariant) {
		spinTermZemachKern<Int2Type<inParams.spin>><<<numBlocks, blockSize>>>(n, kParams);
	} else {
		spinTermCovKern<Int2Type<inParams.spin>><<<numBlocks, blockSize>>>(n, kParams);
	}

	kParams->sync();

	inParams.leg->insert(inParams.leg->begin(), &kParams->leg[0], &kParams->leg[n]);
	inParams.spinTerms->insert(inParams.spinTerms->begin(), &kParams->spinTerms[0], &kParams->spinTerms[n]);

	delete kParams;
}

void calcSpinTermCPU(const SpinTermParams & inParams)
{

    for (int i = 0; i < inParams.cosHel->size(); i++) {
        inParams.leg->at(i) = legFunc<Int2Type<inParams.spin>>(inParams.cosHel->at(i));
		float pProd = inParams.p->at(i) * inParams.q->at(i);
		inParams.spinTerms->at(i) = inParams.leg->at(i) * pow(pProd, inParams.spin) * covFactor<Int2Type<inParams.spin>>(inParams.erm->at(i));
    }
}

int main(int argc, char const *argv[]) {

    SpinTermParams pars(int(1E8));

	// Things to consider:
	// floats -> keep an eye on the precision
	// Might want to keep some things in GPU memory (e.g., spin terms) for further calculations

	std::fill(pars.cosHel->begin(), pars.cosHel->end(), 0.2);
	std::fill(pars.q->begin(), pars.q->end(), 0.05);
	std::fill(pars.p->begin(), pars.p->end(), 0.003);
	std::fill(pars.erm->begin(), pars.erm->end(), 3.3);
	std::fill(pars.leg->begin(), pars.leg->end(), 1.0);
	std::fill(pars.spinTerms->begin(), pars.spinTerms->end(), 1.0);

	// calcSpinTerm(pars);
	calcSpinTermCPU(pars);

	std::cout << (pars.leg)->at(5) << std::endl;
	std::cout << (pars.spinTerms)->at(5) << std::endl;

	return 0;
}
