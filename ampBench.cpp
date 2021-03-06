// Compare SoA, AoS, and Eigen (with expression parser)
// Ideal situation: everything known at compile time
//
// g++ ampBench.cpp -O3 -march=native -std=c++17 -ffast-math -I/usr/local/opt/eigen/include/eigen3 -I/usr/local/Cellar/boost/1.72.0/include/boost/ -Rpass-missed=loop-vectorize -lboost_timer -o ampBench
//

// SoA:
//  1.167841s wall, 1.100000s user + 0.060000s system = 1.160000s CPU (99.3%)
//
// SoA (array):
//  0.637546s wall, 0.570000s user + 0.060000s system = 0.630000s CPU (98.8%)
//
// SoA (stack):
//  0.926747s wall, 0.870000s user + 0.060000s system = 0.930000s CPU (100.4%)
//
// AoS:
//  14.228890s wall, 12.890000s user + 1.260000s system = 14.150000s CPU (99.4%)
//
// AoS (stack):
//  1.105483s wall, 0.910000s user + 0.180000s system = 1.090000s CPU (98.6%)
//
// Eigen:
//  1.072285s wall, 0.930000s user + 0.140000s system = 1.070000s CPU (99.8%)

#include <iostream>
#include <vector>
#include <array>
#include <memory>

#include <Eigen/Dense>

#include <boost/timer/timer.hpp>

// #include <blaze/math/StaticVector.h>
// #include <blaze/math/DynamicVector.h>

#include <blaze/Math.h>

#include "xtensor/xtensor.hpp"
#include "xtensor/xfixed.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"

using blaze::StaticVector;
using blaze::DynamicVector;

static const int n_global = 1E6;

// Spin 5 - no need for template here
float legFunc(float cosHel)
{
    return -32.0*(63.0*cosHel*cosHel*cosHel*cosHel*cosHel - 70.0*cosHel*cosHel*cosHel + 15.0*cosHel)/63.0;
}

// AoS
struct ResParams
{
public:

    float mass;
    float qTerm;
    float ffRatioP;
    float ffRatioR;
    float spinTerms;

    float ampRe;
    float ampIm;

    ResParams()
    : mass(0),
      qTerm(0),
      ffRatioP(0),
      ffRatioR(0),
      spinTerms(0),
      ampRe(0),
      ampIm(0) { }

};

struct ResParamsSoA
{
public:

    float resMass;
    float resWidth;

    std::vector<float> * mass;
    std::vector<float> * qTerm;
    std::vector<float> * ffRatioP;
    std::vector<float> * ffRatioR;
    std::vector<float> * spinTerms;

    std::vector<float> * ampRe;
    std::vector<float> * ampIm;

    // Deal with these guys later...
    static const int spin = 4;
    static const int spinType = 1;

    ResParamsSoA(int s)
    {
        mass = new std::vector<float>(s);
        qTerm = new std::vector<float>(s);
        ffRatioP = new std::vector<float>(s);
        ffRatioR = new std::vector<float>(s);
        spinTerms = new std::vector<float>(s);

        ampRe = new std::vector<float>(s);
        ampIm = new std::vector<float>(s);
    }

    ~ResParamsSoA()
    {
        delete mass;
        delete qTerm;
        delete ffRatioP;
        delete ffRatioR;
        delete spinTerms;

        delete ampRe;
        delete ampIm;
    }

};

struct ResParamsSoAArray
{
public:

    float resMass;
    float resWidth;

    std::array<float, n_global> * mass;
    std::array<float, n_global> * qTerm;
    std::array<float, n_global> * ffRatioP;
    std::array<float, n_global> * ffRatioR;
    std::array<float, n_global> * spinTerms;

    std::array<float, n_global> * ampRe;
    std::array<float, n_global> * ampIm;

    // Deal with these guys later...
    static const int spin = 4;
    static const int spinType = 1;

    ResParamsSoAArray(int s)
    {
        mass = new std::array<float, n_global>();
        qTerm = new std::array<float, n_global>();
        ffRatioP = new std::array<float, n_global>();
        ffRatioR = new std::array<float, n_global>();
        spinTerms = new std::array<float, n_global>();

        ampRe = new std::array<float, n_global>();
        ampIm = new std::array<float, n_global>();
    }

    ~ResParamsSoAArray()
    {
        delete mass;
        delete qTerm;
        delete ffRatioP;
        delete ffRatioR;
        delete spinTerms;

        delete ampRe;
        delete ampIm;
    }

};

struct ResParamsSoAStack
{
public:

    float resMass;
    float resWidth;

    std::vector<float> mass;
    std::vector<float> qTerm;
    std::vector<float> ffRatioP;
    std::vector<float> ffRatioR;
    std::vector<float> spinTerms;

    std::vector<float> ampRe;
    std::vector<float> ampIm;

    // Deal with these guys later...
    static const int spin = 4;
    static const int spinType = 1;

    ResParamsSoAStack(int n) {

        mass.resize(n);
        qTerm.resize(n);
        ffRatioP.resize(n);
        ffRatioR.resize(n);
        spinTerms.resize(n);

        ampRe.resize(n);
        ampIm.resize(n);
    }

    ~ResParamsSoAStack() { }

};

struct ResParamsEigen
{
public:

    float resMass;
    float resWidth;

    // Too large to allocate on stack (?)
    // Possible to do more with static allocation?

    Eigen::Array<float,Eigen::Dynamic,1> mass;
    Eigen::Array<float,Eigen::Dynamic,1> qTerm;
    Eigen::Array<float,Eigen::Dynamic,1> ffRatioP;
    Eigen::Array<float,Eigen::Dynamic,1> ffRatioR;
    Eigen::Array<float,Eigen::Dynamic,1> spinTerms;

    Eigen::Array<float,Eigen::Dynamic,1> ampRe;
    Eigen::Array<float,Eigen::Dynamic,1> ampIm;

    // Deal with these guys later...
    static const int spin = 4;
    static const int spinType = 1;

    ResParamsEigen(int n)
    {

        resMass = 0;
        resWidth = 0;

        mass = Eigen::Array<float,Eigen::Dynamic,1>(n);
        qTerm = Eigen::Array<float,Eigen::Dynamic,1>(n);
        ffRatioP = Eigen::Array<float,Eigen::Dynamic,1>(n);
        ffRatioR = Eigen::Array<float,Eigen::Dynamic,1>(n);
        spinTerms = Eigen::Array<float,Eigen::Dynamic,1>(n);

        ampRe = Eigen::Array<float,Eigen::Dynamic,1>(n);
        ampIm = Eigen::Array<float,Eigen::Dynamic,1>(n);
    }

    ~ResParamsEigen()
    {

    }

};

struct ResParamsBlaze
{
public:

    float resMass;
    float resWidth;

    static const int n = n_global;

    // DynamicVector<float> mass;
    // DynamicVector<float> qTerm;
    // DynamicVector<float> ffRatioP;
    // DynamicVector<float> ffRatioR;
    // DynamicVector<float> spinTerms;
    //
    // DynamicVector<float> ampRe;
    // DynamicVector<float> ampIm;

    StaticVector<float, n> mass;
    StaticVector<float, n> qTerm;
    StaticVector<float, n> ffRatioP;
    StaticVector<float, n> ffRatioR;
    StaticVector<float, n> spinTerms;

    StaticVector<float, n> ampRe;
    StaticVector<float, n> ampIm;

    // Deal with these guys later...
    static const int spin = 4;
    static const int spinType = 1;

    ResParamsBlaze()
    {

        resMass = 0;
        resWidth = 0;

        // mass = DynamicVector<float>(this->n);
        // qTerm = DynamicVector<float>(this->n);
        // ffRatioP = DynamicVector<float>(this->n);
        // ffRatioR = DynamicVector<float>(this->n);
        // spinTerms = DynamicVector<float>(this->n);
        //
        // ampRe = DynamicVector<float>(this->n);
        // ampIm = DynamicVector<float>(this->n);

    }

    ~ResParamsBlaze()
    {

    }

};

struct ResParamsXTensor
{
public:

    float resMass;
    float resWidth;

    static const int n = n_global;

    // DynamicVector<float> mass;
    // DynamicVector<float> qTerm;
    // DynamicVector<float> ffRatioP;
    // DynamicVector<float> ffRatioR;
    // DynamicVector<float> spinTerms;
    //
    // DynamicVector<float> ampRe;
    // DynamicVector<float> ampIm;

    xt::xtensor_fixed<double, xt::xshape<n, 1>> mass;
    xt::xtensor_fixed<double, xt::xshape<n, 1>> qTerm;
    xt::xtensor_fixed<double, xt::xshape<n, 1>> ffRatioP;
    xt::xtensor_fixed<double, xt::xshape<n, 1>> ffRatioR;
    xt::xtensor_fixed<double, xt::xshape<n, 1>> spinTerms;

    xt::xtensor_fixed<double, xt::xshape<n, 1>> ampRe;
    xt::xtensor_fixed<double, xt::xshape<n, 1>> ampIm;

    // Deal with these guys later...
    static const int spin = 4;
    static const int spinType = 1;

    ResParamsXTensor()
    {

        resMass = 0;
        resWidth = 0;

        // mass = DynamicVector<float>(this->n);
        // qTerm = DynamicVector<float>(this->n);
        // ffRatioP = DynamicVector<float>(this->n);
        // ffRatioR = DynamicVector<float>(this->n);
        // spinTerms = DynamicVector<float>(this->n);
        //
        // ampRe = DynamicVector<float>(this->n);
        // ampIm = DynamicVector<float>(this->n);

    }

    ~ResParamsXTensor()
    {

    }

};

void calcAmpSoA(const ResParamsSoA & inParams)
{
    float resMass = inParams.resMass;
    float resWidth = inParams.resWidth;

    for (int i = 0; i < inParams.mass->size(); i++) {
        float totWidth = resWidth * inParams.qTerm->at(i);
        totWidth *= (resMass / inParams.mass->at(i));
        totWidth *= inParams.ffRatioP->at(i) * inParams.ffRatioR->at(i);

        float m2 = inParams.mass->at(i) * inParams.mass->at(i);
        float m2Term = resMass * resMass - m2;

        float scale = inParams.spinTerms->at(i);
        scale /= m2Term * m2Term + resMass * resMass * totWidth * totWidth;
        scale *= inParams.ffRatioP->at(i) * inParams.ffRatioR->at(i); // Optional -> template specialise?
        scale *= legFunc(inParams.spinTerms->at(i));

        inParams.ampRe->at(i) = m2Term * scale;
        inParams.ampIm->at(i) = resMass * totWidth * scale;
    }
}

void calcAmpSoA(const ResParamsSoAArray & inParams)
{
    float resMass = inParams.resMass;
    float resWidth = inParams.resWidth;

    #pragma clang loop vectorize(assume_safety)
    for (int i = 0; i < n_global; i++) {
        float totWidth = resWidth * inParams.qTerm->at(i);
        totWidth *= (resMass / inParams.mass->at(i));
        totWidth *= inParams.ffRatioP->at(i) * inParams.ffRatioR->at(i);

        float m2 = inParams.mass->at(i) * inParams.mass->at(i);
        float m2Term = resMass * resMass - m2;

        float scale = inParams.spinTerms->at(i);
        scale /= m2Term * m2Term + resMass * resMass * totWidth * totWidth;
        scale *= inParams.ffRatioP->at(i) * inParams.ffRatioR->at(i); // Optional -> template specialise?
        scale *= legFunc(inParams.spinTerms->at(i));

        inParams.ampRe->at(i) = m2Term * scale;
        inParams.ampIm->at(i) = resMass * totWidth * scale;
    }
}

void calcAmpSoAStack(ResParamsSoAStack & inParams)
{
    float resMass = inParams.resMass;
    float resWidth = inParams.resWidth;

    for (int i = 0; i < inParams.mass.size(); i++) {
        float totWidth = resWidth * inParams.qTerm.at(i);
        totWidth *= (resMass / inParams.mass.at(i));
        totWidth *= inParams.ffRatioP.at(i) * inParams.ffRatioR.at(i);

        float m2 = inParams.mass.at(i) * inParams.mass.at(i);
        float m2Term = resMass * resMass - m2;

        float scale = inParams.spinTerms.at(i);
        scale /= m2Term * m2Term + resMass * resMass * totWidth * totWidth;
        scale *= inParams.ffRatioP.at(i) * inParams.ffRatioR.at(i); // Optional -> template specialise?
        scale *= legFunc(inParams.spinTerms.at(i));

        inParams.ampRe.at(i) = m2Term * scale;
        inParams.ampIm.at(i) = resMass * totWidth * scale;
    }
}

void calcAmpAoS(const std::vector<std::unique_ptr<ResParams> > & inParams, float resMass, float resWidth)
{

    for (int i = 0; i < inParams.size(); i++) {
        float totWidth = resWidth * inParams.at(i)->qTerm;
        totWidth *= (resMass / inParams.at(i)->mass);
        totWidth *= inParams.at(i)->ffRatioP * inParams.at(i)->ffRatioR;

        float m2 = inParams.at(i)->mass * inParams.at(i)->mass;
        float m2Term = resMass * resMass - m2;

        float scale = inParams.at(i)->spinTerms;
        scale /= m2Term * m2Term + resMass * resMass * totWidth * totWidth;
        scale *= inParams.at(i)->ffRatioP * inParams.at(i)->ffRatioR; // Optional -> template specialise?
        scale *= legFunc(inParams.at(i)->spinTerms);

        inParams.at(i)->ampRe = m2Term * scale;
        inParams.at(i)->ampIm = resMass * totWidth * scale;
    }
}

void calcAmpAoSStack(std::vector<ResParams> & inParams, float resMass, float resWidth)
{

    for (int i = 0; i < inParams.size(); i++) {
        float totWidth = resWidth * inParams.at(i).qTerm;
        totWidth *= (resMass / inParams.at(i).mass);
        totWidth *= inParams.at(i).ffRatioP * inParams.at(i).ffRatioR;

        float m2 = inParams.at(i).mass * inParams.at(i).mass;
        float m2Term = resMass * resMass - m2;

        float scale = inParams.at(i).spinTerms;
        scale /= m2Term * m2Term + resMass * resMass * totWidth * totWidth;
        scale *= inParams.at(i).ffRatioP * inParams.at(i).ffRatioR; // Optional -> template specialise?
        scale *= legFunc(inParams.at(i).spinTerms);

        inParams.at(i).ampRe = m2Term * scale;
        inParams.at(i).ampIm = resMass * totWidth * scale;
    }
}

void calcAmpEigen(ResParamsEigen & inParams)
{
    float resMass = inParams.resMass;
    float resWidth = inParams.resWidth;

    Eigen::Array<float,Eigen::Dynamic,1> totWidth = resWidth * (inParams.qTerm);
    totWidth *= (resMass / (inParams.mass));
    totWidth *= totWidth * (inParams.ffRatioP) * (inParams.ffRatioR);

    Eigen::Array<float,Eigen::Dynamic,1> m2 = (inParams.mass) * (inParams.mass);
    Eigen::Array<float,Eigen::Dynamic,1> m2Term = resMass * resMass - m2;

    Eigen::Array<float,Eigen::Dynamic,1> scale = (inParams.spinTerms);
    scale /= m2Term * m2Term + resMass * resMass * totWidth * totWidth;
    scale *= (inParams.ffRatioP) * (inParams.ffRatioR);

    inParams.ampRe = m2Term * scale;
    inParams.ampIm = resMass * totWidth * scale;
}

void calcAmpBlaze(ResParamsBlaze & inParams)
{
    float resMass = inParams.resMass;
    float resWidth = inParams.resWidth;

    StaticVector<float, n_global> totWidth = resWidth * (inParams.qTerm);
    totWidth *= (resMass / (inParams.mass));
    totWidth *= totWidth * (inParams.ffRatioP) * (inParams.ffRatioR);

    StaticVector<float, n_global> m2 = (inParams.mass) * (inParams.mass);
    StaticVector<float, n_global> m2Term = resMass * resMass - m2;

    StaticVector<float, n_global> scale = (inParams.spinTerms);
    scale /= m2Term * m2Term + resMass * resMass * totWidth * totWidth;
    scale *= (inParams.ffRatioP) * (inParams.ffRatioR);

    inParams.ampRe = m2Term * scale;
    inParams.ampIm = resMass * totWidth * scale;
}

void calcAmpXTensor(ResParamsXTensor & inParams)
{
    float resMass = inParams.resMass;
    float resWidth = inParams.resWidth;

    xt::xtensor_fixed<double, xt::xshape<n_global, 1>> totWidth = resWidth * (inParams.qTerm);
    totWidth *= (resMass / (inParams.mass));
    totWidth *= totWidth * (inParams.ffRatioP) * (inParams.ffRatioR);

    xt::xtensor_fixed<double, xt::xshape<n_global, 1>> m2 = (inParams.mass) * (inParams.mass);
    xt::xtensor_fixed<double, xt::xshape<n_global, 1>> m2Term = resMass * resMass - m2;

    xt::xtensor_fixed<double, xt::xshape<n_global, 1>> scale = (inParams.spinTerms);
    scale /= m2Term * m2Term + resMass * resMass * totWidth * totWidth;
    scale *= (inParams.ffRatioP) * (inParams.ffRatioR);

    inParams.ampRe = m2Term * scale;
    inParams.ampIm = resMass * totWidth * scale;
}

void benchAoS()
{
    static const int n = n_global;

    std::vector<std::unique_ptr<ResParams>> resParamsAoS;
    resParamsAoS.resize(n);

    for (auto & p : resParamsAoS) {

        p = std::make_unique<ResParams>();

        p->qTerm = 0.05;
        p->mass = 0.3;
        p->ffRatioP = 3.3;
        p->ffRatioR = 1.0;
        p->spinTerms = 1.0;

        p->ampRe = 1.0;
        p->ampIm = 1.0;
    }

    float resMass = 1.0;
    float resWidth = 0.1;

    calcAmpAoS(resParamsAoS, resMass, resWidth);

    // std::cout << resParamsAoS.at(5)->ampRe << std::endl;

}

void benchAoSStack()
{
    int n = n_global;

    std::vector<ResParams> resParamsAoSStack;
    resParamsAoSStack.reserve(n);

    for (int i = 0; i < n; i++) {

        ResParams params;

        params.qTerm = 0.05;
        params.mass = 0.3;
        params.ffRatioP = 3.3;
        params.ffRatioR = 1.0;
        params.spinTerms = 1.0;

        params.ampRe = 1.0;
        params.ampIm = 1.0;

        resParamsAoSStack.push_back(params);

    }

    float resMass = 1.0;
    float resWidth = 0.1;

    calcAmpAoSStack(resParamsAoSStack, resMass, resWidth);

    // std::cout << resParamsAoSStack.at(5).ampRe << std::endl;
    // std::cout << resParamsAoSStack.at(5).ampIm << std::endl;
}

void benchSoA()
{
    ResParamsSoA parsR(n_global);

    std::random_device r;

    std::default_random_engine e1(r());
    std::uniform_real_distribution<float> uniform_dist(0, 1);

    std::fill(parsR.qTerm->begin(), parsR.qTerm->end(), uniform_dist(e1));
    std::fill(parsR.mass->begin(), parsR.mass->end(), uniform_dist(e1));
    std::fill(parsR.ffRatioP->begin(), parsR.ffRatioP->end(), uniform_dist(e1));
    std::fill(parsR.ffRatioR->begin(), parsR.ffRatioR->end(), uniform_dist(e1));
    std::fill(parsR.spinTerms->begin(), parsR.spinTerms->end(), uniform_dist(e1));

    std::fill(parsR.ampRe->begin(), parsR.ampRe->end(), uniform_dist(e1));
    std::fill(parsR.ampIm->begin(), parsR.ampIm->end(), uniform_dist(e1));

    parsR.resMass =uniform_dist(e1);
    parsR.resWidth =uniform_dist(e1);

    calcAmpSoA(parsR);

    // std::cout << (parsR.ampRe)->at(5) << std::endl;
    // std::cout << (parsR.ampIm)->at(5) << std::endl;
}

void benchSoAArray()
{
    ResParamsSoAArray parsR(n_global);

    std::random_device r;

    std::default_random_engine e1(r());
    std::uniform_real_distribution<float> uniform_dist(0, 1);

    std::fill(parsR.qTerm->begin(), parsR.qTerm->end(), uniform_dist(e1));
    std::fill(parsR.mass->begin(), parsR.mass->end(), uniform_dist(e1));
    std::fill(parsR.ffRatioP->begin(), parsR.ffRatioP->end(), uniform_dist(e1));
    std::fill(parsR.ffRatioR->begin(), parsR.ffRatioR->end(), uniform_dist(e1));
    std::fill(parsR.spinTerms->begin(), parsR.spinTerms->end(), uniform_dist(e1));

    std::fill(parsR.ampRe->begin(), parsR.ampRe->end(), uniform_dist(e1));
    std::fill(parsR.ampIm->begin(), parsR.ampIm->end(), uniform_dist(e1));

    parsR.resMass = uniform_dist(e1);
    parsR.resWidth = uniform_dist(e1);

    calcAmpSoA(parsR);

    // std::cout << (parsR.ampRe)->at(5) << std::endl;
    // std::cout << (parsR.ampIm)->at(5) << std::endl;
}

void benchSoAStack()
{
    ResParamsSoAStack parsR(n_global);

    std::random_device r;

    std::default_random_engine e1(r());
    std::uniform_real_distribution<float> uniform_dist(0, 1);

    std::fill(parsR.qTerm.begin(), parsR.qTerm.end(), uniform_dist(e1));
    std::fill(parsR.mass.begin(), parsR.mass.end(), uniform_dist(e1));
    std::fill(parsR.ffRatioP.begin(), parsR.ffRatioP.end(), uniform_dist(e1));
    std::fill(parsR.ffRatioR.begin(), parsR.ffRatioR.end(), uniform_dist(e1));
    std::fill(parsR.spinTerms.begin(), parsR.spinTerms.end(), uniform_dist(e1));

    std::fill(parsR.ampRe.begin(), parsR.ampRe.end(), uniform_dist(e1));
    std::fill(parsR.ampIm.begin(), parsR.ampIm.end(), uniform_dist(e1));

    parsR.resMass = uniform_dist(e1);
    parsR.resWidth = uniform_dist(e1);

    calcAmpSoAStack(parsR);

    // std::cout << parsR.ampRe.at(5) << std::endl;
    // std::cout << parsR.ampIm.at(5) << std::endl;
}

void benchEigen()
{
    int n = n_global;
    ResParamsEigen parsR(n);

    parsR.qTerm = Eigen::ArrayXf::Ones(n, 1);
    parsR.mass = Eigen::ArrayXf::Ones(n, 1);
    parsR.ffRatioP = Eigen::ArrayXf::Ones(n, 1);
    parsR.ffRatioR = Eigen::ArrayXf::Ones(n, 1);
    parsR.spinTerms = Eigen::ArrayXf::Ones(n, 1);

    parsR.ampRe = Eigen::ArrayXf::Ones(n, 1);
    parsR.ampIm = Eigen::ArrayXf::Ones(n, 1);

    parsR.resMass = 1.0;
    parsR.resWidth = 0.1;

    calcAmpEigen(parsR);

    // std::cout << (*parsR.ampRe)[5] << std::endl;
    // std::cout << (*parsR.ampIm)[5] << std::endl;
}

void benchBlaze()
{
    ResParamsBlaze parsR;

    std::random_device r;

    std::default_random_engine e1(r());
    std::uniform_real_distribution<float> uniform_dist(0, 1);

    parsR.qTerm = uniform_dist(e1);
    parsR.mass = uniform_dist(e1);
    parsR.ffRatioP = uniform_dist(e1);
    parsR.ffRatioR = uniform_dist(e1);
    parsR.spinTerms = uniform_dist(e1);

    parsR.ampRe = uniform_dist(e1);
    parsR.ampIm = uniform_dist(e1);

    parsR.resMass = uniform_dist(e1);
    parsR.resWidth = uniform_dist(e1);

    calcAmpBlaze(parsR);

    // std::cout << parsR.ampRe[5] << std::endl;
    // std::cout << parsR.ampIm[5] << std::endl;
}

void benchXTensor()
{
    ResParamsXTensor parsR;

    std::random_device r;

    std::default_random_engine e1(r());
    std::uniform_real_distribution<float> uniform_dist(0.0, 1.0);

    parsR.qTerm.fill(uniform_dist(e1));
    parsR.mass.fill(uniform_dist(e1));
    parsR.ffRatioP.fill(uniform_dist(e1));
    parsR.ffRatioR.fill(uniform_dist(e1));
    parsR.spinTerms.fill(uniform_dist(e1));

    parsR.ampRe.fill(uniform_dist(e1));
    parsR.ampIm.fill(uniform_dist(e1));

    parsR.resMass = uniform_dist(e1);
    parsR.resWidth = uniform_dist(e1);

    calcAmpXTensor(parsR);

    // std::cout << parsR.ampRe[5] << std::endl;
    // std::cout << parsR.ampIm[5] << std::endl;
}

int main(int argc, char const *argv[]) {

    int n_itr = 100;

    std::cout<< "SoA:" << std::endl;
    {
    boost::timer::auto_cpu_timer t;
    for (int i = 0; i < n_itr; i++) { benchSoA(); }
    }

    std::cout << std::endl;

    std::cout<< "SoA (array):" << std::endl;
    {
    boost::timer::auto_cpu_timer t;
    for (int i = 0; i < n_itr; i++) { benchSoAArray(); }
    }

    std::cout << std::endl;

    std::cout<< "SoA (stack):" << std::endl;
    {
    boost::timer::auto_cpu_timer t;
    for (int i = 0; i < n_itr; i++) { benchSoAStack(); }
    }

    std::cout << std::endl;

    std::cout<< "AoS:" << std::endl;
    {
    boost::timer::auto_cpu_timer t;
    for (int i = 0; i < n_itr; i++) { benchAoS(); }
    }

    std::cout << std::endl;

    std::cout<< "AoS (stack):" << std::endl;
    {
    boost::timer::auto_cpu_timer t;
    for (int i = 0; i < n_itr; i++) { benchAoSStack(); }
    }

    std::cout << std::endl;

    // Without Legendre term

    std::cout<< "Eigen:" << std::endl;
    {
    boost::timer::auto_cpu_timer t;
    for (int i = 0; i < n_itr; i++) { benchEigen(); }
    }

    // Segfaults for some reason?

    // std::cout << std::endl;
    //
    // std::cout<< "Blaze:" << std::endl;
    // {
    // boost::timer::auto_cpu_timer t;
    // for (int i = 0; i < n_itr; i++) { benchBlaze(); }
    // }
    //
    // std::cout << std::endl;
    //
    // std::cout<< "XTensor:" << std::endl;
    // {
    // boost::timer::auto_cpu_timer t;
    // for (int i = 0; i < n_itr; i++) { benchXTensor(); }
    // }

    return 0;
}
