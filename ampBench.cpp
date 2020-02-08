// Compare SoA, AoS, and Eigen (with expression parser)
// Ideal situation: everything known at compile time
//
// g++ ampBench.cpp -O3 -ffast-math -I/usr/local/opt/eigen/include/eigen3 -I/usr/local/Cellar/boost/1.72.0/include/boost/ -lboost_timer -o ampBench
//
// SoA:
//  0.755422s wall, 0.750000s user + 0.000000s system = 0.750000s CPU (99.3%)
//
// SoA (stack):
//  0.548469s wall, 0.540000s user + 0.010000s system = 0.550000s CPU (100.3%)
//
// AoS:
//  13.221095s wall, 12.510000s user + 0.680000s system = 13.190000s CPU (99.8%)
//
// AoS (stack):
//  0.497928s wall, 0.490000s user + 0.000000s system = 0.490000s CPU (98.4%)
//
// Eigen:
//  1.702296s wall, 1.160000s user + 0.530000s system = 1.690000s CPU (99.3%)

#include <iostream>
#include <vector>
#include <memory>

#include <Eigen/Dense>

#include <boost/timer/timer.hpp>

// #include <blaze/math/StaticVector.h>
// #include <blaze/math/DynamicVector.h>

#include <blaze/Math.h>

using blaze::StaticVector;
using blaze::DynamicVector;

static const int n_global = 1E5;

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

    (inParams.ampRe) = m2Term * scale;
    (inParams.ampIm) = resMass * totWidth * scale;
}

void calcAmpBlaze(ResParamsBlaze & inParams)
{
    float resMass = inParams.resMass;
    float resWidth = inParams.resWidth;

    DynamicVector<float> totWidth = resWidth * (inParams.qTerm);
    totWidth *= (resMass / (inParams.mass));
    totWidth *= totWidth * (inParams.ffRatioP) * (inParams.ffRatioR);

    DynamicVector<float> m2 = (inParams.mass) * (inParams.mass);
    DynamicVector<float> m2Term = resMass * resMass - m2;

    DynamicVector<float> scale = (inParams.spinTerms);
    scale /= m2Term * m2Term + resMass * resMass * totWidth * totWidth;
    scale *= (inParams.ffRatioP) * (inParams.ffRatioR);

    (inParams.ampRe) = m2Term * scale;
    (inParams.ampIm) = resMass * totWidth * scale;
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

    std::fill(parsR.qTerm->begin(), parsR.qTerm->end(), 0.05);
    std::fill(parsR.mass->begin(), parsR.mass->end(), 0.3);
    std::fill(parsR.ffRatioP->begin(), parsR.ffRatioP->end(), 3.3);
    std::fill(parsR.ffRatioR->begin(), parsR.ffRatioR->end(), 1.0);
    std::fill(parsR.spinTerms->begin(), parsR.spinTerms->end(), 1.0);

    std::fill(parsR.ampRe->begin(), parsR.ampRe->end(), 1.0);
    std::fill(parsR.ampIm->begin(), parsR.ampIm->end(), 1.0);

    parsR.resMass = 1.0;
    parsR.resWidth = 0.1;

    calcAmpSoA(parsR);

    // std::cout << (parsR.ampRe)->at(5) << std::endl;
    // std::cout << (parsR.ampIm)->at(5) << std::endl;
}

void benchSoAStack()
{
    ResParamsSoAStack parsR(n_global);

    std::fill(parsR.qTerm.begin(), parsR.qTerm.end(), 0.05);
    std::fill(parsR.mass.begin(), parsR.mass.end(), 0.3);
    std::fill(parsR.ffRatioP.begin(), parsR.ffRatioP.end(), 3.3);
    std::fill(parsR.ffRatioR.begin(), parsR.ffRatioR.end(), 1.0);
    std::fill(parsR.spinTerms.begin(), parsR.spinTerms.end(), 1.0);

    std::fill(parsR.ampRe.begin(), parsR.ampRe.end(), 1.0);
    std::fill(parsR.ampIm.begin(), parsR.ampIm.end(), 1.0);

    parsR.resMass = 1.0;
    parsR.resWidth = 0.1;

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

    // parsR.qTerm = 1.3;
    // parsR.mass = 1.0;
    // parsR.ffRatioP = 1.3;
    // parsR.ffRatioR = 1.2;
    // parsR.spinTerms = 1.8;
    //
    // parsR.ampRe = 1.1;
    // parsR.ampIm = 1.2;
    //
    // parsR.resMass = 1.2;
    // parsR.resWidth = 0.3;

    // std::cout << parsR.qTerm << std::endl;

    calcAmpBlaze(parsR);

    // std::cout << (*parsR.ampRe)[5] << std::endl;
    // std::cout << (*parsR.ampIm)[5] << std::endl;
}

int main(int argc, char const *argv[]) {

    int n_itr = 100;

    std::cout<< "SoA:" << std::endl;
    {
    boost::timer::auto_cpu_timer t;
    for (int i = 0; i < n_itr; i++) { benchSoA(); }
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

    std::cout<< "Eigen:" << std::endl;
    {
    boost::timer::auto_cpu_timer t;
    for (int i = 0; i < n_itr; i++) { benchEigen(); }
    }

    std::cout << std::endl;

    std::cout<< "Blaze:" << std::endl;
    {
    boost::timer::auto_cpu_timer t;
    for (int i = 0; i < n_itr; i++) { benchBlaze(); }
    }

    return 0;
}
