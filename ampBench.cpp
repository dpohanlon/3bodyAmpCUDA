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

    // static const int n = 1E8;

    Eigen::Array<float,Eigen::Dynamic,1> * mass;
    Eigen::Array<float,Eigen::Dynamic,1> * qTerm;
    Eigen::Array<float,Eigen::Dynamic,1> * ffRatioP;
    Eigen::Array<float,Eigen::Dynamic,1> * ffRatioR;
    Eigen::Array<float,Eigen::Dynamic,1> * spinTerms;

    Eigen::Array<float,Eigen::Dynamic,1> * ampRe;
    Eigen::Array<float,Eigen::Dynamic,1> * ampIm;

    // Deal with these guys later...
    static const int spin = 4;
    static const int spinType = 1;

    ResParamsEigen(int n)
    {

        resMass = 0;
        resWidth = 0;

        mass = new Eigen::Array<float,Eigen::Dynamic,1>(n);
        qTerm = new Eigen::Array<float,Eigen::Dynamic,1>(n);
        ffRatioP = new Eigen::Array<float,Eigen::Dynamic,1>(n);
        ffRatioR = new Eigen::Array<float,Eigen::Dynamic,1>(n);
        spinTerms = new Eigen::Array<float,Eigen::Dynamic,1>(n);

        ampRe = new Eigen::Array<float,Eigen::Dynamic,1>(n);
        ampIm = new Eigen::Array<float,Eigen::Dynamic,1>(n);
    }

    ~ResParamsEigen()
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

void calcAmpEigen(const ResParamsEigen & inParams)
{
    float resMass = inParams.resMass;
    float resWidth = inParams.resWidth;

    Eigen::Array<float,Eigen::Dynamic,1> totWidth = resWidth * (*inParams.qTerm);
    totWidth *= (resMass / (*inParams.mass));
    totWidth *= totWidth * (*inParams.ffRatioP) * (*inParams.ffRatioR);

    Eigen::Array<float,Eigen::Dynamic,1> m2 = (*inParams.mass) * (*inParams.mass);
    Eigen::Array<float,Eigen::Dynamic,1> m2Term = resMass * resMass - m2;

    Eigen::Array<float,Eigen::Dynamic,1> scale = (*inParams.spinTerms);
    scale /= m2Term * m2Term + resMass * resMass * totWidth * totWidth;
    scale *= (*inParams.ffRatioP) * (*inParams.ffRatioR);

    (*inParams.ampRe) = m2Term * scale;
    (*inParams.ampIm) = resMass * totWidth * scale;
}

void benchAoS()
{
    boost::timer::auto_cpu_timer t;

    static const int n = int(1E8);

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
    boost::timer::auto_cpu_timer t;

    int n = int(1E8);

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
    boost::timer::auto_cpu_timer t;

    ResParamsSoA parsR(int(1E8));

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
    boost::timer::auto_cpu_timer t;

    ResParamsSoAStack parsR(int(1E8));

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
    boost::timer::auto_cpu_timer t;

    int n = int(1E8);
    ResParamsEigen parsR(n);

    *parsR.qTerm = Eigen::ArrayXf::Ones(n, 1);
    *parsR.mass = Eigen::ArrayXf::Ones(n, 1);
    *parsR.ffRatioP = Eigen::ArrayXf::Ones(n, 1);
    *parsR.ffRatioR = Eigen::ArrayXf::Ones(n, 1);
    *parsR.spinTerms = Eigen::ArrayXf::Ones(n, 1);

    *parsR.ampRe = Eigen::ArrayXf::Ones(n, 1);
    *parsR.ampIm = Eigen::ArrayXf::Ones(n, 1);

    parsR.resMass = 1.0;
    parsR.resWidth = 0.1;

    calcAmpEigen(parsR);

    // std::cout << (*parsR.ampRe)[5] << std::endl;
    // std::cout << (*parsR.ampIm)[5] << std::endl;
}

int main(int argc, char const *argv[]) {
    std::cout<< "SoA:" << std::endl;

    for (int i = 0; i < 10; i++) { benchSoA(); }

    std::cout << std::endl;
    std::cout<< "SoA (stack):" << std::endl;

    for (int i = 0; i < 10; i++) { benchSoAStack(); }

    std::cout << std::endl;
    std::cout<< "AoS:" << std::endl;

    for (int i = 0; i < 10; i++) { benchAoS(); }

    std::cout << std::endl;
    std::cout<< "AoS (stack):" << std::endl;

    for (int i = 0; i < 10; i++) { benchAoSStack(); }

    std::cout << std::endl;
    std::cout<< "Eigen:" << std::endl;

    for (int i = 0; i < 10; i++) { benchEigen(); }

    return 0;
}
