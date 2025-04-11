/**
  * @file createMatrix.cpp
  * @brief Arquivo contendo uma rotina de inicialização de uma matriz
  * esparsa definida em SMmatrix.hpp para solução de um modelo quantico
  * do universo para FRW escalar
  * @author Hamilton José Brumatto
  * @since 11/07/2022
  * @version 1.0 (2022)
  * @version 1.5 (2023) Corrigido psi(a,\infty) = 0 ==> (i > = Npi*(Na+1)) não havia "="
  */

#include <complex>
#include <cmath>
#include <omp.h>
#include "SMmatrix.hpp"

using namespace std;

vector<complex<double>> createPsi0(int Na,int Nphi, double a, double phi, double Ea, double Ephi) {
  int i, j;
  vector<complex<double>> b((Na+1)*(Nphi+1),0);
  #pragma omp parallel for private(i, j) shared(b)
  for(i = 1; i < Na; i++){
    double ai = a*i*1.0/Na;
    for(j = 1; j < Nphi; j++) {
       double phij = 2*phi*j*1.0/Nphi-phi;
       b[j*(Na+1)+i] = pow(2,15.0/4)*pow(Ea,0.75)*pow(Ephi,0.25)*ai*exp(-4*Ea*ai*ai-2*Ephi*phij*phij)/sqrt(M_PI);
    }
  }
  return b;
}


sm_matrix<complex<double>> createMatrix(int Na,int Nphi,complex<double> ra,complex<double> rphi, vector<complex<double>> a) {
  sm_matrix<complex<double>> m((Na+1)*(Nphi+1));
  int i;
  
  #pragma omp parallel for private(i) shared(m, Na, Nphi, ra, rphi, a) schedule(guided,(int) sqrt((Na+1)*(Nphi+1)))
  for(i = 0; i < (Na+1)*(Nphi+1); i++) {
    //Condições de contorno:
    if(i < Na+1) m.setCell(i,i,1); // psi(a,-\infty) = 0;
    else if(i >= Nphi*(Na+1)) m.setCell(i,i,1); // psi(a,\infty) = 0;
    else if(i%(Na+1)==0) m.setCell(i,i,1); // psi(0,phi) = 0;
    else if(i%(Na+1)==Na) m.setCell(i,i,1); // psi(\infty,phi) = 0;
    else { // Matriz de Cranck-Nicolson 
      if(abs(a[i])>0) m.setCell(i,i,a[i]);
      m.setCell(i,i+1,-ra);
      m.setCell(i,i-1,-ra);
      m.setCell(i,i+(Na+1),-rphi);
      m.setCell(i,i-(Na+1),-rphi);
    }
  }
  return m;
}


sm_matrix<complex<double>> mMtc( sm_matrix<complex<double>> &m) {
  sm_matrix<complex<double>> t(m.size());
  int i;
  complex<double> val;

  #pragma omp parallel for private(i) shared(m, t) schedule(guided,(int)sqrt(m.size()))
    for(i =0; i < m.size(); i++) {
        auto pa = m.begin(i); auto paend = m.end(i);
        while(pa!=paend) {
           if(abs(pa->second) > ZERO)
             #pragma omp critical
             t.setCell(pa->first,i,conj(pa->second));
           pa++;
      }
    }
  return t;
}


complex<double> iprod(const vector<pair<int,complex<double>>> &vt, const vector<pair<int,complex<double>>> v) {
  complex<double> inner(0,0);
  auto pt = vt.begin();
  auto p = v.begin();
  while(pt != vt.end() && p != v.end()) {
    if(pt->first == p->first) {
      inner+= conj(pt->second)*p->second;
      pt++;
      p++;
    } else if(pt->first < p->first) pt++;
    else p++;
  }
  return inner;
}

sm_matrix<complex<double>> mMtM(sm_matrix<complex<double>> &m, int Na) {
    sm_matrix<complex<double>> prod(m.size());
    #pragma omp parallel for shared(prod,m) schedule(guided, (int)sqrt(m.size()))
      for(int i = 0; i < m.size(); i++) {
        vector<pair<int,complex<double>>> p;;
        auto vt = m.line(i);
        if(i-2*(Na+1) >=0) {
          auto v = m.line(i-2*(Na+1));
          p.push_back(make_pair(i-2*(Na+1),iprod(vt,v)));
        }
        if(i-(Na+2) >=0) {
          auto v = m.line(i-(Na+2));
          p.push_back(make_pair(i-(Na+2),iprod(vt,v)));
        }
        if(i-(Na+1) >=0) {
          auto v = m.line(i-(Na+1));
          p.push_back(make_pair(i-(Na+1),iprod(vt,v)));
        }
        if(i-Na >=0) {
          auto v = m.line(i-Na);
          p.push_back(make_pair(i-Na,iprod(vt,v)));
        }
        if(i-2 >=0) {
          auto v = m.line(i-2);
          p.push_back(make_pair(i-2,iprod(vt,v)));
        }
        if(i-1 >=0) {
          auto v = m.line(i-1);
          p.push_back(make_pair(i-1,iprod(vt,v)));
        }
        p.push_back(make_pair(i,iprod(vt,vt)));
        if(i+1 < m.size()) {
          auto v = m.line(i+1);
          p.push_back(make_pair(i+1,iprod(vt,v)));
        }
        if(i+2 < m.size()) {
          auto v = m.line(i+2);
          p.push_back(make_pair(i+2,iprod(vt,v)));
        }
        if(i+Na < m.size()) {
          auto v = m.line(i+Na);
          p.push_back(make_pair(i+Na,iprod(vt,v)));
        }
        if(i+(Na+1) < m.size()) {
          auto v = m.line(i+(Na+1));
          p.push_back(make_pair(i+(Na+1),iprod(vt,v)));
        }
        if(i+(Na+2) < m.size()) {
          auto v = m.line(i+(Na+2));
          p.push_back(make_pair(i+(Na+2),iprod(vt,v)));
        }
        if(i+2*(Na+1) < m.size()) {
          auto v = m.line(i+2*(Na+1));
          p.push_back(make_pair(i+2*(Na+1),iprod(vt,v)));
        }
        prod.setline(i,p);
      }
    return prod;
}
