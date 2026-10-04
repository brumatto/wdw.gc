#include "GCsolve.hpp"

using namespace std;

vector<complex<double>> calcF(sm_matrix<complex<double>> &A, vector<complex<double>> &x, vector<complex<double>> &b, double &norma) {
  vector<complex<double>> f;
  norma = 0;
  f = sm_vmultMatrix<complex<double>>(A,x);
  #pragma omp parallel for shared(f, b) reduction(max:norma)
  for(int i=0; i < A.size(); i++) {
    f[i]=b[i]-f[i];
    if(abs(f[i]) < ZERO) f[i] = complex<double>(0,0);
    norma = abs(f[i]) < norma ? norma : abs(f[i]);
  }
  return f;
}

complex<double> calcvtu(vector<complex<double>> &v, vector<complex<double>> &u) {
  complex<double> q(0,0), qaux;
  double qr=0, qi=0;
  #pragma omp parallel for shared(v,u) reduction(+:qr,qi) private(qaux)
  for(int i=0; i < (int) v.size(); i++) {
    qaux = (conj(v[i])*u[i]);
    qr += qaux.real();
    qi += qaux.imag();
  }
  q = complex<double>(qr,qi);
  if(abs(q) < ZERO) q = complex<double>(0,0);
  return q;
}

// H = I, K = A*A, N = K
double GCsolve(sm_matrix<complex<double>> &M, vector<complex<double>> &x, vector<complex<double>> &b, sm_matrix<complex<double>> &Mt, sm_matrix<complex<double>> &K, int &imax, double emax, int Na, int Nphi) {
  double err, eaux, norma, psim;
  vector<complex<double>> res0, res1(x.size()), g, p, Kp, Kg, Mp;
  complex<double> alpha, beta, den;
  int cont;

  res0 = calcF(M,x,b,norma);
  g = sm_vmultMatrix<complex<double>>(Mt,res0);
  p = sm_vmultMatrix<complex<double>>(K,g);
  cont = 0; cerr << "Iniciando iterações..." << endl;
  while(norma > emax && cont < imax) {
    cont++;
    Kp = sm_vmultMatrix<complex<double>>(K,p);
    Mp = sm_vmultMatrix<complex<double>>(M,p);
    den = calcvtu(p,Kp);
    alpha = calcvtu(g,p)/den;
    norma = 0;
    #pragma omp parallel for private(eaux) shared(res0, res1, alpha) schedule(guided,(int) sqrt(res0.size())) reduction(max:norma)
    for(int i = 0; i < (int) res0.size(); i++) {
      res1[i] = res0[i] - alpha*Mp[i];
      eaux = abs(res1[i]-res0[i]);
      norma = norma > eaux ? norma : eaux;
    }
    err = 0;
    psim = 0;
    g = sm_vmultMatrix<complex<double>>(Mt,res1);
    Kg = sm_vmultMatrix<complex<double>>(K,g);
    beta = -calcvtu(Kp,Kg)/den;
    #pragma omp parallel for shared(alpha, p, x, Kg) reduction(max:err,psim) private(eaux)
    for(int i = 0; i < (int) p.size(); i++) {
      eaux = abs(alpha*p[i]);
      err = err > eaux ? err : eaux;
      if(i/(Na+1) == 0 || i/(Na+1) == Nphi || i%(Na+1) == 0 || i%(Na+1) == Na)
        x[i] = complex<double>(0,0);
      else
        x[i] = x[i] + alpha*p[i];
      if(abs(x[i]) > psim) psim = abs(x[i]);
      if(abs(x[i]) < ZERO) x[i] = complex<double>(0,0);
      p[i] = Kg[i] + beta*p[i];
      if(abs(res1[i]) > 0) res0[i]=res1[i];
      else res0[i] = complex<double>(0,0);
    }
    cerr << "norma = " << norma << "; err = " << err << "; Psi_m " << psim << endl;
  }
  imax = cont;
  return norma;
}

