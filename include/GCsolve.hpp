#include<vector>
#include<complex>
#include<omp.h>
#include "SMmatrix.hpp"

using namespace std;

/**
  * @fn vector<complex<double>> calcF(sm_matrix<complex<double>> &A, vector<complex<double>> &x, vector<complex<double>> &b)
  * @brief Transforma o sistema Ax=b em uma função f(x) = Ax-b cuja raiz é a solução do sistema.
  * @param A uma matriz esparsa representando o sistema de coeficientes
  * @param x um vetor de variáveis do sistema
  * @param b um vetor de termos independentes
  * @param err a norma do vetor resultante
  * @return um vetor representando f(x) no ponto x
  */
vector<complex<double>> calcF(sm_matrix<complex<double>> &, vector<complex<double>> &, vector<complex<double>> &, double &);


/**
  * @fn complex<double> calcvtu(vector<complex<double>> &v, vector<complex<double>> &u)
  * @brief Calcula o produto interno v.u
  * @param v primeiro vetor do produto interno
  * @param u segundo vetor do produto interno do mesmo tamanho do primeiro.
  * @return O produto interno.
  */
complex<double> calcvtu(vector<complex<double>> &, vector<complex<double>> &);

/**
  * @fn double GCsolve(sm_matrix<complex<double>> &M, vector<complex<double>> &x, vector<complex<double>> &b, sm_matrix<complex<double>> &Mt, sm_matrix<complex<double>> &K, int &imax, double emax, int Na, int Nphi)
  * @brief Calcula a solução do sistema linear Ax = b através do método dos gradientes conjugados para obter a raiz de Ax-b=0. Alfa = r(k)Tp(k)/p(k)TAp(k)
  * @param M uma matriz esparsa de elementos complexos representando o sistema de coeficientes
  * @param x um vetor de variáveis complexas
  * @param b um vetor fonte de elementos complexos
  * @param Mt é a matriz transposta conjugada de M: M*
  * @param K é a matriz hermitiana positiva M*M
  * @param imax o número máximo de iteraçoes para o algoritmo caso não convirja
  * @param emax o teto para a precisão desejada, seja na norma do resíduo ou na diferença entre soluções de iterações
  * @param NA número de pontos para o escalar a
  * @param NPhi número de pontos para o escalar phi
  * @return o erro máximo obtido na norma ou diferença entre iterações.
  */
double GCsolve(sm_matrix<complex<double>> &, vector<complex<double>> &, vector<complex<double>> &, sm_matrix<complex<double>> &, sm_matrix<complex<double>> &, int &, double, int, int);


