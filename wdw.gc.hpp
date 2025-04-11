/**
  * @file wdw.gc.hpp
  * @brief Arquivo para o método iterativo baseado em Gradiente Conjugado que resolve o sistema linear do método Cranck-Nicholson para o modelo FRW-escalar quântico do universo.
  * @author Hamilton José Brumatto
  * @since 11/07/2022
  * @version 1.0 (2022)
  */


#ifndef __WDW_GC_HPP__
#define __WDW_GC_HPP__

#include <complex>
#include "SMmatrix.hpp"

using namespace std;

/**
  * @fn vector<complex> *createPsi0(int Na, int Nphi, double a, double phi, double Ea, double Ephi)
  * @brief Cria um vetor de B inicial para a função de estado Psi usando Cranck-Nicolson com tamanho (Ma+1)x(Mphi+1)
  * @brief \Psi(a,\varphi) = \dfrac{2^{15/4}}{\sqrt{\pi}}E^{3/4}_aE^{1/4}_{\varphi}a\exp\left(-4E_aa^2 - 2E_{\varphi}\varphi^2\right)
  * @param *Na número de pontos na dimensão a
  * @param *Nphi número de pontos na dimensão y
  * @param a infinito numérico da coordenada a (0 .. \infty)
  * @param phi infinito numérico da coordenada phi (-\infty ..\infty)
  * @param Ea Energia média na coordenada a
  * @param Ephi Energia média na coordenada phi
  * @return um vector com esta configuração
  */
vector<complex<double>> createPsi0(int, int, double, double, double, double);

/**
  * @fn sm_matrix<complex<double>> createMatrix(int Na,int Nphi,complex ra, complex rb, vector<complex<double>> a)
  * @brief Cria uma matriz (Mx+1)(My+1) x (Mx+1)(My+1) dos sistema linear de diferenças finitas usando Cranck-Nicolson
  * @param *Mx número de pontos na dimensão x
  * @param *My número de pontos na dimensão y
  * @param alfa diagonal principal
  * @param betax (-betax) diagonal superior e inferior à principal
  * @param betay (-betay) diagonal distante My superior e inferior à principal
  * @return uma matriz esparsa com esta configuração
  */
sm_matrix<complex<double>> createMatrix(int,int,complex<double>,complex<double>,vector<complex<double>>);

/**
  * @fn sm_matrix<complex<double>> mMtc(sm_matrix<complex<double>> &m)
  * @brief mMtc retorna a transposta conjugada da matriz M complexa
  * @param m, uma matriz esparsa nxn complexa
  * @return a transposta conjugada.
  */
sm_matrix<complex<double>> mMtc(sm_matrix<complex<double>> &);

/**
  * @fn sm_matrix<complex<double>> mMtM(sm_matrix<complex<double>> &m, int Na)
  * @brief mMtM retorna o produto M*M onde M* é matriz conjugada complexa de M
  * @param m, uma matriz esparsa nxn complexa pentadiagonal com elementos nas posições: (i,i-(Na+1))..(i,i-1)..(i,i)..(i,i+1)..(i,i+(Na+1))
  * @param Na espaço das diagonais externas.
  * @return a matriz hermitiana M*M
  */
sm_matrix<complex<double>> mMtM(sm_matrix<complex<double>> &, int);

#endif //__FRW_ESCALAR_HPP__
