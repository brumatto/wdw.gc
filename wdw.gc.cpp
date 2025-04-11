/**
  * @file wdw.gc.cpp
  * @brief Arquivo para o método iterativo baseado em Gradiente Conjugado que resolve o sistema linear do método Cranck-Nicholson para o modelo FRW-escalar quântico do universo.
  * @author Hamilton José Brumatto
  * @since 16/08/2022
  * @version 1.0 (2022)
  */

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <complex>
#include <omp.h>
#include <cstring>
#include "SMmatrix.hpp"
#include "GCsolve.hpp"
#include "wdw.gc.hpp"

using namespace std;

double potential(double a, double phi, double lambda, double mass) {
  double a2 = a*a;
  double phi2 = phi*phi;
  return (3*a2 - lambda*a2*a2 - phi2/2 - mass*mass*a2*phi2/2);
}


int main(int argc, char **args) {
  int Na, Nphi, Nt, nproc=0, pt;
  double inftya, inftyphi, inftyt, lastT, Da, Dphi, Dt, t0, Ea, Ephi;
  double lambda, mass;
  complex<double> ra(0,0), rphi(0,0), ic(0,1);
  vector<complex<double>> m, b;
  vector<double> Vef;
  double err, emax, avmax;
  double Psi2, Psi2b, Psi2a, Psi2in, amed, amedb, ameda, amedin, phimed, phimedb, phimeda, phimedin, phi2med, phi2medb, phi2meda, phi2medin;
  int imax, iter;
  sm_matrix<complex<double>> M, B, Mt, K;
  vector<complex<double>> Bpsi, psi;
  int s;
  FILE *inifile;
  char linha[120];
  string variavel, sres, arqout, resfile;
  char filename[50], dir[20];
  bool goon = true, resume = false;

  if(argc == 1) inifile = fopen("wdw.ini","rt");  // Primeiro lemos os parâmetros do arquivo de iniciação.
  else inifile = fopen(args[1],"rt");
  if(inifile != NULL) {
    while(!feof(inifile)) {
        fgets(linha,120,inifile);
        string slinha = linha;
        auto i = find_if_not(slinha.begin(),slinha.end(),[](char c){ return c==' ';});
        if(isalpha(*i)) {
            stringstream iss(slinha);
            iss >> variavel;
            // Parâmetros de escala
            if(variavel =="Na") iss >> Na;
            else if(variavel == "Nphi")  iss >> Nphi;
            else if(variavel == "a")     iss >> inftya;
            else if(variavel == "phi")   iss >> inftyphi;
            // Vef
            else if(variavel == "Lambda")iss >> lambda;
            else if(variavel == "m")     iss >> mass;
            else if(variavel == "aVmax") iss >> avmax;
            // \Psi inicial
            else if(variavel == "Ea")    iss >> Ea;
            else if(variavel == "Ephi")  iss >> Ephi;
            // Evolução temporal
            else if(variavel == "Tlast")     iss >> lastT;
            else if(variavel == "T")     iss >> inftyt;
            else if(variavel == "Nt")    iss >> Nt;
            else if(variavel == "Print") iss >> pt;
            // Parâmetros para iterações e versões
            else if(variavel == "Emax")  iss >> emax;
            else if(variavel == "Imax")  iss >> imax;
            else if(variavel == "file")  iss >> arqout;
            else if(variavel == "Nproc") iss >> nproc;
            else if(variavel == "BaseDir") iss >> dir;
            else if(variavel == "Resume") iss >> sres;
            else if(variavel == "rest") iss >> t0;
            else if(variavel == "resfile") iss >> resfile;
        }
    }
    if(nproc == 0) {
      nproc = omp_get_num_procs();
    }
    if (sres == "true") resume = true;
    resfile="/"+resfile+".dat";
    resfile=dir+resfile;
    fclose(inifile);
    cerr << "Resume: " << resfile << " with " << nproc << " threads" << endl;

    omp_set_dynamic(0);     // Não há escolha dinâmica
    omp_set_num_threads(nproc); // Número explícito de threads
    Dt = inftyt/Nt;
    Da = inftya/Na;
    Dphi = 2*inftyphi/Nphi;
    ra = -Dt/(24.0*ic*Da*Da);
    rphi = Dt/(4.0*ic*Dphi*Dphi);
    Vef = vector<double>((Na+1)*(Nphi+1));
    m = vector<complex<double>>((Na+1)*(Nphi+1));
    b = vector<complex<double>>((Na+1)*(Nphi+1));
    #pragma omp parallel for shared(Vef, m, b, Da, Dphi, Dt, ic, ra, rphi)
    for(int i = 0; i <= Na; i++) {
      double ai = Da*i;
      for(int j = 0; j <= Nphi; j++) {
        double phij = (Dphi*j-inftyphi);
        Vef[j*(Na+1)+i] = potential(ai,phij,lambda,mass);
        m[j*(Na+1)+i] = 1.0+2.0*ra+2.0*rphi+ic*Dt*Vef[j*(Na+1)+i]/2.0;
        b[j*(Na+1)+i] = 1.0-2.0*ra-2.0*rphi-ic*Dt*Vef[j*(Na+1)+i]/2.0;
      }
    }
    M = createMatrix(Na,Nphi,ra,rphi,m);
    B = createMatrix(Na,Nphi,-ra,-rphi,b);
    Mt = mMtc(M); cerr << "Matriz transposta calculada" << endl;
    K = mMtM(M,Na); cerr << "K = M*M calculado" << endl;
    s = 0;
    if(!resume) {
      t0 = 0;
      psi = createPsi0(Na,Nphi,inftya,inftyphi,Ea,Ephi);
      sprintf(filename,"%s/%s%06.3lf.dat",dir,arqout.c_str(),0.0);
      ofstream datfile(filename,ios::out | ios::binary);
      if(!datfile.is_open()) {
        cerr << "Could not open dat file" << endl;
        cerr << strerror(errno) << endl;
        goon = false;
      }
      else {
        for(int i = 0; i < (Na+1)*(Nphi+1); i++) {
          double real = psi[i].real();
          double imag = psi[i].imag();
          datfile.write(reinterpret_cast<char *>(&real),sizeof(double));
          datfile.write(reinterpret_cast<char *>(&imag),sizeof(double));
        }
        if(datfile.is_open()) datfile.close();
      }
    }
    else {
      psi = vector<complex<double>>((Na+1)*(Nphi+1));
      ifstream r(resfile.c_str(), ios::in | ios::binary);
      if(!r.is_open()) {
        cerr << "Error --> Could not open resume file" << endl;
        goon = false;
      }
      else {
        for(int i = 0; (i < (Na+1)*(Nphi+1)) && r.is_open(); i++) {
            double real, imag;
            r.read(reinterpret_cast<char *>(&real),sizeof(double));
            r.read(reinterpret_cast<char *>(&imag),sizeof(double));
            psi[i]= complex<double>(real,imag);
        }
      }	
      if(r.is_open()) r.close();
    }
//    time(&solving);
    printf("t\tPsi2\tPsi2_a\tPsi2_b\tPsi2_in\tamed\tamed_a\tamed_b\tamed_in\tphimed\tphimed_a\tphimed_b\tphimed_in\tphi2m\tphi2m_a\tphi2m_b\tphi2m_in\n");
    for(; s < (Nt*(lastT-t0)/inftyt) && goon; s++) {
      double tnow = (s+1)*inftyt/Nt+t0;
      amedb = ameda = amedin = phimedb = phimeda = phimedin = 0;
      phi2medb = phi2meda = phi2medin = Psi2b = Psi2a = Psi2in = 0;
      #pragma omp parallel for shared(Da, Dphi, inftyphi) reduction(+:Psi2b, Psi2a, Psi2in, amedb, ameda, amedin, phimedb, phimeda, phimedin, phi2medb, phi2meda, phi2medin)
      for(int j = 0; j <= Nphi; j++)
        for(int i = 0; i <= Na; i++) {
          double p2 = abs(psi[j*(Na+1)+i])*abs(psi[j*(Na+1)+i])*Da*Dphi;
          if(Da*i < avmax) {
            Psi2b+=p2;
            amedb+=p2*Da*i;
            phimedb+=p2*(Dphi*j-inftyphi);
            phi2medb+= p2*(Dphi*j-inftyphi)*(Dphi*j-inftyphi);
          }
          else {
            Psi2a+=p2;
            ameda+=p2*Da*i;
            phimeda+=p2*(Dphi*j-inftyphi);
            phi2meda+= p2*(Dphi*j-inftyphi)*(Dphi*j-inftyphi);
          }
          if(Vef[j*(Na+1)+i] > 0) {
            Psi2in+=p2;
            amedin+=p2*Da*i;
            phimedin+=p2*(Dphi*j-inftyphi);
            phi2medin+=p2*(Dphi*j-inftyphi)*(Dphi*j-inftyphi);
          }
        } cerr << "Médias calculadas" << endl;
      Psi2 = Psi2a+Psi2b;
      amed = (ameda+amedb)/Psi2;
      amedb/=Psi2b; ameda = (Psi2a>0 ? ameda/Psi2a : 0); amedin= (Psi2in > 0 ? amedin/Psi2in : 0);
      phimed = (phimeda+phimedb)/Psi2;
      phimedb/=Psi2b; phimeda = (Psi2a>0 ? phimeda/Psi2a : 0); phimedin= (Psi2in > 0 ? phimedin/Psi2in : 0);
      phi2med = (phi2meda+phi2medb)/Psi2;
      phi2medb/=Psi2b; phi2meda = (Psi2a>0 ? phi2meda/Psi2a : 0); phi2medin= (Psi2in > 0 ? phi2medin/Psi2in : 0);
      printf("%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n",tnow,Psi2,Psi2a,Psi2b,Psi2in, amed,ameda,amedb,amedin, phimed,phimeda,phimedb,phimedin, phi2med,phi2meda,phi2medb,phi2medin);
      Bpsi = sm_vmultMatrix(B,psi);
      iter = imax;
      cerr << "Entrando no GCSolve..." << endl;
      err = GCsolve(M,psi,Bpsi,Mt,K,iter,emax,Na,Nphi);
      cerr << "Iteração " << s << " .. err = " << err << "  com "<< iter << " iterações" << endl;
      if(err > emax) {
        cerr << "Iteração " << s << " não atingiu a precisão" << endl;
        goon = false;
      }
      if((s+1)%pt==0 && goon) {
        sprintf(filename,"%s/%s%06.3lf.dat",dir,arqout.c_str(),tnow);
        ofstream datfile(filename,ios::out | ios::binary);
        if(!datfile.is_open()) {
          cerr << "Could not open dat file" << endl;
          cerr << strerror(errno) << endl;
          goon = false;
        }
        else {
          for(int i = 0; i < (Na+1)*(Nphi+1); i++) {
            double real = psi[i].real();
            double imag = psi[i].imag();
            datfile.write(reinterpret_cast<char *>(&real),sizeof(double));
            datfile.write(reinterpret_cast<char *>(&imag),sizeof(double));
          }
          if(datfile.is_open()) datfile.close();
        }
      }
    }
  } else {
    cout << "Ini file not found: wdw.ini" << endl;
    cout << "or no ini file provided: wdw.gc <file.ini>" << endl;
  }
  return 0;
}
