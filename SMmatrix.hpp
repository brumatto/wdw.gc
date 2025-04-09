/**
  * @file SMmatrix.hpp
  * @brief Arquivo contendo uma classe de uma matriz esparsa quadrada e os métodos que manipulam esta matriz. Os elementos são definidos pelo template T
  *
  * @author Hamilton José Brumatto
  * @since 16/1/2013
  * @version 1.0 (2013) - C
  * @version 1.1 (2022) - C paralelizado
  * @version 2.0 (2022) - C++ paralelizado
  * @version 2.1 (2023) - Comentários corrigidos
  */

#ifndef __SM_MATRIX_HPP__
#define __SM_MATRIX_HPP__

#define ZERO (1e-300)
#include <utility>
#include <vector>
#include <cassert>
#include <complex>
#include <omp.h>
#include <iostream>
using namespace std;

/** Uma célula da matriz é um pair<int,T>
 *  cada célula carrega seu índice 'j' como primeiro elemento do pair
 *  cada célula contém também um valor associado, como segundo elemento do pair
 *  se o valor da célula for 0, esta deve ser removida da estrutura.
 *  um vetor de células é uma linha da matriz
 **/


/** A estrutura sm_matrix permite construir um vetor de linhas de células
  * Uma célula (i,j) da matriz tem o seu índice 'i' (linha) como sendo o índice do vetor de linhas.
  * o índice 'j' da célula (coluna) é obtido dentro da célula, no vetor de células que forma uma linha.
  **/

template <class T>
class  sm_matrix{
    int n;
    vector< vector <pair<int,T> > >m;

public:
    sm_matrix() {n = 0;}
    sm_matrix(int n);
    int size() ;
    void setCell(int,int,T);
    T getCell(int,int) ;
    void swapLine(int, int);
    vector<pair<int,T>> &line(int) ;
    void setline(int, vector<pair<int,T>>);
    typename vector<pair<int,T>>::iterator begin(int) ;
    typename vector<pair<int,T>>::iterator end(int) ;
};


/**
  * @fn sm_matrix (int n)
  * @brief Função que inicia na memória uma matriz esparsa de tamanho n.
  * @param n Tamanho da Matriz, supõe-se possível n x n
  */
template<class T>
sm_matrix<T>::sm_matrix(int n): n(n) {
    m = vector<vector< pair<int,T> > >(n);
    #pragma omp parallel for shared(m)
    for(int i=0; i<n; i++)
      m[i] = vector< pair<int,T> >();
}

/**
  * @fn int sm_matrix<T>::size()
  * @brief Retorna o tamanho n da matriz quadrada esparsa nxn
  * @return A dimensão da matriz;
  */

template<class T>
int sm_matrix<T>::size()  {
  return n;
}

/**
  * @fn void sm_setCell (int i, int j, T value)
  * @brief Função que altera o valor de uma célula, criando uma célula se for nula ou removendo se vier a ser nula.
  * @param i Linha da célula a ser alterada.
  * @param j Coluna da célula a ser alterada.
  * @param value Valor a ser atribuído à célula (somente células não nulas são mantidas).
  */
template<class T>
void sm_matrix<T>::setCell(int i, int j, T value) {
    if(i>=0 && j>=0 && i<n && j<n)		// Somente i e j válidos são aceitos 0 <= i,j < n
    {
        auto c = m[i].begin();
        while(c!=m[i].end() && c->first<j) c++;	 // A linha não é nula, caminho enquanto indice da célula < j
        if((c==m[i].end() || c->first > j) && abs(value) >= ZERO) // Célula não existe e valor é não nulo
        {
            if(c!=m[i].end()) m[i].insert(c,pair<int,T>(j,value));
            else m[i].push_back(pair<int,T>(j,value));
        }
        else
	    {
            if(c!=m[i].end() && (c->first == j))				// Célula já existe
	        {
                if(abs(value) < ZERO) m[i].erase(c);
                else c->second = value;
	        }
	    }
    }
    return;
}

/**
  * @fn T getCell (int i, int j)
  * @brief Função que retorna um valor para a célula na posição i, j, ou zero se tal célula não existir.
  * @param i Linha da célula consultada.
  * @param j Coluna da célula consultada.
  * @return o valor da célula na posição i,j; ou zero se não existir tal célula em uma matriz esparsa.
  */
template<class T>
T sm_matrix<T>::getCell(int i, int j)  {
    T retval = 0;
    if(i>=0 && j>=0 && i<n && j<n) 		// Se o índice é válido, 0 <= i,j < n
    {
        auto c = m[i].begin();
        while((c!=m[i].end()) && (c->first<j)) c++; 		// Procura o o índice j na linha.
        if(c!=m[i].end() && c->first == j) retval=c->second;		// Se achou o valor de retorno é aceito, caso contrário, 0.
    }
    return retval;
}

template<class T>
vector<pair<int,T>> &sm_matrix<T>::line(int i) {
  assert(i >= 0 && i < n);
  return m[i];
}

template<class T>
void sm_matrix<T>::setline(int i,vector<pair<int,T>> line) {
  assert(i >=0 && i < n);
  m[i] = line;
  return;
}

template<class T>
typename vector<pair<int,T>>::iterator sm_matrix<T>::begin(int i) {
  assert(i >= 0 && i < n);
  return m[i].begin();
}

template<class T>
typename vector<pair<int,T>>::iterator sm_matrix<T>::end(int i) {
  assert(i >= 0 && i < n);
  return m[i].end();
}


/**
  * @fn void sm_matrix::swapLine(int i, int j)
  * @brief Em uma matriz esparsa m, troca as linhas de índices i e j.
  * @param m Matriz a ser manipulada
  * @param i uma das linhas da troca
  * @param j a outra linha da troca
  */
template<class T>
void sm_matrix<T>::swapLine(int i, int j) {
  m[i].swap(m[j]);
  return;
}

/**
  * @fn vector<T> sm_vmultMatrix(sm_matrix<T> &m, vector<T> &b)
  * @brief Faz o produto de uma matriz nxn por um vetor de tamanho n
  * @param m uma matriz esparsa.
  * @param b um vetor.
  * @return Um vetor resultante do produto.
  */

template<class T>
vector<T> sm_vmultMatrix(sm_matrix<T> &m, vector<T> &b) {
  int i;
  T val;
  vector<T> p = vector<T>(m.size(),0);
  #pragma omp parallel for private(i, val) shared(m, b, p) schedule(guided,(int) sqrt(m.size()))
  for(i = 0; i < m.size(); i++) {
    val = 0;
    auto pa = m.begin(i);
    auto fim = m.end(i);
    while(pa!=fim) {
      val +=pa->second * b[pa->first];
      pa++;
    }
    if(abs(val) > ZERO) p[i] = val;
  }
  return p;
}


/**
  * @fn int sm_GJsolve(sm_matrix<T> &m, vector<T> &x,  vector<T> &b, int max, double &e)
  * @brief GJsolve é um resolvedor de sistema linear Ax=b pelo método Gauss-Jacobi
  * @param m, uma matriz esparsa nxn
  * @param x, um vetor de tamanho n com a estimativa inicial da iteração.
  * @param b, um vetor de tamanho n com a fonte do sistema.
  * @param max, é o número máximo de iterações, caso o sistema não convirja rapidamente.
  * @param *e, é a aproximação máxima esperada para os valores de x, queremos: abs(x(k) - x(k-1)) < e, para todos x, o pior valor da aproximação é retornado nesta variável.
  * @return irá devolver a informação se o número máximo de iterações foi atingido.
  */ 
template<class T>
int sm_GJsolve(sm_matrix<T> &m, vector<T> &x, vector<T> &b, int max, double &e) {
    int k;
    T aii;
    double e_max = e+1;
    for(k = 0; e_max > e && k < max; k++) { //< Cada uma das iterações
        e_max = 0;
        #pragma omp parallel for private(aii) shared(x, b) reduction(max:e_max)
        for(int i=0; i<m.size(); i++) { //< cada um dos xi
            T axj = 0;
            auto p = m.begin(i);
            auto fim = m.end(i);
            while(p!=fim) {
              if(p->first == i) aii = p->second;
              else axj+= p->second*x[p->first]; //< axj é a soma aij*xj
              p++;
            }
            T xk = (b[i]-axj)/aii; //< próximo iteração para xi
            if(abs(xk) < ZERO) xk = 0;
            e_max = e_max < abs(xk-x[i]) ? (abs(xk-x[i]) < ZERO ? 0 : abs(xk-x[i])) : e_max; //< diferença na aproximação
        //    if(e_max > 1) cout << "x[" << i << "]=" << x[i] << " :: xk=" << xk << " .. e_max = " << e_max << "...." << endl;
            x[i] = xk;

        }
    }
    e = e_max;
    return k;
}

/**sm_matrix<T> sm_invertMatrix(sm_matrix<T> &m)
  * @brief invertMatrix retorna a matriz inversa. Quando a coluna da identidade é nula, é feita uma troca de linhas para a linha cuja coluna não seja nula.
  * @param m, uma matriz esparsa nxn
  * @return a inversa da matriz.
  */

template<class T>
sm_matrix<T> sm_invertMatrix(sm_matrix<T> &m)
{
  T m_factor, aux;
  int i, j, d;
  sm_matrix<T> I(m.size());

  #pragma omp parallel for
  for(i = 0; i < m.size(); i++) I.setCell(i,i,1); // Esta será a inversa.
  i = 0;
  while(i<m.size())				// Escalonando para o triangulo inferior ficar nulo
  {
    auto p = m.begin(i); auto pend = m.end(i);
    if(i < p->first)	// Elemento da diagonal principal é nula, precisa haver uma troca de linha
    {
      d = i+1;
      while( d < m.size() && i < p->first)
      {
	      p = m.begin(d);
      	d++;
      }
      assert(i >= p->first);
      m.swapLine(i,(d-1));
      I.swapLine(i,(d-1));
      p = m.begin(i); pend = m.end(i);
    }
    while ((j = p->first) < i)		// Para todo m(i,j) não nulo, j < i (anterior à diagonal principal)
    {
       auto r = m.begin(j); auto rend = m.end(j);// r propaga a operação na linha i toda, com a linha j anterior.
       m_factor=p->second;				// O primeiro r é da diagonal principal
							// A operação é para zerar o elemento m(i,j)
       m.setCell(i,p->first,0);
       p = m.begin(i); pend=m.end(i);
       r++;
       while(r!=rend)
       {
      	 if(p==pend)	// Não tinha mais elemento na linha i, então acrescento os da linha j
	         while(r!=rend)
	         {
	           m.setCell(i,r->first,-m_factor*r->second);
	           r++;
	         }
         else						// Tem elemento na linha i,
	       {
	         if(p->first == r->first)			// Tem elemento na mesma coluna na linha j
	         {
	           p->second = p->second - r->second * m_factor;	// Atualizo o valor de m(i,j)
	           r++;
	           if(abs(p->second) < ZERO) {
                 p = m.line(i).erase(p); // Mas se o valor é zero, removo o elemento.
                 pend = m.end(i);
               } else p++;
	         }
      	   else if(p->first > r->first)      // Tem elemento na linha j, mas não tem na mesma coluna na linha i.
	         {
	           m.setCell(i,r->first,-m_factor*r->second); // Então crio o elemento m(i,j).
	           r++;
	         }
	         else {p++;}	// Se para o elemento da linha i, não tem outro na mesma coluna da linha j, faço nada.
         }
       }
       auto v = I.line(j);
       #pragma omp parallel for private(aux) shared(I, v, i)
       for(int k = 0; k < (int) v.size(); k++) {
         aux = I.getCell(i,v[k].first) - v[k].second*m_factor;
         #pragma omp critical
         abs(aux) < 0 ? I.setCell(i,v[k].first,0) : I.setCell(i,v[k].first,aux);
       }
       p = m.begin(i); pend = m.end(i);
    }					// Repito isto enquanto houver m(i,j) != 0, para j < i
    if(i == p->first) {
      m_factor=p->second;
      while(p!=pend)					// Propagando a divisão para o resto da linha
      {
        aux = (p->second/=m_factor);
        if(abs(aux) < ZERO) {
          p = m.line(i).erase(p);
          pend = m.end(i);
        } else {
          p->second = aux;
          p++;
        }
      }
      p = I.begin(i); pend = I.end(i);
      while(p!=pend)					// Propagando a divisão para a linha na matriz inversa
      {
        aux = (p->second/=m_factor);
        if(abs(aux) < ZERO) {
          p = I.line(i).erase(p);
          pend = I.end(i);
        } else {
          p->second = aux;
          p++;
        }
      }
      i++;
    }
  } 					// Agora é só repetir para todas as linhas i < n.

  for(i = m.size()-1; i>=0; i--)      // Calculando a solução na subida
  {
    auto p = m.begin(i); auto pend = m.end(i);			// Como o elemento da diagonal já é unitário, começo com o próximo.
    p++;

    while(p!=pend)			// Não preciso zerar o triângulo superior, somente
    {					// Calcular a inversa.
//      #pragma omp parallel for private(jaux, aux) shared(I, i, p)
      for(int jaux = 0; jaux < I.size(); jaux++) {
        aux = I.getCell(i,jaux) - p->second*I.getCell(p->first,jaux);
        abs(aux) < 0 ? I.setCell(i,jaux,0) : I.setCell(i,jaux,aux);
      }
      p++;
    }
  }
  return I;
}

/**
  * @fn sm_matrix<T> mmultMatrix( sm_matrix<T> &a,  sm_matrix<T> &b)
  * @brief mmultMatrix retorna uma matriz que é o produto das duas matrizes passadas
  * @param a, uma matriz esparsa nxn
  * @param b, uma matriz esparsa nxn
  * @return o produto das matrizes.
  */


template<class T>
sm_matrix<T> sm_mmultMatrix( sm_matrix<T> &a,  sm_matrix<T> &b) {
  sm_matrix<T> m(a.size());
  int i,j;
  T val;

  #pragma omp parallel for private(i, j, val) shared(a, b) schedule(guided,(int) sqrt(m.size()))
    for(i =0; i < a.size(); i++) {
      for(j = 0; j < a.size(); j++) {
        auto pa = a.begin(i); auto paend = a.end(i);
        val = 0;
        while(pa!=paend) {
              auto pb = b.begin(pa->first); auto pbend = b.end(pa->first);
              while(pb!=pbend && pb->first < j) pb++;
              if (pb!=pbend && pb->first == j) val += pa->second * pb->second;
              pa++;
        }
        if(abs(val) > ZERO) m.setCell(i,j,val);
      }
    }
  return m;
}

/* Solução de um sistema linear utilizado o algoritmo de Gauss
 * Para evitar erros de aproximação, é feita troca de linhas para representar a identidade
 * a linha escolhida é a de maior valor na coluna da identidade entre as linhas existente
 */

/**
  * @fn vector<T> sm_gaussSolve( sm_matrix<T> &m,  vector<T> &b)
  * @brief sm_gaussSolve resolve por eliminação de gauss mx = b, retornando x
  * @param a, uma matriz esparsa nxn
  * @param b, um vetor fonte
  * @return um vetor solução
  */

template<class T>
vector<T> sm_gaussSolve(sm_matrix<T> m,const vector<T> &b)
{
  vector<T> a = b;
  T m_factor, aux;
  int i, j, d;

  i = 0;
  while(i<m.size())				// Escalonando para o triangulo inferior ficar nulo
  { cout << i << ".." << flush;
    auto p = m.begin(i); auto pend = m.end(i);
    if(i < p->first)    // Elemento da diagonal principal é nula, precisa haver uma troca de linha
    {
      d = i+1;
      while( d < m.size() && i < p->first)
      {
        p = m.begin(d);
        d++;
      }
      assert(i >= p->first);
      m.swapLine(i,(d-1));
      aux = a[i];
      a[i] = a[d-1];
      a[d-1] = aux;
      p = m.begin(i); pend = m.end(i);
    }
    while ((j = p->first) < i)		// Para todo m(i,j) não nulo, j < i (anterior à diagonal principal)
    {
       auto r = m.begin(j); auto rend = m.end(j);// r propaga a operação na linha i toda, com a linha j anterior.
       m_factor=p->second;	// O primeiro r é da diagonal principal
							// A operação é para zerar o elemento m(i,j)
       m.setCell(i,p->first,0);
       p = m.begin(i); pend = m.end(i);
       r++;
       while(r!=rend)
       {
	     if(p==pend)	// Não tinha mais elemento na linha i, então acrescento os da linha j
	       while(r!=rend)
	       {
	         m.setCell(i,r->first,-m_factor*r->second);
	         r++;
           }
         else						// Tem elemento na linha i,
         {
           if(p->first == r->first)	// Tem elemento na mesma coluna na linha j
           {
             p->second = p->second - r->second * m_factor;	// Atualizo o valor de m(i,j)
	         r++;
             if(abs(p->second) < ZERO) {
               p = m.line(i).erase(p); // Mas se o valor é zero, removo o elemento.
               pend = m.end(i);
             } else p++;
	       }
	       else if(p->first > r->first)   // Tem elemento na linha j, mas não tem na mesma coluna na linha i.
	       {
	         m.setCell(i,r->first,-m_factor*r->second); // Então crio o elemento m(i,j).
	         r++;
	       }
	       else {p++;}			// Se para o elemento da linha i, não tem outro na mesma coluna da linha j, faço nada.
	     }
       }
       a[i]=a[i]-a[j]*m_factor;  		// Por fim, atualizo os valores livres.
       if(abs(a[i]) < ZERO) a[i] = 0;
       p = m.begin(i); pend = m.end(i);
    }					// Repito isto enquanto houver m(i,j) != 0, para j < i
    if(i == p->first) {
      m_factor = p->second;
      while(p!=pend) // Propagando a divisão para o resto da linha
      {
        aux = (p->second/=m_factor);
        if(abs(aux) < ZERO) {
          p = m.line(i).erase(p);
          pend = m.end(i);
        } else {
          p->second = aux;
          p++;
        }
      }
      a[i]/=m_factor;
      if(abs(a[i]) < ZERO) a[i] = 0;
      i++;
    }
  } 					// Agora é só repetir para todas as linhas i < n.

  for(i = m.size()-1; i>=0; i--)      // Calculando a solução na subida
  {
    auto p = m.begin(i); auto pend = m.end(i);
    p++;

    while(p!=pend)			// Não preciso zerar o triângulo superior, somente
    {					// Calcular os coeficientes.
      a[i] = a[i] - p->second*a[p->first];
      p++;
    }
  }
  return a;
}



#endif // __SM_MATRIX_HPP__

