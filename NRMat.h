#ifndef _NRA_MATRIX_
#define _NRA_MATRIX_
#pragma once

#include <vector>

namespace nr {

// NRVec, NRMat, NRMat3d, NRMat4d are all borrowed from "Numerical Recipes in C++" by Willam H. Press et al.

// ---------------- NRVec ---------------------

template<typename T>
class NRVec {
private:
  int nn; // size of array. upper index is nn-1
  T *v;
public:
  NRVec();
  NRVec & resize(const int n);
  explicit NRVec(const int n); // zero-based array
  NRVec(const T &a, const int n); // initialize to constant value
  NRVec(const T *a, const int n); // initialize to array
  NRVec(const NRVec &rhs); // copy constructor
  NRVec & operator=(const NRVec &rhs); // assignment
  NRVec & operator=(const T &a); // assign a to every element
  NRVec & operator=(const std::vector<T> &a);
  inline T & operator[](const int i); // i'th element
  inline const T & operator[](const int i) const;
  inline int size() const;
  inline void dealloc();
  inline T * c_buf();
  inline void randu(int seed = 0);


  ~NRVec();
};

template<typename T>
NRVec<T>::NRVec() :
    nn(0), v(0) {
}

template<typename T>
NRVec<T> & NRVec<T>::resize(const int n) {
  if (nn != n) {
    if (v != 0) {
      delete[] (v);
    }
    nn = n;
    v = new T[n];
  }
  return *this;
}

template<typename T>
NRVec<T>::NRVec(const int n) :
    nn(0), v(0) {
  resize(n);
}

template<typename T>
NRVec<T>::NRVec(const T &a, const int n) :
    nn(n), v(new T[n]) {
  for (int i = 0; i < n; i++)
    v[i] = a;
}

template<typename T>
NRVec<T>::NRVec(const T *a, const int n) :
    nn(n), v(new T[n]) {
  for (int i = 0; i < n; i++)
    v[i] = *a++;
}

template<typename T>
NRVec<T>::NRVec(const NRVec<T> &rhs) :
    nn(rhs.nn), v(new T[nn]) {
  for (int i = 0; i < nn; i++)
    v[i] = rhs[i];
}

template<typename T>
NRVec<T> & NRVec<T>::operator=(const NRVec<T> &rhs) {
  // postcondition: normal assignment via copying has been performed;
  //		if vector and rhs were of different sizes, vector has been resized to match the size of rhs
  if (this != &rhs) {
    if (nn != rhs.nn) {
      if (v != 0)
        delete[] (v);
      nn = rhs.nn;
      v = new T[nn];
    }

    for (int i = 0; i < nn; i++)
      v[i] = rhs[i];
  }
  return *this;
}

template<typename T>
NRVec<T> & NRVec<T>::operator=(const std::vector<T> &a) {
  if (nn != a.size()) {
    if (v != 0)
      delete[] (v);
    nn = a.size();
    v = new T[nn];
  }

  for (int i = 0; i < nn; i++)
    v[i] = a[i];
  return *this;
}

template<typename T>
NRVec<T> & NRVec<T>::operator=(const T &a) { // assign a to every element
  for (int i = 0; i < nn; i++)
    v[i] = a;
  return *this;
}

template<typename T>
inline T & NRVec<T>::operator[](const int i) { // subscripting
  return v[i];  // no boundary check?
}

template<typename T>
inline const T & NRVec<T>::operator[](const int i) const { // subscripting
  return v[i];  // no boundary check?
}

template<typename T>
inline int NRVec<T>::size() const {
  return nn;
}

template<typename T>
NRVec<T>::~NRVec() {
  dealloc();
}

template<typename T>
inline void NRVec<T>::dealloc() {
  if (v != 0) {
    delete[] (v);
    v = 0;
    nn = 0;
  }
}

template<typename T>
inline T * NRVec<T>::c_buf() {
  return v;
}

template<typename T>
inline void NRVec<T>::randu(int seed){
  srand(seed);
  for (int i = 0; i < nn; i++)
  {
    v[i] = (T)(1.0 *rand() / RAND_MAX);
  }
}

// ---------------- NRHeap ---------------------
// ---------------- Min Heap -------------------
template<typename T, typename compare>
class NRHeap {
private:
  int nn; // size of heap. upper index is nn-1
  int ne; // size of elem
  int maxsize;
  T *v;
public:
  NRHeap();
  NRHeap & resize(const int n);
  explicit NRHeap(const int n); // zero-based array
  NRHeap(const NRHeap &rhs); // copy constructor
  NRHeap & operator=(const NRHeap &rhs); // assignment
  inline T & operator[](const int i); // i'th element
  inline const T & operator[](const int i) const;
  inline int size() const;
  inline int elemsize() const;
  inline int heapsize() const;
  inline void dealloc();
  inline T * c_buf();
  inline void randu(int seed = 0);

  ~NRHeap();

  bool add_elem(const T &a) {
    if (ne == maxsize) {
      if (compare()(v[0], a) < 0) {
        T tmp = v[0];
        v[0] = a;
        trickleDown(0);
        return true;
      } else {
        return false;
      }
    } else {
      ////assert(ne < maxsize);
      v[ne] = a;
      nn++;
      bubble_up(ne);
      ne++;
      //return 0; //return NULL;
      return true;
    }
  }

  void sort_elem() {
    while (nn > 1) {
      swap(0, nn - 1);
      nn--;
      trickleDown(0);
    }
  }

protected:
  inline int left(int i) {
    return 2 * i + 1;
  }
  inline int right(int i) {
    return 2 * i + 2;
  }
  inline int parent(int i) {
    return (i - 1) / 2;
  }
  inline void swap(int i, int j) {
    T tmp = v[i];
    v[i] = v[j];
    v[j] = tmp;
  }

  inline void bubble_up(int i) {
    int p = parent(i);
    while (i > 0 && compare()(v[i], v[p]) < 0) {
      swap(i, p);
      i = p;
      p = parent(i);
    }
  }

  inline void trickleDown(int i) {
    do {
      int j = -1;
      int r = right(i);
      if (r < nn && compare()(v[r], v[i]) < 0) {
        int l = left(i);
        if (compare()(v[l], v[r]) < 0) {
          j = l;
        } else {
          j = r;
        }
      } else {
        int l = left(i);
        if (l < nn && compare()(v[l], v[i]) < 0) {
          j = l;
        }
      }
      if (j >= 0)
        swap(i, j);
      i = j;
    } while (i >= 0);
  }

};

template<typename T, typename compare>
NRHeap<T, compare>::NRHeap() :
    nn(0), ne(0), v(0), maxsize(0) {
}

template<typename T, typename compare>
NRHeap<T, compare> & NRHeap<T, compare>::resize(const int n) {
  if (nn != n) {
    if (v != 0) {
      delete[] (v);
    }
    nn = 0;
    maxsize = n;
    v = new T[n];
    ne = 0;
  }
  return *this;
}

template<typename T, typename compare>
NRHeap<T, compare>::NRHeap(const int n) :
    nn(0), v(0) {
  resize(n);
}

template<typename T, typename compare>
NRHeap<T, compare>::NRHeap(const NRHeap<T, compare> &rhs) :
    nn(rhs.nn), ne(rhs.ne), maxsize(rhs.maxsize), v(new T[maxsize]) {
  for (int i = 0; i < ne; i++)
    v[i] = rhs[i];
}

template<typename T, typename compare>
inline T & NRHeap<T, compare>::operator[](const int i) { // subscripting
  return v[i];  // no boundary check?
}

template<typename T, typename compare>
inline const T & NRHeap<T, compare>::operator[](const int i) const { // subscripting
  return v[i];  // no boundary check?
}

template<typename T, typename compare>
inline int NRHeap<T, compare>::heapsize() const {
  return nn;
}

template<typename T, typename compare>
inline int NRHeap<T, compare>::size() const {
  return maxsize;
}

template<typename T, typename compare>
inline int NRHeap<T, compare>::elemsize() const {
  return ne;
}

template<typename T, typename compare>
NRHeap<T, compare>::~NRHeap() {
  dealloc();
}

template<typename T, typename compare>
inline void NRHeap<T, compare>::dealloc() {
  if (v != 0) {
    delete[] (v);
    v = 0;
    nn = 0;
    ne = 0;
    maxsize = 0;
  }
}

template<typename T, typename compare>
inline T * NRHeap<T, compare>::c_buf() {
  return v;
}

template<typename T, typename compare>
inline void NRHeap<T, compare>::randu(int seed){
  srand(seed);
  for (int i = 0; i < nn; i++)
  {
    v[i] = (T)(1.0 *rand() / RAND_MAX);
  }
}

// ---------------- NRMat: matrix (2d) ---------------------

template<typename T>
class NRMat {
private:
  int nn;
  int mm;
  int tot_sz;
  T **v;
public:
  NRMat();
  NRMat & resize(const int n, const int m);
  explicit NRMat(const int n, const int m); // zero-based array

  NRMat(const T &a, const int n, const int m); // initialize to constant value
  NRMat(const T *a, const int n, const int m); // initialize to array
  NRMat(const NRMat &rhs); // copy constructor
  NRMat & operator=(const NRMat &rhs); // assignment
  NRMat & operator=(const T &a); // assign a to every element

  inline T* operator[](const int i); // subscripting: pointer to row i
  inline const T* operator[](const int i) const;
  inline int nrows() const;
  inline int ncols() const;
  inline int total_size() const;
  inline T * c_buf();
  inline void randu(int seed = 0);
  inline void dealloc();
  ~NRMat();
};

template<typename T>
NRMat<T>::~NRMat() {
  dealloc();
}

template<typename T>
void NRMat<T>::dealloc() {
  if (v != 0) {
    delete[] (v[0]);
    delete[] (v);
    v = 0;
    nn = 0;
    mm = 0;
    tot_sz = 0;
  }
}

template<typename T>
T * NRMat<T>::c_buf() {
  //assert(v != 0 && v[0] != 0);
  if (v)
    return v[0];
  else
    return 0;
}

template<typename T>
NRMat<T>::NRMat() :
    nn(0), mm(0), tot_sz(0), v(0) {
}

template<typename T>
NRMat<T> & NRMat<T>::resize(const int n, const int m) {
  if (nn != n || mm != m) {
    dealloc();
    nn = n;
    mm = m;
    tot_sz = n * m;

    v = new T*[n];
    v[0] = new T[tot_sz];

    for (int i = 1; i < n; i++)
      v[i] = v[i - 1] + m; // all pointers
  }

  return *this;
}

template<typename T>
NRMat<T>::NRMat(const int n, const int m) :
    nn(0), mm(0), tot_sz(0), v(0) {
  resize(n, m);
}

template<typename T>
NRMat<T>::NRMat(const T &a, const int n, const int m) {
  resize(n, m);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < m; j++)
      v[i][j] = a;
}

template<typename T>
NRMat<T>::NRMat(const T *a, const int n, const int m) {
  resize(n, m);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < m; j++)
      v[i][j] = *a++;
}

template<typename T>
NRMat<T>::NRMat(const NRMat &rhs) {
  resize(rhs.nn, rhs.mm);
  for (int i = 0; i < nn; i++)
    for (int j = 0; j < mm; j++)
      v[i][j] = rhs[i][j];
}

template<typename T>
NRMat<T> & NRMat<T>::operator=(const NRMat<T> &rhs) {
  // postcondition: normal assignment via copying has been performed;
  //		if matrix and rhs were of different sizes, matrix has been resized to match the size of rhs
  if (this != &rhs) {
    resize(rhs.nn, rhs.mm);
    for (int i = 0; i < nn; i++)
      for (int j = 0; j < mm; j++)
        v[i][j] = rhs[i][j];
  }
  return *this;
}

template<typename T>
NRMat<T> & NRMat<T>::operator=(const T &a) { // assign a to every element
  for (int i = 0; i < nn; i++)
    for (int j = 0; j < mm; j++)
      v[i][j] = a;
  return *this;
}

template<typename T>
inline T* NRMat<T>::operator[](const int i) { // subscripting: pointer to row i
  return v[i];  // no boundary check?
}

template<typename T>
inline const T* NRMat<T>::operator[](const int i) const { // subscripting: pointer to row i
  return v[i];  // no boundary check?
}

template<typename T>
inline int NRMat<T>::nrows() const {
  return nn;
}

template<typename T>
inline int NRMat<T>::ncols() const {
  return mm;
}

template<typename T>
inline int NRMat<T>::total_size() const {
  return tot_sz;
}

template<typename T>
inline void NRMat<T>::randu(int seed){
  srand(seed);
  for (int i = 0; i < nn; i++)
    for (int j = 0; j < mm; j++)
      v[i][j] = (T)(1.0 *rand() / RAND_MAX);
}

// ---------------- NRMat3d: matrix (3d) ---------------------

template<typename T>
class NRMat3d {
private:
  int nn;
  int mm;
  int kk;
  int tot_sz;
  T ***v;
public:
  NRMat3d();
  NRMat3d & resize(const int n, const int m, const int k);
  explicit NRMat3d(const int n, const int m, const int k); // zero-based array
  NRMat3d & operator=(const T &a); // assign a to every element
  inline T**operator[](const int i); // subscripting: pointer to row i. should not it be: T* const *?? (i think the pointers should not change).
  inline const T* const * operator[](const int i) const;
  inline int dim1() const;
  inline int dim2() const;
  inline int dim3() const;
  inline int total_size() const;
  inline void dealloc();
  inline T* c_buf();
  inline void randu(int seed = 0);
  ~NRMat3d();

private:
  NRMat3d(const NRMat3d &rhs) {
  } // forbid: copy constructor
  NRMat3d & operator=(const NRMat3d &rhs) {
  } // forbid: assignment

};

template<typename T>
NRMat3d<T>::NRMat3d() :
    nn(0), mm(0), kk(0), tot_sz(0), v(0) {
}

template<typename T>
NRMat3d<T>::~NRMat3d() {
  dealloc();
}

template<typename T>
void NRMat3d<T>::dealloc() {
  if (v != 0) {
    delete[] (v[0][0]);
    delete[] (v[0]);
    delete[] (v);
    v = 0;
    nn = 0;
    mm = 0;
    kk = 0;
    tot_sz = 0;
  }
}

template<typename T>
T * NRMat3d<T>::c_buf() {
  //assert(v != 0 && v[0] != 0 && v[0][0] != 0);
  if (v)
    return v[0][0];
  else
    return 0;
}

template<typename T>
NRMat3d<T>::NRMat3d(const int n, const int m, const int k) :
    nn(0), mm(0), kk(0), tot_sz(0), v(0) {
  resize(n, m, k);
}

template<typename T>
NRMat3d<T> & NRMat3d<T>::resize(const int n, const int m, const int k) {
  if (nn != n || mm != m || kk != k) {
    dealloc();

    nn = n;
    mm = m;
    kk = k;
    tot_sz = n * m * k;

    v = new T**[n];
    v[0] = new T*[n * m];
    v[0][0] = new T[tot_sz];

    int i, j;
    for (j = 1; j < m; j++)
      v[0][j] = v[0][j - 1] + k;

    for (i = 1; i < n; i++) {
      v[i] = v[i - 1] + m;
      v[i][0] = v[i - 1][0] + m * k;

      for (j = 1; j < m; j++)
        v[i][j] = v[i][j - 1] + k;
    }
  }
  return *this;
}

template<typename T>
NRMat3d<T> & NRMat3d<T>::operator=(const T &a) { // assign a to every element
  for (int i = 0; i < nn; i++)
    for (int j = 0; j < mm; j++)
      for (int k = 0; k < kk; k++)
        v[i][j][k] = a;
  return *this;
}

template<typename T>
inline T** NRMat3d<T>::operator[](const int i) { // subscripting: pointer to row i
  return v[i];  // no boundary check?
}

template<typename T>
inline const T* const * NRMat3d<T>::operator[](const int i) const { // subscripting: pointer to row i
  return v[i];  // no boundary check?
}

template<typename T>
inline int NRMat3d<T>::dim1() const {
  return nn;
}

template<typename T>
inline int NRMat3d<T>::dim2() const {
  return mm;
}

template<typename T>
inline int NRMat3d<T>::dim3() const {
  return kk;
}

template<typename T>
inline int NRMat3d<T>::total_size() const {
  return tot_sz;
}

template<typename T>
inline void NRMat3d<T>::randu(int seed){
  srand(seed);
  for (int i = 0; i < nn; i++)
    for (int j = 0; j < mm; j++)
      for (int k = 0; k < kk; k++)
        v[i][j][k] = (T)(1.0 *rand() / RAND_MAX);
}

// ---------------- NRMat3d: matrix (4d) ---------------------

template<typename T>
class NRMat4d {
private:
  int nn;
  int mm;
  int kk;
  int ll;
  int tot_sz;
  T ****v;
public:
  NRMat4d();
  NRMat4d & resize(const int n, const int m, const int k, const int l);
  explicit NRMat4d(const int n, const int m, const int k, const int l); // zero-based array
  NRMat4d & operator=(const T &a); // assign a to every element
  inline T***operator[](const int i); // subscripting: pointer to row i. should not it be: T* const *?? (i think the pointers should not change).
  inline const T* const * const * operator[](const int i) const;
  inline int dim1() const;
  inline int dim2() const;
  inline int dim3() const;
  inline int dim4() const;
  inline int total_size() const;

  inline void dealloc();
  inline T * c_buf();
  inline void randu(int seed = 0);
  ~NRMat4d();

private:
  NRMat4d(const NRMat4d &rhs) {
  } // forbid: copy constructor
  NRMat4d & operator=(const NRMat4d &rhs) {
  } // forbid: assignment

};

template<typename T>
NRMat4d<T>::NRMat4d() :
    nn(0), mm(0), kk(0), ll(0), tot_sz(0), v(0) {
}

template<typename T>
NRMat4d<T>::~NRMat4d() {
  dealloc();
}

template<typename T>
void NRMat4d<T>::dealloc() {
  if (v != 0) {
    delete[] (v[0][0][0]);
    delete[] (v[0][0]);
    delete[] (v[0]);
    delete[] (v);
    v = 0;
    nn = 0;
    mm = 0;
    kk = 0;
    ll = 0;
    tot_sz = 0;
  }
}

template<typename T>
T * NRMat4d<T>::c_buf() {
  //assert(v != 0 && v[0] != 0 && v[0][0] != 0 && v[0][0][0] != 0);
  if (v)
    return v[0][0][0];
  else
    return 0;
}

template<typename T>
NRMat4d<T>::NRMat4d(const int n, const int m, const int k, const int l) :
    nn(0), mm(0), kk(0), ll(0), v(0) {
  resize(n, m, k, l);
}

template<typename T>
NRMat4d<T> & NRMat4d<T>::resize(const int n, const int m, const int k, const int l) {
  if (nn != n || mm != m || kk != k || ll != l) {
    dealloc();

    nn = n;
    mm = m;
    kk = k;
    ll = l;
    tot_sz = n * m * k * l;

    v = new T***[n];
    v[0] = new T**[n * m];
    v[0][0] = new T*[n * m * k];
    v[0][0][0] = new T[n * m * k * l];

    int i, j, z;

    for (z = 1; z < k; z++) {
      v[0][0][z] = v[0][0][z - 1] + l;
    }

    for (j = 1; j < m; j++) {
      v[0][j] = v[0][j - 1] + k;
      v[0][j][0] = v[0][j - 1][0] + k * l;
      for (z = 1; z < k; z++) {
        v[0][j][z] = v[0][j][z - 1] + l;
      }
    }

    for (i = 1; i < n; i++) {
      v[i] = v[i - 1] + m;
      v[i][0] = v[i - 1][0] + m * k;
      v[i][0][0] = v[i - 1][0][0] + m * k * l;

      for (z = 1; z < k; z++) {
        v[i][0][z] = v[i][0][z - 1] + l;
      }

      for (j = 1; j < m; j++) {
        v[i][j] = v[i][j - 1] + k;
        v[i][j][0] = v[i][j - 1][0] + k * l;

        for (z = 1; z < k; z++) {
          v[i][j][z] = v[i][j][z - 1] + l;
        }
      }
    }

    // for test
    /*			for (i = 0; i < n*m*k*l; i++) {
     v[0][0][0][i] = i;
     }

     for (i = 0; i < n; i++)
     for (j = 0; j < m; j++)
     for (z = 0; z < k; z++)
     for (int y = 0; y < l; ++y)
     cerr << v[i][j][z][y] << "\t";
     cerr << endl; */
  }
  return *this;
}

template<typename T>
NRMat4d<T> & NRMat4d<T>::operator=(const T &a) { // assign a to every element
  for (int i = 0; i < nn; i++)
    for (int j = 0; j < mm; j++)
      for (int k = 0; k < kk; k++)
        for (int l = 0; l < ll; l++)
          v[i][j][k][l] = a;
  return *this;
}

template<typename T>
inline T*** NRMat4d<T>::operator[](const int i) { // subscripting: pointer to row i
  return v[i];  // no boundary check?
}

template<typename T>
inline const T* const * const * NRMat4d<T>::operator[](const int i) const { // subscripting: pointer to row i
  return v[i];  // no boundary check?
}

template<typename T>
inline int NRMat4d<T>::dim1() const {
  return nn;
}

template<typename T>
inline int NRMat4d<T>::dim2() const {
  return mm;
}

template<typename T>
inline int NRMat4d<T>::dim3() const {
  return kk;
}

template<typename T>
inline int NRMat4d<T>::dim4() const {
  return ll;
}

template<typename T>
inline int NRMat4d<T>::total_size() const {
  return tot_sz;
}

template<typename T>
inline void NRMat4d<T>::randu(int seed){
  srand(seed);
  for (int i = 0; i < nn; i++)
    for (int j = 0; j < mm; j++)
      for (int k = 0; k < kk; k++)
        for (int l = 0; l < ll; l++)
          v[i][j][k][l] = (T)(1.0 *rand() / RAND_MAX);
}

}

#endif

