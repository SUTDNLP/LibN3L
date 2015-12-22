/*
 * UniLayer1O.h
 *
 *  Created on: Mar 18, 2015
 *      Author: mszhang
 */
/*
 *  use it only for output layer
 */
#ifndef SRC_UniLayer1O_H_
#define SRC_UniLayer1O_H_
#include "tensor.h"
#include "MyLib.h"
#include "Utiltensor.h"

using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;

template<typename xpu>
class UniLayer1O {

public:

  Tensor<xpu, 2, dtype> _W;

  Tensor<xpu, 2, dtype> _gradW;

  Tensor<xpu, 2, dtype> _eg2W;

public:
  UniLayer1O() {
  }

  inline void initial(int nISize, int seed = 0) {
    dtype bound = sqrt(6.0 / (1 + nISize + 1));
    //dtype bound = 0.01;

    _W = NewTensor<xpu>(Shape2(1, nISize), d_zero);
    _gradW = NewTensor<xpu>(Shape2(1, nISize), d_zero);
    _eg2W = NewTensor<xpu>(Shape2(1, nISize), d_zero);

    random(_W, -1.0 * bound, 1.0 * bound, seed);

  }

  inline void initial(Tensor<xpu, 2, dtype> W) {
    static int nISize;
    nISize = W.size(1);

    _W = NewTensor<xpu>(Shape2(1, nISize), d_zero);
    _gradW = NewTensor<xpu>(Shape2(1, nISize), d_zero);
    _eg2W = NewTensor<xpu>(Shape2(1, nISize), d_zero);
    Copy(_W, W);

  }

  inline void release() {
    FreeSpace(&_W);
    FreeSpace(&_gradW);
    FreeSpace(&_eg2W);
  }

  virtual ~UniLayer1O() {
    // TODO Auto-generated destructor stub
  }

  inline dtype squarenormAll() {
    dtype result = squarenorm(_gradW);

    return result;
  }

  inline void scaleGrad(dtype scale) {
    _gradW = _gradW * scale;
  }

public:
  inline void ComputeForwardScore(Tensor<xpu, 2, dtype> x, dtype& y) {
    static int nISize;
    nISize = _W.size(1);
    y = 0.0;
    for(int idx = 0; idx < nISize; idx++){
    	y += x[0][idx] * _W[0][idx];
    }
  }


  //please allocate the memory outside here
  inline void ComputeBackwardLoss(Tensor<xpu, 2, dtype> x, dtype ly, Tensor<xpu, 2, dtype> lx, bool bclear = false) {
    //_gradW
    _gradW += ly * x;

    if (bclear)
      lx = 0.0;
    //lx
    lx += ly * _W;

  }


  inline void randomprint(int num) {
    static int nISize;
    nISize = _W.size(1);
    int count = 0;
    while (count < num) {
      int idy = rand() % nISize;
      std::cout << "_W[" << 0 << "," << idy << "]=" << _W[0][idy] << " ";
      count++;
    }

    std::cout << std::endl;
  }

  inline void updateAdaGrad(dtype regularizationWeight, dtype adaAlpha, dtype adaEps) {
    _gradW = _gradW + _W * regularizationWeight;
    _eg2W = _eg2W + _gradW * _gradW;
    _W = _W - _gradW * adaAlpha / F<nl_sqrt>(_eg2W + adaEps);


    clearGrad();
  }

  inline void clearGrad() {
    _gradW = 0;
  }

  void writeModel(LStream &outf) {
    SaveBinary(outf, _W);
    SaveBinary(outf, _gradW);
    SaveBinary(outf, _eg2W);

  }

  void loadModel(LStream &inf) {
    LoadBinary(inf, &_W, false);
    LoadBinary(inf, &_gradW, false);
    LoadBinary(inf, &_eg2W, false);
  }
};

#endif /* SRC_UniLayer1O_H_ */
