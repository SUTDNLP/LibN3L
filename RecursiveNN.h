/*
 * RecursiveNN.h
 *
 *  Created on: Mar 18, 2015
 *      Author: mszhang
 */

#ifndef SRC_RecursiveNN_H_
#define SRC_RecursiveNN_H_
#include "tensor.h"

#include "BiLayer.h"
#include "MyLib.h"
#include "Utiltensor.h"

using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;

// Actually, we do not need such a class, BiLayer satisfies it

template<typename xpu>
class RecursiveNN {
public:
  BiLayer<xpu> _rnn;

public:
  RecursiveNN() {
  }

  inline void initial(int dimension, int seed = 0) {
    _rnn.initial(dimension, dimension, dimension, true, seed, 0);
  }


  inline void initial(Tensor<xpu, 2, dtype> WL, Tensor<xpu, 2, dtype> WR, Tensor<xpu, 2, dtype> b) {
    _rnn.initial(WL, WR, b, true);
  }

  inline void release() {
    _rnn.release();
  }

  virtual ~RecursiveNN() {
    // TODO Auto-generated destructor stub
  }

  inline dtype squarenormAll() {
    dtype norm = _rnn.squarenormAll();

    return norm;
  }

  inline void scaleGrad(dtype scale) {
    _rnn.scaleGrad(scale);
  }

public:

  inline void ComputeForwardScore(Tensor<xpu, 2, dtype> xl, Tensor<xpu, 2, dtype> xr, Tensor<xpu, 2, dtype> y) {
    y = 0.0;
   _rnn.ComputeForwardScore(xl, xr, y);

  }

  //please allocate the memory outside here
  inline void ComputeBackwardLoss(Tensor<xpu, 2, dtype> xl, Tensor<xpu, 2, dtype> xr, Tensor<xpu, 2, dtype> y, Tensor<xpu, 2, dtype> ly,
      Tensor<xpu, 2, dtype> lxl, Tensor<xpu, 2, dtype> lxr, bool bclear = false) {
    if (bclear){
      lxl = 0.0; lxr = 0.0;
    }
    _rnn.ComputeBackwardLoss(xl, xr, y, ly, lxl, lxr);
  }


  inline void randomprint(int num) {
    _rnn.randomprint(num);
  }

  inline void updateAdaGrad(dtype regularizationWeight, dtype adaAlpha, dtype adaEps) {
    _rnn.updateAdaGrad(regularizationWeight, adaAlpha, adaEps);
  }

  void writeModel(LStream &outf) {
    _rnn.writeModel(outf);
  }

  void loadModel(LStream &inf) {
    _rnn.loadModel(inf);
  }

};

#endif /* SRC_RecursiveNN_H_ */
