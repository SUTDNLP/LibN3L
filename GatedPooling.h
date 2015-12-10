/*
 * GatedPooling.h
 *
 *  Created on: Mar 18, 2015
 *      Author: mszhang
 */

#ifndef SRC_GatedPooling_H_
#define SRC_GatedPooling_H_
#include "tensor.h"
#include "MyLib.h"
#include "Utiltensor.h"
#include "Pooling.h"
#include "UniLayer.h"

using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;

// For simpleness, we do not provide pooling on specified words,
// which has been implemented in Pooling.h


template<typename xpu>
class GatedPooling {

public:
  UniLayer<xpu> _uni_gates;

public:
  GatedPooling() {
  }

  inline void initial(int hiddenSize, int seed = 0) {
    _uni_gates.initial(hiddenSize, hiddenSize, false, seed, 3);
  }

  inline void initial(Tensor<xpu, 2, dtype> W) {
    _uni_gates.initial(W, 3);
  }


  inline void release() {
    _uni_gates.release();
  }

  virtual ~GatedPooling() {
    // TODO Auto-generated destructor stub
  }

  inline dtype squarenormAll() {
    return _uni_gates.squarenormAll();
  }

  inline void scaleGrad(dtype scale) {
    _uni_gates.scaleGrad(scale);
  }

public:
  // xExp, xSumIndex, xSum ad xPoolIndex are temporal variables, which reduce computation in back-propagation
  inline void ComputeForwardScore(Tensor<xpu, 3, dtype> x, Tensor<xpu, 3, dtype> xExp,
      Tensor<xpu, 2, dtype> xSum, Tensor<xpu, 3, dtype> xPoolIndex, Tensor<xpu, 2, dtype> y) {
    y = 0.0;
    int seq_size = x.size(0);
    if(seq_size == 0) return;
    int dim1 = x.size(1), dim2 = x.size(2);
    int odim1 = y.size(0), odim2 = y.size(1);

    if (dim1 != odim1 || dim2 != odim2 || dim1 != 1) {
      std::cerr << "GatedPooling Forward error: dim invalid" << std::endl;
    }

    _uni_gates.ComputeForwardScore(x, xExp);

    sumpool_forward(xExp, xSum);
    for (int idx = 0; idx < seq_size; idx++) {
      xPoolIndex[idx] = xExp[idx] / xSum;
    }
    for (int idx = 0; idx < seq_size; idx++) {
      y += x[idx] * xPoolIndex[idx];
    }
  }

  inline void ComputeForwardScore(const std::vector<Tensor<xpu, 2, dtype> >& x, std::vector<Tensor<xpu, 2, dtype> >& xExp,
      Tensor<xpu, 2, dtype> xSum, std::vector<Tensor<xpu, 2, dtype> >& xPoolIndex, Tensor<xpu, 2, dtype> y) {
    y = 0.0;
    int seq_size = x.size();
    if(seq_size == 0) return;
    int dim1 = x[0].size(0), dim2 = x[0].size(1);
    int odim1 = y.size(0), odim2 = y.size(1);

    if (dim1 != odim1 || dim2 != odim2 || dim1 != 1) {
      std::cerr << "GatedPooling Forward error: dim invalid" << std::endl;
    }

    _uni_gates.ComputeForwardScore(x, xExp);

    sumpool_forward(xExp, xSum);
    for (int idx = 0; idx < seq_size; idx++) {
      xPoolIndex[idx] = xExp[idx] / xSum;
    }
    for (int idx = 0; idx < seq_size; idx++) {
      y += x[idx] * xPoolIndex[idx];
    }
  }


  //please allocate the memory outside here
  inline void ComputeBackwardLoss(Tensor<xpu, 3, dtype> x, Tensor<xpu, 3, dtype> xExp,
      Tensor<xpu, 2, dtype> xSum, Tensor<xpu, 3, dtype> xPoolIndex, Tensor<xpu, 2, dtype> y,
      Tensor<xpu, 2, dtype> ly, Tensor<xpu, 3, dtype> lx, bool bclear = false) {
    int seq_size = x.size(0);
    if(seq_size == 0) return;
    int dim1 = x.size(1), dim2 = x.size(2);
    int odim1 = y.size(0), odim2 = y.size(1);

    if(bclear) lx = 0.0;

    Tensor<xpu, 3, dtype> xExpLoss = NewTensor<xpu>(Shape3(seq_size, dim1, dim2), d_zero);
    Tensor<xpu, 2, dtype> xSumLoss = NewTensor<xpu>(Shape2(dim1, dim2), d_zero);
    Tensor<xpu, 3, dtype> xPoolIndexLoss = NewTensor<xpu>(Shape3(seq_size, dim1, dim2), d_zero);

    for (int idx = 0; idx < seq_size; idx++) {
      xPoolIndexLoss[idx] = ly * x[idx];
      lx[idx] += ly * xPoolIndex[idx];
    }

    for (int idx = 0; idx < seq_size; idx++) {
      xExpLoss[idx] += xPoolIndexLoss[idx] / xSum;
      xSumLoss -= xPoolIndexLoss[idx] * xExp[idx] / xSum / xSum;
    }

    sumpool_backward(xSumLoss, xExpLoss);

    _uni_gates.ComputeBackwardLoss(x, xExp, xExpLoss, lx);

    FreeSpace(&xExpLoss);
    FreeSpace(&xSumLoss);
    FreeSpace(&xPoolIndexLoss);
  }

  inline void ComputeBackwardLoss(const std::vector<Tensor<xpu, 2, dtype> >& x, std::vector<Tensor<xpu, 2, dtype> >& xExp,
      Tensor<xpu, 2, dtype> xSum, std::vector<Tensor<xpu, 2, dtype> >& xPoolIndex, Tensor<xpu, 2, dtype> y,
      Tensor<xpu, 2, dtype> ly, std::vector<Tensor<xpu, 2, dtype> >& lx, bool bclear = false) {
    int seq_size = x.size();
    if(seq_size == 0) return;
    int dim1 = x[0].size(0), dim2 = x[0].size(1);
    int odim1 = y.size(0), odim2 = y.size(1);


    if(bclear){
      for (int idx = 0; idx < seq_size; idx++) {
        lx[idx] = 0.0;
      }
    }

    vector<Tensor<xpu, 3, dtype> > xExpLoss(seq_size), xPoolIndexLoss(seq_size);
    for (int idx = 0; idx < seq_size; idx++) {
      xExpLoss[idx] = NewTensor<xpu>(Shape2(dim1, dim2), d_zero);
      xPoolIndexLoss[idx] = NewTensor<xpu>(Shape2(dim1, dim2), d_zero);
    }

    Tensor<xpu, 2, dtype> xSumLoss = NewTensor<xpu>(Shape2(dim1, dim2), d_zero);

    for (int idx = 0; idx < seq_size; idx++) {
      xPoolIndexLoss[idx] = ly * x[idx];
      lx[idx] += ly * xPoolIndex[idx];
    }

    for (int idx = 0; idx < seq_size; idx++) {
      xExpLoss[idx] += xPoolIndexLoss[idx] / xSum;
      xSumLoss -= xPoolIndexLoss[idx] * xExp[idx] / xSum / xSum;
    }

    sumpool_backward(xSumLoss, xExpLoss);

    _uni_gates.ComputeBackwardLoss(x, xExp, xExpLoss, lx);

    FreeSpace(&xSumLoss);
    for (int idx = 0; idx < seq_size; idx++) {
      FreeSpace(&(xExpLoss[idx]));
      FreeSpace(&(xPoolIndexLoss[idx]));
    }
  }

  inline void randomprint(int num) {
    _uni_gates.randomprint(num);
  }

  inline void updateAdaGrad(dtype regularizationWeight, dtype adaAlpha, dtype adaEps) {
    _uni_gates.updateAdaGrad(regularizationWeight, adaAlpha, adaEps);
  }

  void writeModel(LStream &outf) {
    _uni_gates.writeModel(outf);

  }

  void loadModel(LStream &inf) {
    _uni_gates.loadModel(inf);
  }

};

#endif /* SRC_GatedPooling_H_ */
