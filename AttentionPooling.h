/*
 * AttentionPooling.h
 *
 *  Created on: Mar 18, 2015
 *      Author: mszhang
 */

#ifndef SRC_AttentionPooling_H_
#define SRC_AttentionPooling_H_
#include "tensor.h"

#include "BiLayer.h"
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
class AttentionPooling {

public:
  BiLayer<xpu> _bi_gates;
  UniLayer<xpu> _uni_gates;

public:
  AttentionPooling() {
  }

  inline void initial(int hiddenSize, int attentionSize, bool bUseB = true, int seed = 0) {
    _bi_gates.initial(hiddenSize, hiddenSize, attentionSize, bUseB, seed);
    _uni_gates.initial(hiddenSize, hiddenSize, false, seed + 10, 3);
  }

  inline void initial(Tensor<xpu, 2, dtype> W1, Tensor<xpu, 2, dtype> W2, Tensor<xpu, 2, dtype> W3, Tensor<xpu, 2, dtype> b, bool bUseB = true) {
    _bi_gates.initial(W1, W2);
    _uni_gates.initial(W3, b, false, 3);

  }


  inline void release() {
    _bi_gates.release();
    _uni_gates.release();
  }

  virtual ~AttentionPooling() {
    // TODO Auto-generated destructor stub
  }

  inline dtype squarenormAll() {
    return _bi_gates.squarenormAll() + _uni_gates.squarenormAll();
  }

  inline void scaleGrad(dtype scale) {
    _bi_gates.scaleGrad(scale);
    _uni_gates.scaleGrad(scale);
  }

public:
  // xExp, xSumIndex, xSum ad xPoolIndex are temporal variables, which reduce computation in back-propagation
  inline void ComputeForwardScore(Tensor<xpu, 3, dtype> x, Tensor<xpu, 3, dtype> xAtt,
      Tensor<xpu, 3, dtype> xMExp, Tensor<xpu, 3, dtype> xExp,
      Tensor<xpu, 2, dtype> xSum, Tensor<xpu, 3, dtype> xPoolIndex, Tensor<xpu, 2, dtype> y) {
    y = 0.0;
    int seq_size = x.size(0);
    if(seq_size == 0) return;
    int dim1 = x.size(1), dim2 = x.size(2);
    int odim1 = y.size(0), odim2 = y.size(1);

    if (dim1 != odim1 || dim2 != odim2 || dim1 != 1) {
      std::cerr << "AttentionPooling Forward error: dim invalid" << std::endl;
    }

    _bi_gates.ComputeForwardScore(x, xAtt, xMExp);
    _uni_gates.ComputeForwardScore(xMExp, xExp);

    sumpool_forward(xExp, xSum);
    for (int idx = 0; idx < seq_size; idx++) {
      xPoolIndex[idx] = xExp[idx] / xSum;
    }
    for (int idx = 0; idx < seq_size; idx++) {
      y += x[idx] * xPoolIndex[idx];
    }
  }

  inline void ComputeForwardScore(const std::vector<Tensor<xpu, 2, dtype> >& x, const std::vector<Tensor<xpu, 2, dtype> >& xAtt,
      std::vector<Tensor<xpu, 2, dtype> >& xMExp, std::vector<Tensor<xpu, 2, dtype> >& xExp, Tensor<xpu, 2, dtype> xSum,
      std::vector<Tensor<xpu, 2, dtype> >& xPoolIndex, Tensor<xpu, 2, dtype> y) {
    y = 0.0;
    int seq_size = x.size();
    if(seq_size == 0) return;
    int dim1 = x[0].size(0), dim2 = x[0].size(1);
    int odim1 = y.size(0), odim2 = y.size(1);

    if (dim1 != odim1 || dim2 != odim2 || dim1 != 1) {
      std::cerr << "AttentionPooling Forward error: dim invalid" << std::endl;
    }

    _bi_gates.ComputeForwardScore(x, xAtt, xMExp);
    _uni_gates.ComputeForwardScore(xMExp, xExp);

    sumpool_forward(xExp, xSum);
    for (int idx = 0; idx < seq_size; idx++) {
      xPoolIndex[idx] = xExp[idx] / xSum;
    }
    for (int idx = 0; idx < seq_size; idx++) {
      y += x[idx] * xPoolIndex[idx];
    }
  }


  // xExp, xSumIndex, xSum ad xPoolIndex are temporal variables, which reduce computation in back-propagation
  inline void ComputeForwardScore(Tensor<xpu, 3, dtype> x, Tensor<xpu, 2, dtype> xAtt,
      Tensor<xpu, 3, dtype> xMExp, Tensor<xpu, 3, dtype> xExp,
      Tensor<xpu, 2, dtype> xSum, Tensor<xpu, 3, dtype> xPoolIndex, Tensor<xpu, 2, dtype> y) {
    y = 0.0;
    int seq_size = x.size(0);
    if(seq_size == 0) return;
    int dim1 = x.size(1), dim2 = x.size(2);
    int odim1 = y.size(0), odim2 = y.size(1);

    if (dim1 != odim1 || dim2 != odim2 || dim1 != 1) {
      std::cerr << "AttentionPooling Forward error: dim invalid" << std::endl;
    }

    for (int idx = 0; idx < seq_size; idx++) {
      _bi_gates.ComputeForwardScore(x[idx], xAtt, xMExp[idx]);
    }
    _uni_gates.ComputeForwardScore(xMExp, xExp);

    sumpool_forward(xExp, xSum);
    for (int idx = 0; idx < seq_size; idx++) {
      xPoolIndex[idx] = xExp[idx] / xSum;
    }
    for (int idx = 0; idx < seq_size; idx++) {
      y += x[idx] * xPoolIndex[idx];
    }
  }

  inline void ComputeForwardScore(const std::vector<Tensor<xpu, 2, dtype> >& x, Tensor<xpu, 2, dtype> xAtt,
      std::vector<Tensor<xpu, 2, dtype> >& xMExp, std::vector<Tensor<xpu, 2, dtype> >& xExp, Tensor<xpu, 2, dtype> xSum,
      std::vector<Tensor<xpu, 2, dtype> >& xPoolIndex, Tensor<xpu, 2, dtype> y) {
    y = 0.0;
    int seq_size = x.size();
    if(seq_size == 0) return;
    int dim1 = x[0].size(0), dim2 = x[0].size(1);
    int odim1 = y.size(0), odim2 = y.size(1);

    if (dim1 != odim1 || dim2 != odim2 || dim1 != 1) {
      std::cerr << "AttentionPooling Forward error: dim invalid" << std::endl;
    }

    for (int idx = 0; idx < seq_size; idx++) {
      _bi_gates.ComputeForwardScore(x[idx], xAtt, xMExp[idx]);
    }
    _uni_gates.ComputeForwardScore(xMExp, xExp);

    sumpool_forward(xExp, xSum);
    for (int idx = 0; idx < seq_size; idx++) {
      xPoolIndex[idx] = xExp[idx] / xSum;
    }
    for (int idx = 0; idx < seq_size; idx++) {
      y += x[idx] * xPoolIndex[idx];
    }
  }


  //please allocate the memory outside here
  inline void ComputeBackwardLoss(Tensor<xpu, 3, dtype> x, Tensor<xpu, 3, dtype> xAtt,
      Tensor<xpu, 3, dtype> xMExp, Tensor<xpu, 3, dtype> xExp,
      Tensor<xpu, 2, dtype> xSum, Tensor<xpu, 3, dtype> xPoolIndex, Tensor<xpu, 2, dtype> y,
      Tensor<xpu, 2, dtype> ly, Tensor<xpu, 3, dtype> lx, Tensor<xpu, 3, dtype> lxAtt, bool bclear = false) {
    int seq_size = x.size(0);
    if(seq_size == 0) return;
    int dim1 = x.size(1), dim2 = x.size(2);
    int odim1 = y.size(0), odim2 = y.size(1);

    if(bclear) lx = 0.0;
    if(bclear) lxAtt = 0.0;

    Tensor<xpu, 3, dtype> xMExpLoss = NewTensor<xpu>(Shape3(seq_size, dim1, dim2), d_zero);
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

    _uni_gates.ComputeBackwardLoss(xMExp, xExp, xExpLoss, xMExpLoss);
    _bi_gates.ComputeBackwardLoss(x, xAtt, xMExp, xMExpLoss, lx, lxAtt);

    FreeSpace(&xMExpLoss);
    FreeSpace(&xExpLoss);
    FreeSpace(&xSumLoss);
    FreeSpace(&xPoolIndexLoss);
  }

  inline void ComputeBackwardLoss(const std::vector<Tensor<xpu, 2, dtype> >& x, std::vector<Tensor<xpu, 2, dtype> >& xAtt,
      std::vector<Tensor<xpu, 2, dtype> >& xMExp, std::vector<Tensor<xpu, 2, dtype> >& xExp,
      Tensor<xpu, 2, dtype> xSum, std::vector<Tensor<xpu, 2, dtype> >& xPoolIndex, Tensor<xpu, 2, dtype> y,
      Tensor<xpu, 2, dtype> ly, std::vector<Tensor<xpu, 2, dtype> >& lx, std::vector<Tensor<xpu, 2, dtype> >& lxAtt, bool bclear = false) {
    int seq_size = x.size();
    if(seq_size == 0) return;
    int dim1 = x[0].size(0), dim2 = x[0].size(1);
    int odim1 = y.size(0), odim2 = y.size(1);


    if(bclear){
      for (int idx = 0; idx < seq_size; idx++) {
        lx[idx] = 0.0;
        lxAtt[idx] = 0.0;
      }
    }

    vector<Tensor<xpu, 2, dtype> > xMExpLoss(seq_size), xExpLoss(seq_size), xPoolIndexLoss(seq_size);
    for (int idx = 0; idx < seq_size; idx++) {
      xMExpLoss[idx] = NewTensor<xpu>(Shape2(dim1, dim2), d_zero);
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

    _uni_gates.ComputeBackwardLoss(xMExp, xExp, xExpLoss, xMExpLoss);
    _bi_gates.ComputeBackwardLoss(x, xAtt, xMExp, xMExpLoss, lx, lxAtt);

    FreeSpace(&xSumLoss);
    for (int idx = 0; idx < seq_size; idx++) {
      FreeSpace(&(xMExpLoss[idx]));
      FreeSpace(&(xExpLoss[idx]));
      FreeSpace(&(xPoolIndexLoss[idx]));
    }
  }

  //please allocate the memory outside here
  inline void ComputeBackwardLoss(Tensor<xpu, 3, dtype> x, Tensor<xpu, 2, dtype> xAtt,
      Tensor<xpu, 3, dtype> xMExp, Tensor<xpu, 3, dtype> xExp,
      Tensor<xpu, 2, dtype> xSum, Tensor<xpu, 3, dtype> xPoolIndex, Tensor<xpu, 2, dtype> y,
      Tensor<xpu, 2, dtype> ly, Tensor<xpu, 3, dtype> lx, Tensor<xpu, 2, dtype> lxAtt, bool bclear = false) {
    int seq_size = x.size(0);
    if(seq_size == 0) return;
    int dim1 = x.size(1), dim2 = x.size(2);
    int odim1 = y.size(0), odim2 = y.size(1);

    if(bclear) lx = 0.0;
    if(bclear) lxAtt = 0.0;

    Tensor<xpu, 3, dtype> xMExpLoss = NewTensor<xpu>(Shape3(seq_size, dim1, dim2), d_zero);
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

    _uni_gates.ComputeBackwardLoss(xMExp, xExp, xExpLoss, xMExpLoss);
    for (int idx = 0; idx < seq_size; idx++) {
      _bi_gates.ComputeBackwardLoss(x[idx], xAtt, xMExp[idx], xMExpLoss[idx], lx[idx], lxAtt);
    }

    FreeSpace(&xMExpLoss);
    FreeSpace(&xExpLoss);
    FreeSpace(&xSumLoss);
    FreeSpace(&xPoolIndexLoss);
  }

  inline void ComputeBackwardLoss(const std::vector<Tensor<xpu, 2, dtype> >& x, Tensor<xpu, 2, dtype> xAtt,
      std::vector<Tensor<xpu, 2, dtype> >& xMExp, std::vector<Tensor<xpu, 2, dtype> >& xExp,
      Tensor<xpu, 2, dtype> xSum, std::vector<Tensor<xpu, 2, dtype> >& xPoolIndex, Tensor<xpu, 2, dtype> y,
      Tensor<xpu, 2, dtype> ly, std::vector<Tensor<xpu, 2, dtype> >& lx, Tensor<xpu, 2, dtype> lxAtt, bool bclear = false) {
    int seq_size = x.size();
    if(seq_size == 0) return;
    int dim1 = x[0].size(0), dim2 = x[0].size(1);
    int odim1 = y.size(0), odim2 = y.size(1);


    if(bclear){
      for (int idx = 0; idx < seq_size; idx++) {
        lx[idx] = 0.0;
        lxAtt[idx] = 0.0;
      }
    }

    vector<Tensor<xpu, 2, dtype> > xMExpLoss(seq_size), xExpLoss(seq_size), xPoolIndexLoss(seq_size);
    for (int idx = 0; idx < seq_size; idx++) {
      xMExpLoss[idx] = NewTensor<xpu>(Shape2(dim1, dim2), d_zero);
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

    _uni_gates.ComputeBackwardLoss(xMExp, xExp, xExpLoss, xMExpLoss);
    for (int idx = 0; idx < seq_size; idx++) {
      _bi_gates.ComputeBackwardLoss(x[idx], xAtt, xMExp[idx], xMExpLoss[idx], lx[idx], lxAtt);
    }

    FreeSpace(&xSumLoss);
    for (int idx = 0; idx < seq_size; idx++) {
      FreeSpace(&(xExpLoss[idx]));
      FreeSpace(&(xPoolIndexLoss[idx]));
    }
  }

  inline void randomprint(int num) {
    _bi_gates.randomprint(num);
    _uni_gates.randomprint(num);
  }

  inline void updateAdaGrad(dtype regularizationWeight, dtype adaAlpha, dtype adaEps) {
    _bi_gates.updateAdaGrad(regularizationWeight, adaAlpha, adaEps);
    _uni_gates.updateAdaGrad(regularizationWeight, adaAlpha, adaEps);
  }

  void writeModel(LStream &outf) {
    _bi_gates.writeModel(outf);
    _uni_gates.writeModel(outf);

  }

  void loadModel(LStream &inf) {
    _bi_gates.loadModel(inf);
    _uni_gates.loadModel(inf);

  }
};

#endif /* SRC_AttentionPooling_H_ */
