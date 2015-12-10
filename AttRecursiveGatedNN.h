/*
 * AttRecursiveGatedNN.h
 *  Gated Recursive Neural network structure with attention technique.
 *  Created on: Nov 5, 2015
 *      Author: mszhang
 */

#ifndef SRC_AttRecursiveGatedNN_H_
#define SRC_AttRecursiveGatedNN_H_

#include "tensor.h"

#include "BiLayer.h"
#include "MyLib.h"
#include "Utiltensor.h"

using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;

template<typename xpu>
class AttRecursiveGatedNN {
public:
  BiLayer<xpu> _reset_left;
  BiLayer<xpu> _reset_right;
  BiLayer<xpu> _update_left;
  BiLayer<xpu> _update_right;
  BiLayer<xpu> _update_tilde;
  BiLayer<xpu> _recursive_tilde;


  Tensor<xpu, 2, dtype> nxl;
  Tensor<xpu, 2, dtype> nxr;
  Tensor<xpu, 2, dtype> sum;

  Tensor<xpu, 2, dtype> pxl;
  Tensor<xpu, 2, dtype> pxr;
  Tensor<xpu, 2, dtype> pmy;


  Tensor<xpu, 2, dtype> lrxl;
  Tensor<xpu, 2, dtype> lrxr;
  Tensor<xpu, 2, dtype> lmy;
  Tensor<xpu, 2, dtype> luxl;
  Tensor<xpu, 2, dtype> luxr;
  Tensor<xpu, 2, dtype> lumy;

  Tensor<xpu, 2, dtype> lnxl;
  Tensor<xpu, 2, dtype> lnxr;
  Tensor<xpu, 2, dtype> lsum;

  Tensor<xpu, 2, dtype> lpxl;
  Tensor<xpu, 2, dtype> lpxr;
  Tensor<xpu, 2, dtype> lpmy;


public:
  AttRecursiveGatedNN() {
  }

  inline void initial(int dimension, int attDim, int seed = 0) {
    _reset_left.initial(dimension, dimension, attDim, false, seed, 1);
    _reset_right.initial(dimension, dimension, attDim, false, seed + 10, 1);
    _update_left.initial(dimension, dimension, attDim, false, seed + 20, 3);
    _update_right.initial(dimension, dimension, attDim, false, seed + 30, 3);
    _update_tilde.initial(dimension, dimension, attDim, false, seed + 40, 3);
    _recursive_tilde.initial(dimension, dimension, dimension, false, seed + 50, 0);

    nxl = NewTensor<xpu>(Shape2(1, dimension), d_zero);
    nxr = NewTensor<xpu>(Shape2(1, dimension), d_zero);
    sum = NewTensor<xpu>(Shape2(1, dimension), d_zero);

    pxl = NewTensor<xpu>(Shape2(1, dimension), d_zero);
    pxr = NewTensor<xpu>(Shape2(1, dimension), d_zero);
    pmy = NewTensor<xpu>(Shape2(1, dimension), d_zero);


    lrxl = NewTensor<xpu>(Shape2(1, dimension), d_zero);
    lrxr = NewTensor<xpu>(Shape2(1, dimension), d_zero);
    lmy = NewTensor<xpu>(Shape2(1, dimension), d_zero);
    luxl = NewTensor<xpu>(Shape2(1, dimension), d_zero);
    luxr = NewTensor<xpu>(Shape2(1, dimension), d_zero);
    lumy = NewTensor<xpu>(Shape2(1, dimension), d_zero);

    lnxl = NewTensor<xpu>(Shape2(1, dimension), d_zero);
    lnxr = NewTensor<xpu>(Shape2(1, dimension), d_zero);
    lsum = NewTensor<xpu>(Shape2(1, dimension), d_zero);

    lpxl = NewTensor<xpu>(Shape2(1, dimension), d_zero);
    lpxr = NewTensor<xpu>(Shape2(1, dimension), d_zero);
    lpmy = NewTensor<xpu>(Shape2(1, dimension), d_zero);
  }


  inline void initial(Tensor<xpu, 2, dtype> rW1, Tensor<xpu, 2, dtype> rU1,
      Tensor<xpu, 2, dtype> rW2, Tensor<xpu, 2, dtype> rU2,
      Tensor<xpu, 2, dtype> uW1, Tensor<xpu, 2, dtype> uU1,
      Tensor<xpu, 2, dtype> uW2, Tensor<xpu, 2, dtype> uU2,
      Tensor<xpu, 2, dtype> uW3, Tensor<xpu, 2, dtype> uU3,
      Tensor<xpu, 2, dtype> W1, Tensor<xpu, 2, dtype> W2, Tensor<xpu, 2, dtype> W3,Tensor<xpu, 2, dtype> b) {
    _reset_left.initial(rW1, rU1, 1);
    _reset_right.initial(rW2, rU2, 1);

    _update_left.initial(uW1, uU1, 3);
    _update_right.initial(uW2, uU2, 3);
    _update_tilde.initial(uW3, uU3, 3);

    _recursive_tilde.initial(W1, W2, W3, b, 0);
  }

  inline void release() {
    _reset_left.release();
    _reset_right.release();

    _update_left.release();
    _update_right.release();
    _update_tilde.release();

    _recursive_tilde.release();

    FreeSpace(&nxl);
    FreeSpace(&nxr);
    FreeSpace(&sum);
    FreeSpace(&pxl);
    FreeSpace(&pxr);
    FreeSpace(&pmy);
    FreeSpace(&lnxl);
    FreeSpace(&lnxr);
    FreeSpace(&lsum);
    FreeSpace(&lpxl);
    FreeSpace(&lpxr);
    FreeSpace(&lpmy);
    FreeSpace(&lrxl);
    FreeSpace(&lrxr);
    FreeSpace(&lmy);
    FreeSpace(&luxl);
    FreeSpace(&luxr);
    FreeSpace(&lumy);
  }

  virtual ~AttRecursiveGatedNN() {
    // TODO Auto-generated destructor stub
  }

  inline dtype squarenormAll() {
    dtype norm = _reset_left.squarenormAll();
    norm += _reset_right.squarenormAll();
    norm += _update_left.squarenormAll();
    norm += _update_right.squarenormAll();
    norm += _update_tilde.squarenormAll();
    norm += _recursive_tilde.squarenormAll();

    return norm;
  }

  inline void scaleGrad(dtype scale) {
    _reset_left.scaleGrad(scale);
    _reset_right.scaleGrad(scale);

    _update_left.scaleGrad(scale);
    _update_right.scaleGrad(scale);
    _update_tilde.scaleGrad(scale);

    _recursive_tilde.scaleGrad(scale);
  }

public:

  inline void ComputeForwardScore(Tensor<xpu, 2, dtype> xl, Tensor<xpu, 2, dtype> xr, Tensor<xpu, 2, dtype> a,
      Tensor<xpu, 2, dtype> rxl, Tensor<xpu, 2, dtype> rxr, Tensor<xpu, 2, dtype> my,
      Tensor<xpu, 2, dtype> uxl, Tensor<xpu, 2, dtype> uxr, Tensor<xpu, 2, dtype> umy,
      Tensor<xpu, 2, dtype> y) {

    nxl = 0.0;
    nxr = 0.0;
    sum = 0.0;

    pxl = 0.0;
    pxr = 0.0;
    pmy = 0.0;

    _reset_left.ComputeForwardScore(xl, a, rxl);
    _reset_right.ComputeForwardScore(xr, a, rxr);


    nxl = rxl * xl;
    nxr = rxr * xr;

    _recursive_tilde.ComputeForwardScore(nxl, nxr, my);


    _update_left.ComputeForwardScore(xl, a, uxl);
    _update_right.ComputeForwardScore(xr, a, uxr);
    _update_tilde.ComputeForwardScore(my, a, umy);

    sum = uxl + uxr + umy;

    pxl = uxl / sum;
    pxr = uxr / sum;
    pmy = umy / sum;

    y = pxl * xl + pxr * xr + pmy * my;

  }

  //please allocate the memory outside here
  inline void ComputeBackwardLoss(Tensor<xpu, 2, dtype> xl, Tensor<xpu, 2, dtype> xr, Tensor<xpu, 2, dtype> a,
      Tensor<xpu, 2, dtype> rxl, Tensor<xpu, 2, dtype> rxr, Tensor<xpu, 2, dtype> my,
      Tensor<xpu, 2, dtype> uxl, Tensor<xpu, 2, dtype> uxr, Tensor<xpu, 2, dtype> umy,
      Tensor<xpu, 2, dtype> y, Tensor<xpu, 2, dtype> ly,
      Tensor<xpu, 2, dtype> lxl, Tensor<xpu, 2, dtype> lxr, Tensor<xpu, 2, dtype> la,
      bool bclear = false) {
    if (bclear){
      lxl = 0.0; lxr = 0.0; la = 0.0;
    }

    nxl = 0.0;
    nxr = 0.0;
    sum = 0.0;

    pxl = 0.0;
    pxr = 0.0;
    pmy = 0.0;


    lrxl = 0.0;
    lrxr = 0.0;
    lmy = 0.0;
    luxl = 0.0;
    luxr = 0.0;
    lumy = 0.0;

    lnxl = 0.0;
    lnxr = 0.0;
    lsum = 0.0;

    lpxl = 0.0;
    lpxr = 0.0;
    lpmy = 0.0;

    nxl = rxl * xl;
    nxr = rxr * xr;

    sum = uxl + uxr + umy;

    pxl = uxl / sum;
    pxr = uxr / sum;
    pmy = umy / sum;


    lpxl += ly * xl;
    lxl += ly * pxl;

    lpxr += ly * xr;
    lxr += ly * pxr;

    lpmy += ly * my;
    lmy += ly * pmy;



    luxl += lpxl / sum;
    luxr += lpxr / sum;
    lumy += lpmy / sum;

    lsum -= lpxl * pxl / sum;
    lsum -= lpxr * pxr / sum;
    lsum -= lpmy * pmy / sum;


    luxl += lsum;
    luxr += lsum;
    lumy += lsum;

    _update_left.ComputeBackwardLoss(xl, a, uxl, luxl, lxl, la);
    _update_right.ComputeBackwardLoss(xr, a, uxr, luxr, lxr, la);
    _update_tilde.ComputeBackwardLoss(my, a, umy, lumy, lmy, la);

    _recursive_tilde.ComputeBackwardLoss(nxl, nxr, my, lmy, lnxl, lnxr);

    lrxl += lnxl * xl;
    lxl += lnxl * rxl;

    lrxr += lnxr * xr;
    lxr += lnxr * rxr;

    _reset_left.ComputeBackwardLoss(xl, a, rxl, lrxl, lxl, la);
    _reset_right.ComputeBackwardLoss(xr, a, rxr, lrxr, lxr, la);

  }


  inline void randomprint(int num) {
    _reset_left.randomprint(num);
    _reset_right.randomprint(num);

    _update_left.randomprint(num);
    _update_right.randomprint(num);
    _update_tilde.randomprint(num);

    _recursive_tilde.randomprint(num);
  }

  inline void updateAdaGrad(dtype regularizationWeight, dtype adaAlpha, dtype adaEps) {
    _reset_left.updateAdaGrad(regularizationWeight, adaAlpha, adaEps);
    _reset_right.updateAdaGrad(regularizationWeight, adaAlpha, adaEps);

    _update_left.updateAdaGrad(regularizationWeight, adaAlpha, adaEps);
    _update_right.updateAdaGrad(regularizationWeight, adaAlpha, adaEps);
    _update_tilde.updateAdaGrad(regularizationWeight, adaAlpha, adaEps);

    _recursive_tilde.updateAdaGrad(regularizationWeight, adaAlpha, adaEps);
  }

  void writeModel(LStream &outf) {
    _reset_left.writeModel(outf);
    _reset_right.writeModel(outf);
    _update_left.writeModel(outf);
    _update_right.writeModel(outf);
    _update_tilde.writeModel(outf);
    _recursive_tilde.writeModel(outf);
    
    SaveBinary(outf, nxl);
    SaveBinary(outf, nxr);
    SaveBinary(outf, sum);

    SaveBinary(outf, pxl);
    SaveBinary(outf, pxr);
    SaveBinary(outf, pmy);

    SaveBinary(outf, lrxl);
    SaveBinary(outf, lrxr);
    SaveBinary(outf, lmy);
    SaveBinary(outf, luxl);
    SaveBinary(outf, luxr);
    SaveBinary(outf, lumy);

    SaveBinary(outf, lnxl);
    SaveBinary(outf, lnxr);
    SaveBinary(outf, lsum);

    SaveBinary(outf, lpxl);
    SaveBinary(outf, lpxr);
    SaveBinary(outf, lpmy);

  }

  void loadModel(LStream &inf) {

    _reset_left.loadModel(inf);
    _reset_right.loadModel(inf);
    _update_left.loadModel(inf);
    _update_right.loadModel(inf);
    _update_tilde.loadModel(inf);
    _recursive_tilde.loadModel(inf);


    LoadBinary(inf, &nxl, false);
    LoadBinary(inf, &nxr, false);
    LoadBinary(inf, &sum, false);

    LoadBinary(inf, &pxl, false);
    LoadBinary(inf, &pxr, false);
    LoadBinary(inf, &pmy, false);

    LoadBinary(inf, &lrxl, false);
    LoadBinary(inf, &lrxr, false);
    LoadBinary(inf, &lmy, false);
    LoadBinary(inf, &luxl, false);
    LoadBinary(inf, &luxr, false);
    LoadBinary(inf, &lumy, false);

    LoadBinary(inf, &lnxl, false);
    LoadBinary(inf, &lnxr, false);
    LoadBinary(inf, &lsum, false);

    LoadBinary(inf, &lpxl, false);
    LoadBinary(inf, &lpxr, false);
    LoadBinary(inf, &lpmy, false);

  }
};



#endif /* SRC_AttRecursiveGatedNN_H_ */
