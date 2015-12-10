/*
 * RNN.h
 *
 *  Created on: Mar 18, 2015
 *      Author: mszhang
 */

#ifndef SRC_RNN_H_
#define SRC_RNN_H_
#include "tensor.h"

#include "BiLayer.h"
#include "MyLib.h"
#include "Utiltensor.h"

using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;

template<typename xpu>
class RNN {
public:
  BiLayer<xpu> _rnn;
  bool _left2right;

  Tensor<xpu, 2, dtype> _null, _nullLoss;

public:
  RNN() {
  }

  inline void initial(int outputsize, int inputsize, int seed = 0) {
    _left2right = true;
    _rnn.initial(outputsize, outputsize, inputsize, true, seed, 0);

    _null = NewTensor<xpu>(Shape2(1, outputsize), d_zero);
    _nullLoss = NewTensor<xpu>(Shape2(1, outputsize), d_zero);

  }

  inline void initial(int outputsize, int inputsize, bool left2right, int seed = 0) {
    _left2right = left2right;
    _rnn.initial(outputsize, outputsize, inputsize, true, seed, 0);

    _null = NewTensor<xpu>(Shape2(1, outputsize), d_zero);
    _nullLoss = NewTensor<xpu>(Shape2(1, outputsize), d_zero);

  }

  inline void initial(Tensor<xpu, 2, dtype> WL, Tensor<xpu, 2, dtype> WR, Tensor<xpu, 2, dtype> b, bool left2right = true) {
    _left2right = left2right;
    _rnn.initial(WL, WR, b, true);

    _null = NewTensor<xpu>(Shape2(1, b.size(1)), d_zero);
    _nullLoss = NewTensor<xpu>(Shape2(1, b.size(1)), d_zero);
  }

  inline void release() {
    _rnn.release();

    FreeSpace(&_null);
    FreeSpace(&_nullLoss);
  }

  virtual ~RNN() {
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

  inline void ComputeForwardScore(Tensor<xpu, 3, dtype> x, Tensor<xpu, 3, dtype> y) {
    y = 0.0;
    int seq_size = x.size(0);
    if (seq_size == 0)
      return;

    if (_left2right) {
      for (int idx = 0; idx < seq_size; idx++) {
        if (idx == 0) {
          _rnn.ComputeForwardScore(_null, x[idx], y[idx]);
        } else
          _rnn.ComputeForwardScore(y[idx - 1], x[idx], y[idx]);
      }
    } else {
      for (int idx = seq_size - 1; idx >= 0; idx--) {
        if (idx == seq_size - 1)
          _rnn.ComputeForwardScore(_null, x[idx], y[idx]);
        else
          _rnn.ComputeForwardScore(y[idx + 1], x[idx], y[idx]);
      }
    }
  }

  inline void ComputeForwardScore(const vector<Tensor<xpu, 2, dtype> > &x, vector<Tensor<xpu, 2, dtype> > &y) {
    assign(y, 0.0);
    int seq_size = x.size();
    if (seq_size == 0)
      return;

    if (_left2right) {
      for (int idx = 0; idx < seq_size; idx++) {
        if (idx == 0) {
          _rnn.ComputeForwardScore(_null, x[idx], y[idx]);
        } else
          _rnn.ComputeForwardScore(y[idx - 1], x[idx], y[idx]);
      }
    } else {
      for (int idx = seq_size - 1; idx >= 0; idx--) {
        if (idx == seq_size - 1)
          _rnn.ComputeForwardScore(_null, x[idx], y[idx]);
        else
          _rnn.ComputeForwardScore(y[idx + 1], x[idx], y[idx]);
      }
    }
  }

  // This function is used for computing hidden values incrementally at the start position
  // It is applied only when the sequential inputs are not fixed in advance,
  // which can vary during decoding.
  // We need not provide a backward function, since during backward, inputs will be given.
  inline void ComputeForwardScoreIncremental(Tensor<xpu, 2, dtype> x, Tensor<xpu, 2, dtype> y) {
    assert(_left2right);
    y = 0.0;
    _rnn.ComputeForwardScore(_null, x, y);
  }


  // This function is used for computing hidden values incrementally at the non-start position
  // It is applied only when the sequential inputs are not fixed in advance,
  // which can vary during decoding.
  // We need not provide a backward function, since during backward, inputs will be given.
  inline void ComputeForwardScoreIncremental(Tensor<xpu, 2, dtype> py, Tensor<xpu, 2, dtype> x, Tensor<xpu, 2, dtype> y) {
    assert(_left2right);
    y = 0.0;
    _rnn.ComputeForwardScore(py, x, y);
  }

  //please allocate the memory outside here
  inline void ComputeBackwardLoss(Tensor<xpu, 3, dtype> x, Tensor<xpu, 3, dtype> y, Tensor<xpu, 3, dtype> ly, Tensor<xpu, 3, dtype> lx, bool bclear = false) {
    int seq_size = x.size(0);
    if (seq_size == 0)
      return;

    if (bclear)
      lx = 0.0;
    //left rnn
    Tensor<xpu, 3, dtype> lfy = NewTensor<xpu>(Shape3(y.size(0), y.size(1), y.size(2)), d_zero);
    if (_left2right) {
      for (int idx = seq_size - 1; idx >= 0; idx--) {
        if (idx < seq_size - 1)
          ly[idx] = ly[idx] + lfy[idx];

        if (idx == 0)
          _rnn.ComputeBackwardLoss(_null, x[idx], y[idx], ly[idx], _nullLoss, lx[idx]);
        else
          _rnn.ComputeBackwardLoss(y[idx - 1], x[idx], y[idx], ly[idx], lfy[idx - 1], lx[idx]);
      }
    } else {
      // right rnn
      for (int idx = 0; idx < seq_size; idx++) {
        if (idx > 0)
          ly[idx] = ly[idx] + lfy[idx];

        if (idx == seq_size - 1)
          _rnn.ComputeBackwardLoss(_null, x[idx], y[idx], ly[idx], _nullLoss, lx[idx]);
        else
          _rnn.ComputeBackwardLoss(y[idx + 1], x[idx], y[idx], ly[idx], lfy[idx + 1], lx[idx]);
      }
    }

    FreeSpace(&lfy);
  }

  //please allocate the memory outside here
  inline void ComputeBackwardLoss(const vector<Tensor<xpu, 2, dtype> > &x, const vector<Tensor<xpu, 2, dtype> > &y,
      vector<Tensor<xpu, 2, dtype> > &ly, vector<Tensor<xpu, 2, dtype> > &lx, bool bclear = false) {
    int seq_size = x.size();
    if (seq_size == 0)
      return;

    if (bclear)
      assign(lx, 0.0);

    vector<Tensor<xpu, 2, dtype> > lfy(seq_size);
    for (int idx = 0; idx < seq_size; idx++) {
      lfy[idx] = NewTensor<xpu>(Shape2(ly[0].size(0), ly[0].size(1)), d_zero);
    }

    if (_left2right) {
      //left rnn
      for (int idx = seq_size - 1; idx >= 0; idx--) {
        if (idx < seq_size - 1)
          ly[idx] = ly[idx] + lfy[idx];

        if (idx == 0)
          _rnn.ComputeBackwardLoss(_null, x[idx], y[idx], ly[idx], _nullLoss, lx[idx]);
        else
          _rnn.ComputeBackwardLoss(y[idx - 1], x[idx], y[idx], ly[idx], lfy[idx - 1], lx[idx]);
      }
    } else {
      // right rnn
      for (int idx = 0; idx < seq_size; idx++) {
        if (idx > 0)
          ly[idx] = ly[idx] + lfy[idx];

        if (idx == seq_size - 1)
          _rnn.ComputeBackwardLoss(_null, x[idx], y[idx], ly[idx], _nullLoss, lx[idx]);
        else
          _rnn.ComputeBackwardLoss(y[idx + 1], x[idx], y[idx], ly[idx], lfy[idx + 1], lx[idx]);
      }
    }

    for (int idx = 0; idx < seq_size; idx++) {
      FreeSpace(&(lfy[idx]));
    }
  }

  inline void randomprint(int num) {
    _rnn.randomprint(num);
  }

  inline void updateAdaGrad(dtype regularizationWeight, dtype adaAlpha, dtype adaEps) {
    _rnn.updateAdaGrad(regularizationWeight, adaAlpha, adaEps);
  }

  void writeModel(LStream &outf) {
    _rnn.writeModel(outf);

    SaveBinary(outf, _null);
    SaveBinary(outf, _nullLoss);

    WriteBinary(outf, _left2right);
  }

  void loadModel(LStream &inf) {
    _rnn.loadModel(inf);
    LoadBinary(inf, &_null, false);
    LoadBinary(inf, &_nullLoss, false);

    ReadBinary(inf, _left2right);
  }

};

#endif /* SRC_RNN_H_ */
