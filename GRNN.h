/*
 * GRNN.h
 *
 *  Created on: Mar 18, 2015
 *      Author: mszhang
 */

#ifndef SRC_GRNN_H_
#define SRC_GRNN_H_
#include "tensor.h"

#include "BiLayer.h"
#include "MyLib.h"
#include "Utiltensor.h"

using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;


template<typename xpu>
class GRNN {
public:
  BiLayer<xpu> _rnn_update;
  BiLayer<xpu> _rnn_reset;
  BiLayer<xpu> _rnn;
  bool _left2right;

  Tensor<xpu, 2, dtype> _null, _nullLoss;

public:
  GRNN() {
  }

  inline void initial(int outputsize, int inputsize, int seed = 0) {
    _left2right = true;

    _rnn_update.initial(outputsize, outputsize, inputsize, true, seed, 1);
    _rnn_reset.initial(outputsize, outputsize, inputsize, true, seed + 10, 1);
    _rnn.initial(outputsize, outputsize, inputsize, true, seed + 20, 0);

    _null = NewTensor<xpu>(Shape2(1, outputsize), d_zero);
    _nullLoss = NewTensor<xpu>(Shape2(1, outputsize), d_zero);

  }

  inline void initial(int outputsize, int inputsize, bool left2right, int seed = 0) {
    _left2right = left2right;

    _rnn_update.initial(outputsize, outputsize, inputsize, true, seed, 1);
    _rnn_reset.initial(outputsize, outputsize, inputsize, true, seed + 10, 1);
    _rnn.initial(outputsize, outputsize, inputsize, true, seed + 20, 0);

    _null = NewTensor<xpu>(Shape2(1, outputsize), d_zero);
    _nullLoss = NewTensor<xpu>(Shape2(1, outputsize), d_zero);

  }

  inline void initial(Tensor<xpu, 2, dtype> WL, Tensor<xpu, 2, dtype> WR, Tensor<xpu, 2, dtype> b, Tensor<xpu, 2, dtype> uWL, Tensor<xpu, 2, dtype> uWR,
      Tensor<xpu, 2, dtype> ub, Tensor<xpu, 2, dtype> rWL, Tensor<xpu, 2, dtype> rWR, Tensor<xpu, 2, dtype> rb, bool left2right = true) {
    _left2right = left2right;

    _rnn_update.initial(uWL, uWR, ub, true, 1);
    _rnn_reset.initial(rWL, rWR, rb, true, 1);
    _rnn.initial(WL, WR, b, true);

    _null = NewTensor<xpu>(Shape2(1, b.size(1)), d_zero);
    _nullLoss = NewTensor<xpu>(Shape2(1, b.size(1)), d_zero);
  }

  inline void release() {
    _rnn_update.release();
    _rnn_reset.release();
    _rnn.release();

    FreeSpace(&_null);
    FreeSpace(&_nullLoss);
  }

  virtual ~GRNN() {
    // TODO Auto-generated destructor stub
  }

  inline dtype squarenormAll() {
    dtype norm = _rnn_update.squarenormAll();
    norm += _rnn_reset.squarenormAll();
    norm += _rnn.squarenormAll();

    return norm;
  }

  inline void scaleGrad(dtype scale) {
    _rnn_update.scaleGrad(scale);
    _rnn_reset.scaleGrad(scale);
    _rnn.scaleGrad(scale);
  }

public:

  inline void ComputeForwardScore(Tensor<xpu, 3, dtype> x, Tensor<xpu, 3, dtype> mry, Tensor<xpu, 3, dtype> ry, Tensor<xpu, 3, dtype> uy,
      Tensor<xpu, 3, dtype> cy, Tensor<xpu, 3, dtype> y) {
    mry = 0.0;
    ry = 0.0;
    uy = 0.0;
    cy = 0.0;
    y = 0.0;
    int seq_size = x.size(0);
    if (seq_size == 0)
      return;

    if (_left2right) {
      for (int idx = 0; idx < seq_size; idx++) {
        if (idx == 0) {
          _rnn_update.ComputeForwardScore(_null, x[idx], uy[idx]);
          _rnn.ComputeForwardScore(_null, x[idx], cy[idx]);
          y[idx] = uy[idx] * cy[idx];
        } else {
          _rnn_reset.ComputeForwardScore(y[idx - 1], x[idx], mry[idx]);
          ry[idx] = mry[idx] * y[idx - 1];
          _rnn_update.ComputeForwardScore(y[idx - 1], x[idx], uy[idx]);
          _rnn.ComputeForwardScore(ry[idx], x[idx], cy[idx]);
          y[idx] = (1.0 - uy[idx]) * y[idx - 1] + uy[idx] * cy[idx];
        }
      }
    } else {
      for (int idx = seq_size - 1; idx >= 0; idx--) {
        if (idx == seq_size - 1) {
          _rnn_update.ComputeForwardScore(_null, x[idx], uy[idx]);
          _rnn.ComputeForwardScore(_null, x[idx], cy[idx]);
          y[idx] = uy[idx] * cy[idx];
        } else {
          _rnn_reset.ComputeForwardScore(y[idx + 1], x[idx], mry[idx]);
          ry[idx] = mry[idx] * y[idx + 1];
          _rnn_update.ComputeForwardScore(y[idx + 1], x[idx], uy[idx]);
          _rnn.ComputeForwardScore(ry[idx], x[idx], cy[idx]);
          y[idx] = (1.0 - uy[idx]) * y[idx + 1] + uy[idx] * cy[idx];
        }
      }
    }
  }

  inline void ComputeForwardScore(const vector<Tensor<xpu, 2, dtype> > &x, vector<Tensor<xpu, 2, dtype> > &mry, vector<Tensor<xpu, 2, dtype> > &ry,
      vector<Tensor<xpu, 2, dtype> > &uy, vector<Tensor<xpu, 2, dtype> > &cy, vector<Tensor<xpu, 2, dtype> > &y) {
    assign(mry, 0.0);
    assign(ry, 0.0);
    assign(uy, 0.0);
    assign(cy, 0.0);
    assign(y, 0.0);
    int seq_size = x.size();
    if (seq_size == 0)
      return;

    if (_left2right) {
      for (int idx = 0; idx < seq_size; idx++) {
        if (idx == 0) {
          _rnn_update.ComputeForwardScore(_null, x[idx], uy[idx]);
          _rnn.ComputeForwardScore(_null, x[idx], cy[idx]);
          y[idx] = uy[idx] * cy[idx];
        } else {
          _rnn_reset.ComputeForwardScore(y[idx - 1], x[idx], mry[idx]);
          ry[idx] = mry[idx] * y[idx - 1];
          _rnn_update.ComputeForwardScore(y[idx - 1], x[idx], uy[idx]);
          _rnn.ComputeForwardScore(ry[idx], x[idx], cy[idx]);
          y[idx] = (1.0 - uy[idx]) * y[idx - 1] + uy[idx] * cy[idx];
        }
      }
    } else {
      for (int idx = seq_size - 1; idx >= 0; idx--) {
        if (idx == seq_size - 1) {
          _rnn_update.ComputeForwardScore(_null, x[idx], uy[idx]);
          _rnn.ComputeForwardScore(_null, x[idx], cy[idx]);
          y[idx] = uy[idx] * cy[idx];
        } else {
          _rnn_reset.ComputeForwardScore(y[idx + 1], x[idx], mry[idx]);
          ry[idx] = mry[idx] * y[idx + 1];
          _rnn_update.ComputeForwardScore(y[idx + 1], x[idx], uy[idx]);
          _rnn.ComputeForwardScore(ry[idx], x[idx], cy[idx]);
          y[idx] = (1.0 - uy[idx]) * y[idx + 1] + uy[idx] * cy[idx];
        }
      }
    }
  }


  // This function is used for computing hidden values incrementally at the start position
  // It is applied only when the sequential inputs are not fixed in advance,
  // which can vary during decoding.
  // We need not provide a backward function, since during backward, inputs will be given.
  inline void ComputeForwardScoreIncremental(Tensor<xpu, 2, dtype> x, Tensor<xpu, 2, dtype> mry, Tensor<xpu, 2, dtype> ry,
      Tensor<xpu, 2, dtype> uy, Tensor<xpu, 2, dtype> cy, Tensor<xpu, 2, dtype> y) {
    assert(_left2right);
    _rnn_update.ComputeForwardScore(_null, x, uy);
    _rnn.ComputeForwardScore(_null, x, cy);
    y = uy * cy;
  }


  // This function is used for computing hidden values incrementally at the non-start position
  // It is applied only when the sequential inputs are not fixed in advance,
  // which can vary during decoding.
  // We need not provide a backward function, since during backward, inputs will be given.
  inline void ComputeForwardScoreIncremental(Tensor<xpu, 2, dtype> py, Tensor<xpu, 2, dtype> x, Tensor<xpu, 2, dtype> mry, Tensor<xpu, 2, dtype> ry,
      Tensor<xpu, 2, dtype> uy, Tensor<xpu, 2, dtype> cy, Tensor<xpu, 2, dtype> y) {
    assert(_left2right);
    _rnn_reset.ComputeForwardScore(py, x, mry);
    ry = mry * py;
    _rnn_update.ComputeForwardScore(py, x, uy);
    _rnn.ComputeForwardScore(ry, x, cy);
    y = (1.0 - uy) * py + uy * cy;
  }

  //please allocate the memory outside here
  inline void ComputeBackwardLoss(Tensor<xpu, 3, dtype> x, Tensor<xpu, 3, dtype> mry, Tensor<xpu, 3, dtype> ry, Tensor<xpu, 3, dtype> uy,
      Tensor<xpu, 3, dtype> cy, Tensor<xpu, 3, dtype> y, Tensor<xpu, 3, dtype> ly, Tensor<xpu, 3, dtype> lx, bool bclear = false) {
    int seq_size = x.size(0);
    if (seq_size == 0)
      return;

    if (bclear)
      lx = 0.0;
    //left rnn
    Tensor<xpu, 3, dtype> lfy = NewTensor<xpu>(Shape3(y.size(0), y.size(1), y.size(2)), d_zero);
    Tensor<xpu, 3, dtype> luy = NewTensor<xpu>(Shape3(y.size(0), y.size(1), y.size(2)), d_zero);
    Tensor<xpu, 3, dtype> lcy = NewTensor<xpu>(Shape3(y.size(0), y.size(1), y.size(2)), d_zero);
    Tensor<xpu, 3, dtype> lry = NewTensor<xpu>(Shape3(y.size(0), y.size(1), y.size(2)), d_zero);
    Tensor<xpu, 3, dtype> lmry = NewTensor<xpu>(Shape3(y.size(0), y.size(1), y.size(2)), d_zero);

    if (_left2right) {
      for (int idx = seq_size - 1; idx >= 0; idx--) {
        if (idx < seq_size - 1)
          ly[idx] = ly[idx] + lfy[idx];

        if (idx == 0) {
          luy[idx] = ly[idx] * cy[idx];
          lcy[idx] = ly[idx] * uy[idx];

          _rnn.ComputeBackwardLoss(_null, x[idx], cy[idx], lcy[idx], _nullLoss, lx[idx]);

          _rnn_update.ComputeBackwardLoss(_null, x[idx], uy[idx], luy[idx], _nullLoss, lx[idx]);
        } else {
          luy[idx] = ly[idx] * (cy[idx] - y[idx - 1]);
          lfy[idx - 1] = ly[idx] * (1.0 - uy[idx]);
          lcy[idx] = ly[idx] * uy[idx];

          _rnn.ComputeBackwardLoss(ry[idx], x[idx], cy[idx], lcy[idx], lry[idx], lx[idx]);
          _rnn_update.ComputeBackwardLoss(y[idx - 1], x[idx], uy[idx], luy[idx], lfy[idx - 1], lx[idx]);

          lmry[idx] = lry[idx] * y[idx - 1];
          lfy[idx - 1] += lry[idx] * mry[idx];

          _rnn_reset.ComputeBackwardLoss(y[idx - 1], x[idx], mry[idx], lmry[idx], lfy[idx - 1], lx[idx]);
        }
      }
    } else {
      // right rnn
      for (int idx = 0; idx < seq_size; idx++) {
        if (idx > 0)
          ly[idx] = ly[idx] + lfy[idx];

        if (idx == seq_size - 1) {
          luy[idx] = ly[idx] * cy[idx];
          lcy[idx] = ly[idx] * uy[idx];

          _rnn.ComputeBackwardLoss(_null, x[idx], cy[idx], lcy[idx], _nullLoss, lx[idx]);
          _rnn_update.ComputeBackwardLoss(_null, x[idx], uy[idx], luy[idx], _nullLoss, lx[idx]);
        } else {
          luy[idx] = ly[idx] * (cy[idx] - y[idx + 1]);
          lfy[idx + 1] = ly[idx] * (1.0 - uy[idx]);
          lcy[idx] = ly[idx] * uy[idx];

          _rnn.ComputeBackwardLoss(ry[idx], x[idx], cy[idx], lcy[idx], lry[idx], lx[idx]);
          _rnn_update.ComputeBackwardLoss(y[idx + 1], x[idx], uy[idx], luy[idx], lfy[idx + 1], lx[idx]);

          lmry[idx] = lry[idx] * y[idx + 1];
          lfy[idx + 1] += lry[idx] * mry[idx];

          _rnn_reset.ComputeBackwardLoss(y[idx + 1], x[idx], mry[idx], lmry[idx], lfy[idx + 1], lx[idx]);
        }
      }
    }

    FreeSpace(&lfy);
    FreeSpace(&luy);
    FreeSpace(&lcy);
    FreeSpace(&lry);
    FreeSpace(&lmry);
  }

  //please allocate the memory outside here
  inline void ComputeBackwardLoss(const vector<Tensor<xpu, 2, dtype> > &x, const vector<Tensor<xpu, 2, dtype> > &mry, const vector<Tensor<xpu, 2, dtype> > &ry,
      const vector<Tensor<xpu, 2, dtype> > &uy, const vector<Tensor<xpu, 2, dtype> > &cy, const vector<Tensor<xpu, 2, dtype> > &y, 
      vector<Tensor<xpu, 2, dtype> > &ly, vector<Tensor<xpu, 2, dtype> > &lx, bool bclear = false) {
    int seq_size = x.size();
    if (seq_size == 0)
      return;

    if (bclear)
      assign(lx, 0.0);

    vector<Tensor<xpu, 2, dtype> > lfy(seq_size), lcy(seq_size), luy(seq_size), lry(seq_size), lmry(seq_size);
    for (int idx = 0; idx < seq_size; idx++) {
      lfy[idx] = NewTensor<xpu>(Shape2(ly[0].size(0), ly[0].size(1)), d_zero);
      lcy[idx] = NewTensor<xpu>(Shape2(ly[0].size(0), ly[0].size(1)), d_zero);
      luy[idx] = NewTensor<xpu>(Shape2(ly[0].size(0), ly[0].size(1)), d_zero);
      lry[idx] = NewTensor<xpu>(Shape2(ly[0].size(0), ly[0].size(1)), d_zero);
      lmry[idx] = NewTensor<xpu>(Shape2(ly[0].size(0), ly[0].size(1)), d_zero);
    }

    if (_left2right) {
      for (int idx = seq_size - 1; idx >= 0; idx--) {
        if (idx < seq_size - 1)
          ly[idx] = ly[idx] + lfy[idx];

        if (idx == 0) {
          luy[idx] = ly[idx] * cy[idx];
          lcy[idx] = ly[idx] * uy[idx];

          _rnn.ComputeBackwardLoss(_null, x[idx], cy[idx], lcy[idx], _nullLoss, lx[idx]);

          _rnn_update.ComputeBackwardLoss(_null, x[idx], uy[idx], luy[idx], _nullLoss, lx[idx]);
        } else {
          luy[idx] = ly[idx] * (cy[idx] - y[idx - 1]);
          lfy[idx - 1] = ly[idx] * (1.0 - uy[idx]);
          lcy[idx] = ly[idx] * uy[idx];

          _rnn.ComputeBackwardLoss(ry[idx], x[idx], cy[idx], lcy[idx], lry[idx], lx[idx]);
          _rnn_update.ComputeBackwardLoss(y[idx - 1], x[idx], uy[idx], luy[idx], lfy[idx - 1], lx[idx]);

          lmry[idx] = lry[idx] * y[idx - 1];
          lfy[idx - 1] += lry[idx] * mry[idx];

          _rnn_reset.ComputeBackwardLoss(y[idx - 1], x[idx], mry[idx], lmry[idx], lfy[idx - 1], lx[idx]);
        }
      }
    } else {
      // right rnn
      for (int idx = 0; idx < seq_size; idx++) {
        if (idx > 0)
          ly[idx] = ly[idx] + lfy[idx];

        if (idx == seq_size - 1) {
          luy[idx] = ly[idx] * cy[idx];
          lcy[idx] = ly[idx] * uy[idx];

          _rnn.ComputeBackwardLoss(_null, x[idx], cy[idx], lcy[idx], _nullLoss, lx[idx]);
          _rnn_update.ComputeBackwardLoss(_null, x[idx], uy[idx], luy[idx], _nullLoss, lx[idx]);
        } else {
          luy[idx] = ly[idx] * (cy[idx] - y[idx + 1]);
          lfy[idx + 1] = ly[idx] * (1.0 - uy[idx]);
          lcy[idx] = ly[idx] * uy[idx];

          _rnn.ComputeBackwardLoss(ry[idx], x[idx], cy[idx], lcy[idx], lry[idx], lx[idx]);
          _rnn_update.ComputeBackwardLoss(y[idx + 1], x[idx], uy[idx], luy[idx], lfy[idx + 1], lx[idx]);

          lmry[idx] = lry[idx] * y[idx + 1];
          lfy[idx + 1] += lry[idx] * mry[idx];

          _rnn_reset.ComputeBackwardLoss(y[idx + 1], x[idx], mry[idx], lmry[idx], lfy[idx + 1], lx[idx]);
        }
      }
    }

    for (int idx = 0; idx < seq_size; idx++) {
      FreeSpace(&(lfy[idx]));
      FreeSpace(&(lcy[idx]));
      FreeSpace(&(luy[idx]));
      FreeSpace(&(lry[idx]));
      FreeSpace(&(lmry[idx]));
    }
  }

  inline void randomprint(int num) {
    _rnn_update.randomprint(num);
    _rnn_reset.randomprint(num);
    _rnn.randomprint(num);
  }

  inline void updateAdaGrad(dtype regularizationWeight, dtype adaAlpha, dtype adaEps) {
    _rnn_update.updateAdaGrad(regularizationWeight, adaAlpha, adaEps);
    _rnn_reset.updateAdaGrad(regularizationWeight, adaAlpha, adaEps);
    _rnn.updateAdaGrad(regularizationWeight, adaAlpha, adaEps);
  }

  void writeModel(LStream &outf) {
    _rnn_update.writeModel(outf);
    _rnn_reset.writeModel(outf);
    _rnn.writeModel(outf);
    
    WriteBinary(outf, _left2right);

    SaveBinary(outf, _null);
    SaveBinary(outf, _nullLoss);
  }

  void loadModel(LStream &inf) {
    _rnn_update.loadModel(inf);
    _rnn_reset.loadModel(inf);
    _rnn.loadModel(inf);

    ReadBinary(inf, _left2right);

    LoadBinary(inf, &_null, false);
    LoadBinary(inf, &_nullLoss, false);
  }
};

#endif /* SRC_GRNN_H_ */
