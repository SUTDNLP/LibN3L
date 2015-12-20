/*
 * LSTM_CHD.h
 *
 *  Created on: Mar 18, 2015
 *      Author: mszhang
 */

#ifndef SRC_LSTM_CHD_H_
#define SRC_LSTM_CHD_H_
#include "tensor.h"

#include "BiLayer.h"
#include "MyLib.h"
#include "Utiltensor.h"
#include "TriLayer.h"

using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;


template<typename xpu>
class LSTM_CHD {
public:
  TriLayer<xpu> _lstm_output;
  TriLayer<xpu> _lstm_input;
  BiLayer<xpu> _lstm_cell;
  bool _left2right;

  Tensor<xpu, 2, dtype> _null1, _null1Loss, _null2, _null2Loss;

public:
  LSTM_CHD() {
  }

  inline void initial(int outputsize, int inputsize, int seed = 0) {
    _left2right = true;

    _lstm_output.initial(outputsize, outputsize, inputsize, outputsize, true, seed, 1);
    _lstm_input.initial(outputsize, outputsize, inputsize, outputsize, true, seed + 10, 1);
    _lstm_cell.initial(outputsize, outputsize, inputsize, true, seed + 30, 0);

    _null1 = NewTensor<xpu>(Shape2(1, outputsize), d_zero);
    _null1Loss = NewTensor<xpu>(Shape2(1, outputsize), d_zero);
    _null2 = NewTensor<xpu>(Shape2(1, outputsize), d_zero);
    _null2Loss = NewTensor<xpu>(Shape2(1, outputsize), d_zero);

  }

  inline void initial(int outputsize, int inputsize, bool left2right, int seed = 0) {
    _left2right = left2right;

    _lstm_output.initial(outputsize, outputsize, inputsize, outputsize, true, seed, 1);
    _lstm_input.initial(outputsize, outputsize, inputsize, outputsize, true, seed + 10, 1);
    _lstm_cell.initial(outputsize, outputsize, inputsize, true, seed + 30, 0);

    _null1 = NewTensor<xpu>(Shape2(1, outputsize), d_zero);
    _null1Loss = NewTensor<xpu>(Shape2(1, outputsize), d_zero);
    _null2 = NewTensor<xpu>(Shape2(1, outputsize), d_zero);
    _null2Loss = NewTensor<xpu>(Shape2(1, outputsize), d_zero);

  }

  inline void initial(Tensor<xpu, 2, dtype> cWL, Tensor<xpu, 2, dtype> cWR, Tensor<xpu, 2, dtype> cb, Tensor<xpu, 2, dtype> oW1, Tensor<xpu, 2, dtype> oW2,
      Tensor<xpu, 2, dtype> oW3, Tensor<xpu, 2, dtype> ob, Tensor<xpu, 2, dtype> iW1, Tensor<xpu, 2, dtype> iW2, Tensor<xpu, 2, dtype> iW3,
      Tensor<xpu, 2, dtype> ib, bool left2right = true) {
    _left2right = left2right;

    _lstm_output.initial(oW1, oW2, oW3, ob, true, 1);
    _lstm_input.initial(iW1, iW2, iW3, ib, true, 1);
    _lstm_cell.initial(cWL, cWR, cb, true);

    _null1 = NewTensor<xpu>(Shape2(1, ob.size(1)), d_zero);
    _null1Loss = NewTensor<xpu>(Shape2(1, ob.size(1)), d_zero);
    _null2 = NewTensor<xpu>(Shape2(1, ob.size(1)), d_zero);
    _null2Loss = NewTensor<xpu>(Shape2(1, ob.size(1)), d_zero);
  }

  inline void release() {
    _lstm_output.release();
    _lstm_input.release();
    _lstm_cell.release();

    FreeSpace(&_null1);
    FreeSpace(&_null1Loss);
    FreeSpace(&_null2);
    FreeSpace(&_null2Loss);
  }

  virtual ~LSTM_CHD() {
    // TODO Auto-generated destructor stub
  }

  inline dtype squarenormAll() {
    dtype norm = _lstm_output.squarenormAll();
    norm += _lstm_input.squarenormAll();
    norm += _lstm_cell.squarenormAll();

    return norm;
  }

  inline void scaleGrad(dtype scale) {
    _lstm_output.scaleGrad(scale);
    _lstm_input.scaleGrad(scale);
    _lstm_cell.scaleGrad(scale);
  }

public:

  inline void ComputeForwardScore(Tensor<xpu, 3, dtype> x, Tensor<xpu, 3, dtype> iy, Tensor<xpu, 3, dtype> oy, Tensor<xpu, 3, dtype> fy,
      Tensor<xpu, 3, dtype> mcy, Tensor<xpu, 3, dtype> cy, Tensor<xpu, 3, dtype> my, Tensor<xpu, 3, dtype> y) {
    iy = 0.0;
    oy = 0.0;
    fy = 0.0;
    mcy = 0.0;
    cy = 0.0;
    my = 0.0;
    y = 0.0;
    int seq_size = x.size(0);
    if (seq_size == 0)
      return;

    if (_left2right) {
      for (int idx = 0; idx < seq_size; idx++) {
        if (idx == 0) {
          _lstm_input.ComputeForwardScore(_null1, x[idx], _null2,  iy[idx]);
          _lstm_cell.ComputeForwardScore(_null1, x[idx], mcy[idx]);
          cy[idx] = mcy[idx] * iy[idx];
          _lstm_output.ComputeForwardScore(_null1, x[idx], cy[idx], oy[idx]);
          my[idx] = F<nl_tanh>(cy[idx]);
          y[idx] = my[idx] * oy[idx];
        } else {
          _lstm_input.ComputeForwardScore(y[idx - 1], x[idx], cy[idx - 1], iy[idx]);
          fy[idx] = 1 - iy[idx];
          _lstm_cell.ComputeForwardScore(y[idx - 1], x[idx], mcy[idx]);
          cy[idx] = mcy[idx] * iy[idx] + cy[idx - 1] * fy[idx];
          _lstm_output.ComputeForwardScore(y[idx - 1], x[idx], cy[idx], oy[idx]);
          my[idx] = F<nl_tanh>(cy[idx]);
          y[idx] = my[idx] * oy[idx];
        }
      }
    } else {
      for (int idx = seq_size - 1; idx >= 0; idx--) {
        if (idx == seq_size - 1) {
          _lstm_input.ComputeForwardScore(_null1, x[idx], _null2,  iy[idx]);
          _lstm_cell.ComputeForwardScore(_null1, x[idx], mcy[idx]);
          cy[idx] = mcy[idx] * iy[idx];
          _lstm_output.ComputeForwardScore(_null1, x[idx], cy[idx], oy[idx]);
          my[idx] = F<nl_tanh>(cy[idx]);
          y[idx] = my[idx] * oy[idx];
        } else {
          _lstm_input.ComputeForwardScore(y[idx + 1], x[idx], cy[idx + 1], iy[idx]);
          fy[idx] = 1 - iy[idx];
          _lstm_cell.ComputeForwardScore(y[idx + 1], x[idx], mcy[idx]);
          cy[idx] = mcy[idx] * iy[idx] + cy[idx + 1] * fy[idx];
          _lstm_output.ComputeForwardScore(y[idx + 1], x[idx], cy[idx], oy[idx]);
          my[idx] = F<nl_tanh>(cy[idx]);
          y[idx] = my[idx] * oy[idx];
        }
      }
    }
  }
  
  inline void ComputeForwardScore(const vector<Tensor<xpu, 2, dtype> > &x, vector<Tensor<xpu, 2, dtype> > &iy, 
      vector<Tensor<xpu, 2, dtype> > &oy, vector<Tensor<xpu, 2, dtype> > &fy, vector<Tensor<xpu, 2, dtype> > &mcy, 
      vector<Tensor<xpu, 2, dtype> > &cy, vector<Tensor<xpu, 2, dtype> > &my, vector<Tensor<xpu, 2, dtype> > &y) {
    assign(iy, 0.0);
    assign(oy, 0.0);
    assign(fy, 0.0);
    assign(mcy, 0.0);
    assign(cy, 0.0);
    assign(my, 0.0);
    assign(y, 0.0);
    int seq_size = x.size();
    if (seq_size == 0)
      return;

    if (_left2right) {
      for (int idx = 0; idx < seq_size; idx++) {
        if (idx == 0) {
          _lstm_input.ComputeForwardScore(_null1, x[idx], _null2,  iy[idx]);
          _lstm_cell.ComputeForwardScore(_null1, x[idx], mcy[idx]);
          cy[idx] = mcy[idx] * iy[idx];
          _lstm_output.ComputeForwardScore(_null1, x[idx], cy[idx], oy[idx]);
          my[idx] = F<nl_tanh>(cy[idx]);
          y[idx] = my[idx] * oy[idx];
        } else {
          _lstm_input.ComputeForwardScore(y[idx - 1], x[idx], cy[idx - 1], iy[idx]);
          fy[idx] = 1 - iy[idx];
          _lstm_cell.ComputeForwardScore(y[idx - 1], x[idx], mcy[idx]);
          cy[idx] = mcy[idx] * iy[idx] + cy[idx - 1] * fy[idx];
          _lstm_output.ComputeForwardScore(y[idx - 1], x[idx], cy[idx], oy[idx]);
          my[idx] = F<nl_tanh>(cy[idx]);
          y[idx] = my[idx] * oy[idx];
        }
      }
    } else {
      for (int idx = seq_size - 1; idx >= 0; idx--) {
        if (idx == seq_size - 1) {
          _lstm_input.ComputeForwardScore(_null1, x[idx], _null2,  iy[idx]);
          _lstm_cell.ComputeForwardScore(_null1, x[idx], mcy[idx]);
          cy[idx] = mcy[idx] * iy[idx];
          _lstm_output.ComputeForwardScore(_null1, x[idx], cy[idx], oy[idx]);
          my[idx] = F<nl_tanh>(cy[idx]);
          y[idx] = my[idx] * oy[idx];
        } else {
          _lstm_input.ComputeForwardScore(y[idx + 1], x[idx], cy[idx + 1], iy[idx]);
          fy[idx] = 1 - iy[idx];
          _lstm_cell.ComputeForwardScore(y[idx + 1], x[idx], mcy[idx]);
          cy[idx] = mcy[idx] * iy[idx] + cy[idx + 1] * fy[idx];
          _lstm_output.ComputeForwardScore(y[idx + 1], x[idx], cy[idx], oy[idx]);
          my[idx] = F<nl_tanh>(cy[idx]);
          y[idx] = my[idx] * oy[idx];
        }
      }
    }
  }


  // This function is used for computing hidden values incrementally at the start position
  // It is applied only when the sequential inputs are not fixed in advance,
  // which can vary during decoding.
  // We need not provide a backward function, since during backward, inputs will be given.
  inline void ComputeForwardScoreIncremental(Tensor<xpu, 2, dtype> x, Tensor<xpu, 2, dtype> iy, Tensor<xpu, 2, dtype> oy, Tensor<xpu, 2, dtype> fy,
      Tensor<xpu, 2, dtype> mcy, Tensor<xpu, 2, dtype> cy, Tensor<xpu, 2, dtype> my, Tensor<xpu, 2, dtype> y) {
    assert(_left2right);
    _lstm_input.ComputeForwardScore(_null1, x, _null2, iy);
    _lstm_cell.ComputeForwardScore(_null1, x, mcy);
    cy = mcy * iy;
    _lstm_output.ComputeForwardScore(_null1, x, cy, oy);
    my = F<nl_tanh>(cy);
    y = my * oy;
  }


  // This function is used for computing hidden values incrementally at the non-start position
  // It is applied only when the sequential inputs are not fixed in advance,
  // which can vary during decoding.
  // We need not provide a backward function, since during backward, inputs will be given.
  inline void ComputeForwardScoreIncremental(Tensor<xpu, 2, dtype> pcy, Tensor<xpu, 2, dtype> py, Tensor<xpu, 2, dtype> x, Tensor<xpu, 2, dtype> iy, Tensor<xpu, 2, dtype> oy, Tensor<xpu, 2, dtype> fy,
      Tensor<xpu, 2, dtype> mcy, Tensor<xpu, 2, dtype> cy, Tensor<xpu, 2, dtype> my, Tensor<xpu, 2, dtype> y) {
    assert(_left2right);
    _lstm_input.ComputeForwardScore(py, x, pcy, iy);
    fy = 1 - iy;
    _lstm_cell.ComputeForwardScore(py, x, mcy);
    cy = mcy * iy + pcy * fy;
    _lstm_output.ComputeForwardScore(py, x, cy, oy);
    my = F<nl_tanh>(cy);
    y = my * oy;
  }

  //please allocate the memory outside here
  inline void ComputeBackwardLoss(Tensor<xpu, 3, dtype> x, Tensor<xpu, 3, dtype> iy, Tensor<xpu, 3, dtype> oy, Tensor<xpu, 3, dtype> fy,
      Tensor<xpu, 3, dtype> mcy, Tensor<xpu, 3, dtype> cy, Tensor<xpu, 3, dtype> my, 
      Tensor<xpu, 3, dtype> y, Tensor<xpu, 3, dtype> ly, Tensor<xpu, 3, dtype> lx, bool bclear = false) {
    int seq_size = x.size(0);
    if (seq_size == 0)
      return;

    if (bclear) lx = 0.0;
    	
    //left rnn
    Tensor<xpu, 3, dtype> liy = NewTensor<xpu>(Shape3(y.size(0), y.size(1), y.size(2)), d_zero);
    Tensor<xpu, 3, dtype> lfy = NewTensor<xpu>(Shape3(y.size(0), y.size(1), y.size(2)), d_zero);
    Tensor<xpu, 3, dtype> loy = NewTensor<xpu>(Shape3(y.size(0), y.size(1), y.size(2)), d_zero);
    Tensor<xpu, 3, dtype> lmcy = NewTensor<xpu>(Shape3(y.size(0), y.size(1), y.size(2)), d_zero);
    Tensor<xpu, 3, dtype> lcy = NewTensor<xpu>(Shape3(y.size(0), y.size(1), y.size(2)), d_zero);
    Tensor<xpu, 3, dtype> lmy = NewTensor<xpu>(Shape3(y.size(0), y.size(1), y.size(2)), d_zero);
    
    Tensor<xpu, 3, dtype> lFcy = NewTensor<xpu>(Shape3(y.size(0), y.size(1), y.size(2)), d_zero);
    Tensor<xpu, 3, dtype> lFy = NewTensor<xpu>(Shape3(y.size(0), y.size(1), y.size(2)), d_zero);

    if (_left2right) {
      //left rnn
      for (int idx = seq_size - 1; idx >= 0; idx--) {
        if (idx < seq_size - 1)
          ly[idx] = ly[idx] + lFy[idx];

        lmy[idx] = ly[idx] * oy[idx];
        loy[idx] = ly[idx] * my[idx];
        if (idx < seq_size - 1) {
          lcy[idx] = lmy[idx] * (1.0 - my[idx] * my[idx]) + lFcy[idx];
        } else {
          lcy[idx] = lmy[idx] * (1.0 - my[idx] * my[idx]);
        }

        if (idx == 0) {
          _lstm_output.ComputeBackwardLoss(_null1, x[idx], cy[idx], oy[idx],
              loy[idx], _null1Loss, lx[idx], lcy[idx]);

          lmcy[idx] = lcy[idx] * iy[idx];
          liy[idx] = lcy[idx] * mcy[idx];

          _lstm_cell.ComputeBackwardLoss(_null1, x[idx], mcy[idx], lmcy[idx], _null1Loss, lx[idx]);

          _lstm_input.ComputeBackwardLoss(_null1, x[idx], _null2, iy[idx],
              liy[idx], _null1Loss, lx[idx], _null2Loss);

        } else {
          _lstm_output.ComputeBackwardLoss(y[idx - 1], x[idx], cy[idx], oy[idx],
              loy[idx], lFy[idx - 1], lx[idx], lcy[idx]);

          lmcy[idx] = lcy[idx] * iy[idx];
          liy[idx] = lcy[idx] * mcy[idx];
          lFcy[idx - 1] = lcy[idx] * fy[idx];
          lfy[idx] = lcy[idx] * cy[idx - 1];

          _lstm_cell.ComputeBackwardLoss(y[idx - 1], x[idx], mcy[idx],
              lmcy[idx], lFy[idx - 1], lx[idx]);

          liy[idx] -= lfy[idx];

          _lstm_input.ComputeBackwardLoss(y[idx - 1], x[idx], cy[idx - 1],
              iy[idx], liy[idx], lFy[idx - 1], lx[idx], lFcy[idx - 1]);
        }
      }
    } else {
      // right rnn
      for (int idx = 0; idx < seq_size; idx++) {
        if (idx > 0)
          ly[idx] = ly[idx] + lFy[idx];

        lmy[idx] = ly[idx] * oy[idx];
        loy[idx] = ly[idx] * my[idx];
        if (idx > 0) {
          lcy[idx] = lmy[idx] * (1.0 - my[idx] * my[idx]) + lFcy[idx];
        } else {
          lcy[idx] = lmy[idx] * (1.0 - my[idx] * my[idx]);
        }

        if (idx == seq_size - 1) {
          _lstm_output.ComputeBackwardLoss(_null1, x[idx], cy[idx], oy[idx],
              loy[idx], _null1Loss, lx[idx], lcy[idx]);

          lmcy[idx] = lcy[idx] * iy[idx];
          liy[idx] = lcy[idx] * mcy[idx];

          _lstm_cell.ComputeBackwardLoss(_null1, x[idx], mcy[idx], lmcy[idx], _null1Loss, lx[idx]);

          _lstm_input.ComputeBackwardLoss(_null1, x[idx], _null2, iy[idx],
              liy[idx], _null1Loss, lx[idx], _null2Loss);
              
        } else {
          _lstm_output.ComputeBackwardLoss(y[idx + 1], x[idx], cy[idx], oy[idx],
              loy[idx], lFy[idx + 1], lx[idx], lcy[idx]);

          lmcy[idx] = lcy[idx] * iy[idx];
          liy[idx] = lcy[idx] * mcy[idx];
          lFcy[idx + 1] = lcy[idx] * fy[idx];
          lfy[idx] = lcy[idx] * cy[idx + 1];

          _lstm_cell.ComputeBackwardLoss(y[idx + 1], x[idx], mcy[idx],
              lmcy[idx], lFy[idx + 1], lx[idx]);

          liy[idx] -= lfy[idx];

          _lstm_input.ComputeBackwardLoss(y[idx + 1], x[idx], cy[idx + 1],
              iy[idx], liy[idx], lFy[idx + 1], lx[idx], lFcy[idx + 1]);
        }
      }
    }

    FreeSpace(&liy);
    FreeSpace(&lfy);
    FreeSpace(&loy);
    FreeSpace(&lmcy);
    FreeSpace(&lcy);
    FreeSpace(&lmy);
    FreeSpace(&lFcy);
    FreeSpace(&lFy);   
  }
  
  //please allocate the memory outside here
  inline void ComputeBackwardLoss(const vector<Tensor<xpu, 2, dtype> > &x, const vector<Tensor<xpu, 2, dtype> > &iy, 
      const vector<Tensor<xpu, 2, dtype> > &oy, const vector<Tensor<xpu, 2, dtype> > &fy, const vector<Tensor<xpu, 2, dtype> > &mcy, 
      const vector<Tensor<xpu, 2, dtype> > &cy, const vector<Tensor<xpu, 2, dtype> > &my, const vector<Tensor<xpu, 2, dtype> > &y, 
      vector<Tensor<xpu, 2, dtype> > &ly, vector<Tensor<xpu, 2, dtype> > &lx, bool bclear = false) {
    int seq_size = x.size();
    if (seq_size == 0)
      return;

    if (bclear)assign(lx, 0.0);

    vector<Tensor<xpu, 2, dtype> > liy(seq_size), lfy(seq_size), loy(seq_size), lcy(seq_size);
    vector<Tensor<xpu, 2, dtype> > lmcy(seq_size), lmy(seq_size), lFcy(seq_size), lFy(seq_size);
    for (int idx = 0; idx < seq_size; idx++) {
      liy[idx] = NewTensor<xpu>(Shape2(ly[0].size(0), ly[0].size(1)), d_zero);
      lfy[idx] = NewTensor<xpu>(Shape2(ly[0].size(0), ly[0].size(1)), d_zero);
      loy[idx] = NewTensor<xpu>(Shape2(ly[0].size(0), ly[0].size(1)), d_zero);
      lmcy[idx] = NewTensor<xpu>(Shape2(ly[0].size(0), ly[0].size(1)), d_zero);
      lcy[idx] = NewTensor<xpu>(Shape2(ly[0].size(0), ly[0].size(1)), d_zero);
      lmy[idx] = NewTensor<xpu>(Shape2(ly[0].size(0), ly[0].size(1)), d_zero);
      lFcy[idx] = NewTensor<xpu>(Shape2(ly[0].size(0), ly[0].size(1)), d_zero);
      lFy[idx] = NewTensor<xpu>(Shape2(ly[0].size(0), ly[0].size(1)), d_zero);
    }

    if (_left2right) {
      //left rnn
      for (int idx = seq_size - 1; idx >= 0; idx--) {
        if (idx < seq_size - 1)
          ly[idx] = ly[idx] + lFy[idx];

        lmy[idx] = ly[idx] * oy[idx];
        loy[idx] = ly[idx] * my[idx];
        if (idx < seq_size - 1) {
          lcy[idx] = lmy[idx] * (1.0 - my[idx] * my[idx]) + lFcy[idx];
        } else {
          lcy[idx] = lmy[idx] * (1.0 - my[idx] * my[idx]);
        }

        if (idx == 0) {
          _lstm_output.ComputeBackwardLoss(_null1, x[idx], cy[idx], oy[idx],
              loy[idx], _null1Loss, lx[idx], lcy[idx]);

          lmcy[idx] = lcy[idx] * iy[idx];
          liy[idx] = lcy[idx] * mcy[idx];

          _lstm_cell.ComputeBackwardLoss(_null1, x[idx], mcy[idx], lmcy[idx], _null1Loss, lx[idx]);

          _lstm_input.ComputeBackwardLoss(_null1, x[idx], _null2, iy[idx],
              liy[idx], _null1Loss, lx[idx], _null2Loss);

        } else {
          _lstm_output.ComputeBackwardLoss(y[idx - 1], x[idx], cy[idx], oy[idx],
              loy[idx], lFy[idx - 1], lx[idx], lcy[idx]);

          lmcy[idx] = lcy[idx] * iy[idx];
          liy[idx] = lcy[idx] * mcy[idx];
          lFcy[idx - 1] = lcy[idx] * fy[idx];
          lfy[idx] = lcy[idx] * cy[idx - 1];

          _lstm_cell.ComputeBackwardLoss(y[idx - 1], x[idx], mcy[idx],
              lmcy[idx], lFy[idx - 1], lx[idx]);

          liy[idx] -= lfy[idx];

          _lstm_input.ComputeBackwardLoss(y[idx - 1], x[idx], cy[idx - 1],
              iy[idx], liy[idx], lFy[idx - 1], lx[idx], lFcy[idx - 1]);
        }
      }
    } else {
      // right rnn
      for (int idx = 0; idx < seq_size; idx++) {
        if (idx > 0)
          ly[idx] = ly[idx] + lFy[idx];

        lmy[idx] = ly[idx] * oy[idx];
        loy[idx] = ly[idx] * my[idx];
        if (idx > 0) {
          lcy[idx] = lmy[idx] * (1.0 - my[idx] * my[idx]) + lFcy[idx];
        } else {
          lcy[idx] = lmy[idx] * (1.0 - my[idx] * my[idx]);
        }

        if (idx == seq_size - 1) {
          _lstm_output.ComputeBackwardLoss(_null1, x[idx], cy[idx], oy[idx],
              loy[idx], _null1Loss, lx[idx], lcy[idx]);

          lmcy[idx] = lcy[idx] * iy[idx];
          liy[idx] = lcy[idx] * mcy[idx];

          _lstm_cell.ComputeBackwardLoss(_null1, x[idx], mcy[idx], lmcy[idx], _null1Loss, lx[idx]);

          _lstm_input.ComputeBackwardLoss(_null1, x[idx], _null2, iy[idx],
              liy[idx], _null1Loss, lx[idx], _null2Loss);
              
        } else {
          _lstm_output.ComputeBackwardLoss(y[idx + 1], x[idx], cy[idx], oy[idx],
              loy[idx], lFy[idx + 1], lx[idx], lcy[idx]);

          lmcy[idx] = lcy[idx] * iy[idx];
          liy[idx] = lcy[idx] * mcy[idx];
          lFcy[idx + 1] = lcy[idx] * fy[idx];
          lfy[idx] = lcy[idx] * cy[idx + 1];

          _lstm_cell.ComputeBackwardLoss(y[idx + 1], x[idx], mcy[idx],
              lmcy[idx], lFy[idx + 1], lx[idx]);

          liy[idx] -= lfy[idx];

          _lstm_input.ComputeBackwardLoss(y[idx + 1], x[idx], cy[idx + 1],
              iy[idx], liy[idx], lFy[idx + 1], lx[idx], lFcy[idx + 1]);
        }
      }
    }
    
    for (int idx = 0; idx < seq_size; idx++) {
      FreeSpace(&(liy[idx]));
      FreeSpace(&(lfy[idx]));
      FreeSpace(&(loy[idx]));
      FreeSpace(&(lmcy[idx]));
      FreeSpace(&(lcy[idx]));
      FreeSpace(&(lmy[idx]));
      FreeSpace(&(lFcy[idx]));
      FreeSpace(&(lFy[idx]));      
    }
  }
  

  inline void randomprint(int num) {
    _lstm_output.randomprint(num);
    _lstm_input.randomprint(num);
    _lstm_cell.randomprint(num);
  }

  inline void updateAdaGrad(dtype regularizationWeight, dtype adaAlpha, dtype adaEps) {
    _lstm_output.updateAdaGrad(regularizationWeight, adaAlpha, adaEps);
    _lstm_input.updateAdaGrad(regularizationWeight, adaAlpha, adaEps);
    _lstm_cell.updateAdaGrad(regularizationWeight, adaAlpha, adaEps);
  }

  void writeModel(LStream &outf) {
    _lstm_output.writeModel(outf);
    _lstm_input.writeModel(outf);
    _lstm_cell.writeModel(outf);
    
    WriteBinary(outf, _left2right);

    SaveBinary(outf, _null1);
    SaveBinary(outf, _null1Loss);
    SaveBinary(outf, _null2);
    SaveBinary(outf, _null2Loss);
  }

  void loadModel(LStream &inf) {
    _lstm_output.loadModel(inf);
    _lstm_input.loadModel(inf);
    _lstm_cell.loadModel(inf);

    ReadBinary(inf, _left2right);

    LoadBinary(inf, &_null1, false);
    LoadBinary(inf, &_null1Loss, false);
    LoadBinary(inf, &_null2, false);
    LoadBinary(inf, &_null2Loss, false);
  }

};

#endif /* SRC_LSTM_CHD_H_ */
