/*
 * UniLayer.h
 *
 *  Created on: Mar 18, 2015
 *      Author: mszhang
 */

#ifndef SRC_UniLayer_H_
#define SRC_UniLayer_H_
#include "tensor.h"
#include "MyLib.h"
#include "Utiltensor.h"

using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;

template<typename xpu>
class UniLayer {

public:

  Tensor<xpu, 2, dtype> _W;
  Tensor<xpu, 2, dtype> _b;

  Tensor<xpu, 2, dtype> _gradW;
  Tensor<xpu, 2, dtype> _gradb;

  Tensor<xpu, 2, dtype> _eg2W;
  Tensor<xpu, 2, dtype> _eg2b;

  bool _bUseB;

  int _funcType; // 0: tanh, 1: sigmod, 2: f(x)=x, 3: exp

public:
  UniLayer() {
  }

  inline void initial(int nOSize, int nISize, bool bUseB = true, int seed = 0, int funcType = 0) {
    dtype bound = sqrt(6.0 / (nOSize + nISize + 1));
    //dtype bound = 0.01;

    _W = NewTensor<xpu>(Shape2(nOSize, nISize), d_zero);
    _gradW = NewTensor<xpu>(Shape2(nOSize, nISize), d_zero);
    _eg2W = NewTensor<xpu>(Shape2(nOSize, nISize), d_zero);

    _b = NewTensor<xpu>(Shape2(1, nOSize), d_zero);
    _gradb = NewTensor<xpu>(Shape2(1, nOSize), d_zero);
    _eg2b = NewTensor<xpu>(Shape2(1, nOSize), d_zero);

    random(_W, -1.0 * bound, 1.0 * bound, seed);
    random(_b, -1.0 * bound, 1.0 * bound, seed + 1);

    _bUseB = bUseB;
    _funcType = funcType;
  }

  inline void initial(Tensor<xpu, 2, dtype> W, Tensor<xpu, 2, dtype> b, bool bUseB = true, int funcType = 0) {
    static int nOSize, nISize;
    nOSize = W.size(0);
    nISize = W.size(1);

    _W = NewTensor<xpu>(Shape2(nOSize, nISize), d_zero);
    _gradW = NewTensor<xpu>(Shape2(nOSize, nISize), d_zero);
    _eg2W = NewTensor<xpu>(Shape2(nOSize, nISize), d_zero);
    Copy(_W, W);

    _b = NewTensor<xpu>(Shape2(1, nOSize), d_zero);
    _gradb = NewTensor<xpu>(Shape2(1, nOSize), d_zero);
    _eg2b = NewTensor<xpu>(Shape2(1, nOSize), d_zero);

    if (bUseB)
      Copy(_b, b);

    _bUseB = bUseB;
    _funcType = funcType;
  }

  inline void initial(Tensor<xpu, 2, dtype> W,  int funcType = 0) {
    static int nOSize, nISize;
    nOSize = W.size(0);
    nISize = W.size(1);

    _W = NewTensor<xpu>(Shape2(nOSize, nISize), d_zero);
    _gradW = NewTensor<xpu>(Shape2(nOSize, nISize), d_zero);
    _eg2W = NewTensor<xpu>(Shape2(nOSize, nISize), d_zero);
    Copy(_W, W);

    _b = NewTensor<xpu>(Shape2(1, nOSize), d_zero);
    _gradb = NewTensor<xpu>(Shape2(1, nOSize), d_zero);
    _eg2b = NewTensor<xpu>(Shape2(1, nOSize), d_zero);


    _bUseB = false;
    _funcType = funcType;
  }
  inline void release() {
    FreeSpace(&_W);
    FreeSpace(&_gradW);
    FreeSpace(&_eg2W);
    FreeSpace(&_b);
    FreeSpace(&_gradb);
    FreeSpace(&_eg2b);
  }

  virtual ~UniLayer() {
    // TODO Auto-generated destructor stub
  }

  inline dtype squarenormAll() {
    dtype result = squarenorm(_gradW);

    if (_bUseB) {
      result += squarenorm(_gradb);
    }

    return result;
  }

  inline void scaleGrad(dtype scale) {
    _gradW = _gradW * scale;
    if (_bUseB) {
      _gradb = _gradb * scale;
    }
  }

public:
  inline void ComputeForwardScore(Tensor<xpu, 2, dtype> x, Tensor<xpu, 2, dtype> y) {
    y = dot(x, _W.T());
    if (_bUseB)
      y = y + _b;
    if (_funcType == 0)
      y = F<nl_tanh>(y);
    else if (_funcType == 1)
      y = F<nl_sigmoid>(y);
    else if (_funcType == 3)
      y = F<nl_exp>(y);
  }

  inline void ComputeForwardScore(Tensor<xpu, 3, dtype> x, Tensor<xpu, 3, dtype> y) {
    int seq_size = y.size(0);
    for (int id = 0; id < seq_size; id++) {
      y[id] = dot(x[id], _W.T());
      if (_bUseB)
        y[id] = y[id] + _b;
      if (_funcType == 0)
        y[id] = F<nl_tanh>(y[id]);
      else if (_funcType == 1)
        y[id] = F<nl_sigmoid>(y[id]);
      else if (_funcType == 3)
        y[id] = F<nl_exp>(y[id]);
    }
  }

  inline void ComputeForwardScore(const std::vector<Tensor<xpu, 2, dtype> > &x, std::vector<Tensor<xpu, 2, dtype> > &y) {
    int seq_size = y.size();
    for (int id = 0; id < seq_size; id++) {
      y[id] = dot(x[id], _W.T());
      if (_bUseB)
        y[id] = y[id] + _b;
      if (_funcType == 0)
        y[id] = F<nl_tanh>(y[id]);
      else if (_funcType == 1)
        y[id] = F<nl_sigmoid>(y[id]);
      else if (_funcType == 3)
        y[id] = F<nl_exp>(y[id]);
    }
  }

  //please allocate the memory outside here
  inline void ComputeBackwardLoss(Tensor<xpu, 2, dtype> x, Tensor<xpu, 2, dtype> y, Tensor<xpu, 2, dtype> ly, Tensor<xpu, 2, dtype> lx, bool bclear = false) {
    //_gradW
    Tensor<xpu, 2, dtype> deri_yx(Shape2(y.size(0), y.size(1))), cly(Shape2(y.size(0), y.size(1)));
    AllocSpace(&deri_yx);
    AllocSpace(&cly);

    if (bclear)
      lx = 0.0;
    if (_funcType == 0) {
      deri_yx = F<nl_dtanh>(y);
      cly = ly * deri_yx;
    } else if (_funcType == 1) {
      deri_yx = F<nl_dsigmoid>(y);
      cly = ly * deri_yx;
    } else if (_funcType == 3) {
      cly = ly * y;
    } else {
      //cly = ly;
      Copy(cly, ly);
    }
    //_gradW
    _gradW += dot(cly.T(), x);

    //_gradb
    if (_bUseB)
      _gradb += cly;

    //lx
    lx += dot(cly, _W);

    FreeSpace(&deri_yx);
    FreeSpace(&cly);
  }

  //please allocate the memory outside here
  inline void ComputeBackwardLoss(Tensor<xpu, 3, dtype> x, Tensor<xpu, 3, dtype> y, Tensor<xpu, 3, dtype> ly, Tensor<xpu, 3, dtype> lx, bool bclear = false) {
    //_gradW
    int seq_size = y.size(0);
    int y_dim1 = y.size(1), y_dim2 = y.size(2);
    Tensor<xpu, 2, dtype> deri_yx(Shape2(y_dim1, y_dim2)), cly(Shape2(y_dim1, y_dim2));
    AllocSpace(&deri_yx);
    AllocSpace(&cly);

    if (bclear)
      lx = 0.0;
    for (int id = 0; id < seq_size; id++) {
      if (_funcType == 0) {
        deri_yx = F<nl_dtanh>(y[id]);
        cly = ly[id] * deri_yx;
      } else if (_funcType == 1) {
        deri_yx = F<nl_dsigmoid>(y[id]);
        cly = ly[id] * deri_yx;
      } else if (_funcType == 3) {
        cly = ly[id] * y[id];
      } else {
        //cly = ly;
        Copy(cly, ly[id]);
      }
      //_gradW
      _gradW += dot(cly.T(), x[id]);

      //_gradb
      if (_bUseB)
        _gradb += cly;

      //lx
      lx[id] += dot(cly, _W);
    }

    FreeSpace(&deri_yx);
    FreeSpace(&cly);
  }

  //please allocate the memory outside here
  inline void ComputeBackwardLoss(const std::vector<Tensor<xpu, 2, dtype> > &x, const std::vector<Tensor<xpu, 2, dtype> > &y,
      const std::vector<Tensor<xpu, 2, dtype> > &ly, std::vector<Tensor<xpu, 2, dtype> > &lx, bool bclear = false) {
    //_gradW
    int seq_size = y.size();
    assert(seq_size > 0);
    int y_dim1 = y[0].size(0), y_dim2 = y[0].size(1);
    Tensor<xpu, 2, dtype> deri_yx(Shape2(y_dim1, y_dim2)), cly(Shape2(y_dim1, y_dim2));
    AllocSpace(&deri_yx);
    AllocSpace(&cly);

    if(bclear) {
      for (int id = 0; id < seq_size; id++) {
        lx[id] = 0.0;
      }
    }
    for (int id = 0; id < seq_size; id++) {
      if (_funcType == 0) {
        deri_yx = F<nl_dtanh>(y[id]);
        cly = ly[id] * deri_yx;
      } else if (_funcType == 1) {
        deri_yx = F<nl_dsigmoid>(y[id]);
        cly = ly[id] * deri_yx;
      } else if (_funcType == 3) {
        cly = ly[id] * y[id];
      } else {
        //cly = ly;
        Copy(cly, ly[id]);
      }
      //_gradW
      _gradW += dot(cly.T(), x[id]);

      //_gradb
      if (_bUseB)
        _gradb += cly;

      //lx
      lx[id] += dot(cly, _W);
    }

    FreeSpace(&deri_yx);
    FreeSpace(&cly);
  }

  inline void randomprint(int num) {
    static int nOSize, nISize;
    nOSize = _W.size(0);
    nISize = _W.size(1);
    int count = 0;
    while (count < num) {
      int idx = rand() % nOSize;
      int idy = rand() % nISize;

      std::cout << "_W[" << idx << "," << idy << "]=" << _W[idx][idy] << " ";

      if (_bUseB) {
        int idz = rand() % nOSize;
        std::cout << "_b[0][" << idz << "]=" << _b[0][idz] << " ";
      }
      count++;
    }

    std::cout << std::endl;
  }

  inline void updateAdaGrad(dtype regularizationWeight, dtype adaAlpha, dtype adaEps) {
    _gradW = _gradW + _W * regularizationWeight;
    _eg2W = _eg2W + _gradW * _gradW;
    _W = _W - _gradW * adaAlpha / F<nl_sqrt>(_eg2W + adaEps);

    if (_bUseB) {
      _gradb = _gradb + _b * regularizationWeight;
      _eg2b = _eg2b + _gradb * _gradb;
      _b = _b - _gradb * adaAlpha / F<nl_sqrt>(_eg2b + adaEps);
    }

    clearGrad();
  }

  inline void clearGrad() {
    _gradW = 0;
    if (_bUseB)
      _gradb = 0;
  }

  void writeModel(LStream &outf) {
    SaveBinary(outf, _W);
    SaveBinary(outf, _b);
    SaveBinary(outf, _gradW);
    SaveBinary(outf, _gradb);
    SaveBinary(outf, _eg2W);
    SaveBinary(outf, _eg2b);
    WriteBinary(outf, _bUseB);
    WriteBinary(outf, _funcType);
    // cout << "Unilayer " << _bUseB << _funcType << endl;

  }

  void loadModel(LStream &inf) {
    LoadBinary(inf, &_W, false);
    LoadBinary(inf, &_b, false);
    LoadBinary(inf, &_gradW, false);
    LoadBinary(inf, &_gradb, false);
    LoadBinary(inf, &_eg2W, false);
    LoadBinary(inf, &_eg2b, false);
    ReadBinary(inf, _bUseB);
    ReadBinary(inf, _funcType);
    // cout << "Unilayer " << _bUseB << _funcType << endl;
  }
  
};

#endif /* SRC_UniLayer_H_ */
