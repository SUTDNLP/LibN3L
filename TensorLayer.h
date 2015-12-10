/*
 * TensorLayer.h
 *
 *  Created on: Mar 18, 2015
 *      Author: mszhang
 */

#ifndef SRC_TensorLayer_H_
#define SRC_TensorLayer_H_
#include "tensor.h"
#include "MyLib.h"
#include "Utiltensor.h"

using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;

template<typename xpu>
class TensorLayer {

public:

  Tensor<xpu, 3, dtype> _W;
  Tensor<xpu, 2, dtype> _V;
  Tensor<xpu, 2, dtype> _b;

  Tensor<xpu, 3, dtype> _gradW;
  Tensor<xpu, 2, dtype> _gradV;
  Tensor<xpu, 2, dtype> _gradb;

  Tensor<xpu, 3, dtype> _eg2W;
  Tensor<xpu, 2, dtype> _eg2V;
  Tensor<xpu, 2, dtype> _eg2b;

  int _mode; // 1: x1 W x2; 2: x1 W x2 + V x2; 3: x1 W x2 + V x2 + b

  int _funcType; // 0: tanh, 1: sigmod, 2: f(x)=x, 3: exp

public:
  TensorLayer() {
  }

  inline void initial(int nOSize, int nISize, int mode = 1, int seed = 0, int funcType = 0) {
    dtype bound = sqrt(6.0 / (nOSize + nISize + 1));
    //dtype bound = 0.01;

    _W = NewTensor<xpu>(Shape3(nOSize, nISize, nOSize), d_zero);
    _gradW = NewTensor<xpu>(Shape3(nOSize, nISize, nOSize), d_zero);
    _eg2W = NewTensor<xpu>(Shape3(nOSize, nISize, nOSize), d_zero);

    _V = NewTensor<xpu>(Shape2(nOSize, nOSize), d_zero);
    _gradV = NewTensor<xpu>(Shape2(nOSize, nOSize), d_zero);
    _eg2V = NewTensor<xpu>(Shape2(nOSize, nOSize), d_zero);

    _b = NewTensor<xpu>(Shape2(1, nOSize), d_zero);
    _gradb = NewTensor<xpu>(Shape2(1, nOSize), d_zero);
    _eg2b = NewTensor<xpu>(Shape2(1, nOSize), d_zero);

    random(_W, -1.0 * bound, 1.0 * bound, seed);
    random(_V, -1.0 * bound, 1.0 * bound, seed + 1);
    random(_b, -1.0 * bound, 1.0 * bound, seed + 2);

    _mode = mode;
    _funcType = funcType;
  }

  inline void initial(Tensor<xpu, 3, dtype> W, Tensor<xpu, 2, dtype> V, Tensor<xpu, 2, dtype> b, int mode = 1, int funcType = 0) {
    static int nOSize, nISize;
    nOSize = W.size(0);
    nISize = W.size(1);

    _W = NewTensor<xpu>(Shape3(nOSize, nISize, nOSize), d_zero);
    _gradW = NewTensor<xpu>(Shape3(nOSize, nISize, nOSize), d_zero);
    _eg2W = NewTensor<xpu>(Shape3(nOSize, nISize, nOSize), d_zero);
    Copy(_W, W);

    _V = NewTensor<xpu>(Shape2(nOSize, nOSize), d_zero);
    _gradV = NewTensor<xpu>(Shape2(nOSize, nOSize), d_zero);
    _eg2V = NewTensor<xpu>(Shape2(nOSize, nOSize), d_zero);
    if (mode >= 2)
      Copy(_V, V);

    _b = NewTensor<xpu>(Shape2(1, nOSize), d_zero);
    _gradb = NewTensor<xpu>(Shape2(1, nOSize), d_zero);
    _eg2b = NewTensor<xpu>(Shape2(1, nOSize), d_zero);

    if (mode >= 3)
      Copy(_b, b);

    _mode = mode;
    _funcType = funcType;
  }

  inline void release() {
    FreeSpace(&_W);
    FreeSpace(&_gradW);
    FreeSpace(&_eg2W);
    FreeSpace(&_V);
    FreeSpace(&_gradV);
    FreeSpace(&_eg2V);
    FreeSpace(&_b);
    FreeSpace(&_gradb);
    FreeSpace(&_eg2b);
  }

  virtual ~TensorLayer() {
    // TODO Auto-generated destructor stub
  }

  inline dtype squarenormAll() {
    dtype result = squarenorm(_gradW);

    if (_mode >= 2) {
      result += squarenorm(_gradV);
    }

    if (_mode >= 3) {
      result += squarenorm(_gradb);
    }

    return result;
  }

  inline void scaleGrad(dtype scale) {
    _gradW = _gradW * scale;
    if (_mode >= 2) {
      _gradV = _gradV * scale;
    }
    if (_mode >= 3) {
      _gradb = _gradb * scale;
    }
  }

public:
  inline void ComputeForwardScore(Tensor<xpu, 2, dtype> x1, Tensor<xpu, 2, dtype> x2, Tensor<xpu, 2, dtype> y) {
    Tensor<xpu, 2, dtype> midresult1 = NewTensor<xpu>(Shape2(1, y.size(1)), d_zero);
    Tensor<xpu, 2, dtype> midresult2 = NewTensor<xpu>(Shape2(1, 1), d_zero);
    for (int idy = 0; idy < y.size(1); idy++) {
      midresult1 = dot(x1, _W[idy]);
      midresult2 = dot(midresult1, x2.T());
      y[0][idy] = midresult2[0][0];
    }

    if (_mode >= 2) {
      midresult1 = dot(x2, _V.T());
      y += midresult1;
    }

    if (_mode >= 3)
      y = y + _b;
    if (_funcType == 0)
      y = F<nl_tanh>(y);
    else if (_funcType == 1)
      y = F<nl_sigmoid>(y);
    else if (_funcType == 3)
      y = F<nl_exp>(y);

    FreeSpace(&midresult1);
    FreeSpace(&midresult2);
  }


  inline void ComputeForwardScore(Tensor<xpu, 3, dtype> x1, Tensor<xpu, 3, dtype> x2, Tensor<xpu, 3, dtype> y) {
    int seq_size = y.size(0);
    Tensor<xpu, 2, dtype> midresult1 = NewTensor<xpu>(Shape2(1, y.size(2)), d_zero);
    Tensor<xpu, 2, dtype> midresult2 = NewTensor<xpu>(Shape2(1, 1), d_zero);
    for(int id = 0; id < seq_size; id++){
      for (int idy = 0; idy < y.size(2); idy++) {
        midresult1 = dot(x1[id], _W[idy]);
        midresult2 = dot(midresult1, x2[id].T());
        y[id][0][idy] = midresult2[0][0];
      }

      if (_mode >= 2) {
        midresult1 = dot(x2[id], _V.T());
        y[id] += midresult1;
      }

      if (_mode >= 3)
        y[id] = y[id] + _b;
      if (_funcType == 0)
        y[id] = F<nl_tanh>(y[id]);
      else if (_funcType == 1)
        y[id] = F<nl_sigmoid>(y[id]);
      else if (_funcType == 3)
        y[id] = F<nl_exp>(y[id]);
    }

    FreeSpace(&midresult1);
    FreeSpace(&midresult2);
  }

  inline void ComputeForwardScore(const std::vector<Tensor<xpu, 2, dtype> > &x1, const std::vector<Tensor<xpu, 2, dtype> > &x2,
      std::vector<Tensor<xpu, 2, dtype> > &y) {
    int seq_size = y.size();
    assert(seq_size > 0);
    Tensor<xpu, 2, dtype> midresult1 = NewTensor<xpu>(Shape2(1, y[0].size(1)), d_zero);
    Tensor<xpu, 2, dtype> midresult2 = NewTensor<xpu>(Shape2(1, 1), d_zero);
    for(int id = 0; id < seq_size; id++){
      for (int idy = 0; idy < y.size(2); idy++) {
        midresult1 = dot(x1[id], _W[idy]);
        midresult2 = dot(midresult1, x2[id].T());
        y[id][0][idy] = midresult2[0][0];
      }

      if (_mode >= 2) {
        midresult1 = dot(x2[id], _V.T());
        y[id] += midresult1;
      }

      if (_mode >= 3)
        y[id] = y[id] + _b;
      if (_funcType == 0)
        y[id] = F<nl_tanh>(y[id]);
      else if (_funcType == 1)
        y[id] = F<nl_sigmoid>(y[id]);
      else if (_funcType == 3)
        y[id] = F<nl_exp>(y[id]);
    }

    FreeSpace(&midresult1);
    FreeSpace(&midresult2);
  }

  //please allocate the memory outside here
  inline void ComputeBackwardLoss(Tensor<xpu, 2, dtype> x1, Tensor<xpu, 2, dtype> x2, Tensor<xpu, 2, dtype> y, Tensor<xpu, 2, dtype> ly,
      Tensor<xpu, 2, dtype> lx1, Tensor<xpu, 2, dtype> lx2, bool bclear = false) {
    //_gradW
    Tensor<xpu, 2, dtype> deri_yx(Shape2(y.size(0), y.size(1))), cly(Shape2(y.size(0), y.size(1)));
    AllocSpace(&deri_yx);
    AllocSpace(&cly);

    if(bclear) {
      lx1 = 0.0;
      lx2 = 0.0;
    }
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

    Tensor<xpu, 2, dtype> midresult1 = NewTensor<xpu>(Shape2(1, y.size(1)), d_zero);
    Tensor<xpu, 2, dtype> midresult2 = NewTensor<xpu>(Shape2(1, y.size(1)), d_zero);
    //_gradW
    for (int idy = 0; idy < y.size(1); idy++) {
      midresult1 = dot(x1, _W[idy]);
      lx2 += cly[0][idy] * midresult1;
      midresult2 = cly[0][idy] * x2;
      _gradW[idy] += dot(x1.T(), midresult2);
      lx1 += dot(midresult2, _W[idy].T());
    }

    //_gradV
    if (_mode >= 2) {
      _gradV += dot(cly.T(), x2);
      //lx
      lx2 += dot(cly, _V);
    }

    //_gradb
    if (_mode >= 3)
      _gradb += cly;

    FreeSpace(&deri_yx);
    FreeSpace(&cly);
    FreeSpace(&midresult1);
    FreeSpace(&midresult2);
  }

  inline void ComputeBackwardLoss(Tensor<xpu, 3, dtype> x1, Tensor<xpu, 3, dtype> x2, Tensor<xpu, 3, dtype> y, Tensor<xpu, 3, dtype> ly,
      Tensor<xpu, 3, dtype> lx1, Tensor<xpu, 3, dtype> lx2, bool bclear = false) {
    int seq_size = y.size(0);
    int y_dim1 = y.size(1), y_dim2 = y.size(2);
    assert(y_dim1 == 1);
    //_gradW
    Tensor<xpu, 2, dtype> deri_yx(Shape2(y_dim1, y_dim2)), cly(Shape2(y_dim1, y_dim2));
    Tensor<xpu, 2, dtype> midresult1 = NewTensor<xpu>(Shape2(y_dim1, y_dim2), d_zero);
    Tensor<xpu, 2, dtype> midresult2 = NewTensor<xpu>(Shape2(y_dim1, y_dim2), d_zero);
    AllocSpace(&deri_yx);
    AllocSpace(&cly);

    if(bclear) {
      lx1 = 0.0;
      lx2 = 0.0;
    }
    for (int id = 0; id < seq_size; id++) {
      if (_funcType == 0) {
        deri_yx = F<nl_dtanh>(y[id]);
        cly = ly[id] * deri_yx;
      } else if (_funcType == 1) {
        deri_yx = F<nl_dsigmoid>(y);
        cly = ly[id] * deri_yx;
      } else if (_funcType == 3) {
        cly = ly[id] * y[id];
      } else {
        //cly = ly;
        Copy(cly, ly[id]);
      }

      //_gradW
      for (int idy = 0; idy < y.size(2); idy++) {
        midresult1 = dot(x1[id], _W[idy]);
        lx2[id] += cly[0][idy] * midresult1;
        midresult2 = cly[0][idy] * x2[id];
        _gradW[idy] += dot(x1[id].T(), midresult2);
        lx1[id] += dot(midresult2, _W[idy].T());
      }

      //_gradV
      if (_mode >= 2) {
        _gradV += dot(cly.T(), x2[id]);
        //lx
        lx2[id] += dot(cly, _V);
      }

      //_gradb
      if (_mode >= 3)
        _gradb += cly;
    }

    FreeSpace(&deri_yx);
    FreeSpace(&cly);
    FreeSpace(&midresult1);
    FreeSpace(&midresult2);
  }

  inline void ComputeBackwardLoss(const std::vector<Tensor<xpu, 2, dtype> > &x1, const std::vector<Tensor<xpu, 2, dtype> > &x2,
      const std::vector<Tensor<xpu, 2, dtype> > &y, const std::vector<Tensor<xpu, 2, dtype> > &ly,
      std::vector<Tensor<xpu, 2, dtype> > &lx1, std::vector<Tensor<xpu, 2, dtype> > &lx2, bool bclear = false) {
    int seq_size = y.size();
    assert(seq_size > 0);
    int y_dim1 = y[0].size(0), y_dim2 = y[0].size(1);
    assert(y_dim1 == 1);
    //_gradW
    Tensor<xpu, 2, dtype> deri_yx(Shape2(y_dim1, y_dim2)), cly(Shape2(y_dim1, y_dim2));
    Tensor<xpu, 2, dtype> midresult1 = NewTensor<xpu>(Shape2(y_dim1, y_dim2), d_zero);
    Tensor<xpu, 2, dtype> midresult2 = NewTensor<xpu>(Shape2(y_dim1, y_dim2), d_zero);
    AllocSpace(&deri_yx);
    AllocSpace(&cly);

    if(bclear) {
      for (int id = 0; id < seq_size; id++) {
        lx1[id] = 0.0;
        lx2[id] = 0.0;
      }
    }
    for (int id = 0; id < seq_size; id++) {
      if (_funcType == 0) {
        deri_yx = F<nl_dtanh>(y[id]);
        cly = ly[id] * deri_yx;
      } else if (_funcType == 1) {
        deri_yx = F<nl_dsigmoid>(y);
        cly = ly[id] * deri_yx;
      } else if (_funcType == 3) {
        cly = ly[id] * y[id];
      } else {
        //cly = ly;
        Copy(cly, ly[id]);
      }

      //_gradW
      for (int idy = 0; idy < y.size(2); idy++) {
        midresult1 = dot(x1[id], _W[idy]);
        lx2[id] += cly[0][idy] * midresult1;
        midresult2 = cly[0][idy] * x2[id];
        _gradW[idy] += dot(x1[id].T(), midresult2);
        lx1[id] += dot(midresult2, _W[idy].T());
      }

      //_gradV
      if (_mode >= 2) {
        _gradV += dot(cly.T(), x2[id]);
        //lx
        lx2[id] += dot(cly, _V);
      }

      //_gradb
      if (_mode >= 3)
        _gradb += cly;
    }

    FreeSpace(&deri_yx);
    FreeSpace(&cly);
    FreeSpace(&midresult1);
    FreeSpace(&midresult2);
  }

  inline void randomprint(int num) {
    static int nOSize, nISize;
    nOSize = _W.size(0);
    nISize = _W.size(1);
    int count = 0;
    while (count < num) {
      int idx = rand() % nOSize;
      int idy = rand() % nISize;
      int idz = rand() % nOSize;
      std::cout << "_W[" << idx << "," << idy << "," << idz << "]=" << _W[idx][idy][idz] << " ";

      if (_mode >= 2) {
        int idy = rand() % nOSize;
        int idz = rand() % nOSize;
        std::cout << "_V[" << idy << "," << idz << "]=" << _V[idy][idz] << " ";
      }

      if (_mode >= 3) {
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

    if (_mode >= 2) {
      _gradV = _gradV + _V * regularizationWeight;
      _eg2V = _eg2V + _gradV * _gradV;
      _V = _V - _gradV * adaAlpha / F<nl_sqrt>(_eg2V + adaEps);
    }

    if (_mode >= 3) {
      _gradb = _gradb + _b * regularizationWeight;
      _eg2b = _eg2b + _gradb * _gradb;
      _b = _b - _gradb * adaAlpha / F<nl_sqrt>(_eg2b + adaEps);
    }

    clearGrad();
  }

  inline void clearGrad() {
    _gradW = 0;
    if (_mode >= 2)
      _gradV = 0;
    if (_mode >= 3)
      _gradb = 0;
  }

  void writeModel(LStream &outf) {
    SaveBinary(outf, _W);
    SaveBinary(outf, _V);
    SaveBinary(outf, _b);

    SaveBinary(outf, _gradW);
    SaveBinary(outf, _gradV);
    SaveBinary(outf, _gradb);

    SaveBinary(outf, _eg2W);
    SaveBinary(outf, _eg2V);
    SaveBinary(outf, _eg2b);

    WriteBinary(outf, _mode);
    WriteBinary(outf, _funcType);
  }

  void loadModel(LStream &inf) {
    LoadBinary(inf, &_W, false);
    LoadBinary(inf, &_V, false);
    LoadBinary(inf, &_b, false);

    LoadBinary(inf, &_gradW, false);
    LoadBinary(inf, &_gradV, false);
    LoadBinary(inf, &_gradb, false);

    LoadBinary(inf, &_eg2W, false);
    LoadBinary(inf, &_eg2V, false);
    LoadBinary(inf, &_eg2b, false);

    ReadBinary(inf, _mode);
    ReadBinary(inf, _funcType);
  }

};

#endif /* SRC_TensorLayer_H_ */
