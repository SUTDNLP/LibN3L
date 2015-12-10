/*
 * TriLayer.h
 *
 *  Created on: Mar 18, 2015
 *      Author: mszhang
 */

#ifndef SRC_TriLayer_H_
#define SRC_TriLayer_H_
#include "tensor.h"
#include "MyLib.h"
#include "Utiltensor.h"

using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;

template<typename xpu>
class TriLayer {

public:

  Tensor<xpu, 2, dtype> _W1;
  Tensor<xpu, 2, dtype> _W2;
  Tensor<xpu, 2, dtype> _W3;
  Tensor<xpu, 2, dtype> _b;

  Tensor<xpu, 2, dtype> _gradW1;
  Tensor<xpu, 2, dtype> _gradW2;
  Tensor<xpu, 2, dtype> _gradW3;
  Tensor<xpu, 2, dtype> _gradb;

  Tensor<xpu, 2, dtype> _eg2W1;
  Tensor<xpu, 2, dtype> _eg2W2;
  Tensor<xpu, 2, dtype> _eg2W3;
  Tensor<xpu, 2, dtype> _eg2b;

  bool _bUseB;

  int _funcType; // 0: tanh, 1: sigmod, 2: f(x)=x, 3: exp

public:
  TriLayer() {
  }

  inline void initial(int nOSize, int nISize1, int nISize2, int nISize3, bool bUseB = true, int seed = 0, int funcType = 0) {
    dtype bound = sqrt(6.0 / (nOSize + nISize1 + nISize2 + nISize3 + 1));
    //dtype bound = 0.01;

    _W1 = NewTensor<xpu>(Shape2(nOSize, nISize1), d_zero);
    _gradW1 = NewTensor<xpu>(Shape2(nOSize, nISize1), d_zero);
    _eg2W1 = NewTensor<xpu>(Shape2(nOSize, nISize1), d_zero);

    _W2 = NewTensor<xpu>(Shape2(nOSize, nISize2), d_zero);
    _gradW2 = NewTensor<xpu>(Shape2(nOSize, nISize2), d_zero);
    _eg2W2 = NewTensor<xpu>(Shape2(nOSize, nISize2), d_zero);

    _W3 = NewTensor<xpu>(Shape2(nOSize, nISize3), d_zero);
    _gradW3 = NewTensor<xpu>(Shape2(nOSize, nISize3), d_zero);
    _eg2W3 = NewTensor<xpu>(Shape2(nOSize, nISize3), d_zero);

    _b = NewTensor<xpu>(Shape2(1, nOSize), d_zero);
    _gradb = NewTensor<xpu>(Shape2(1, nOSize), d_zero);
    _eg2b = NewTensor<xpu>(Shape2(1, nOSize), d_zero);

    random(_W1, -1.0 * bound, 1.0 * bound, seed);
    random(_W2, -1.0 * bound, 1.0 * bound, seed+1);
    random(_W3, -1.0 * bound, 1.0 * bound, seed+2);
    random(_b, -1.0 * bound, 1.0 * bound, seed+3);

    _bUseB = bUseB;
    _funcType = funcType;
  }

  inline void initial(Tensor<xpu, 2, dtype> W1, Tensor<xpu, 2, dtype> W2, Tensor<xpu, 2, dtype> W3, Tensor<xpu, 2, dtype> b, bool bUseB = true,
      int funcType = 0) {
    static int nOSize, nISize1, nISize2, nISize3;
    nOSize = W1.size(0);
    nISize1 = W1.size(1);
    nISize2 = W2.size(1);
    nISize3 = W3.size(1);

    _W1 = NewTensor<xpu>(Shape2(nOSize, nISize1), d_zero);
    _gradW1 = NewTensor<xpu>(Shape2(nOSize, nISize1), d_zero);
    _eg2W1 = NewTensor<xpu>(Shape2(nOSize, nISize1), d_zero);
    Copy(_W1, W1);

    _W2 = NewTensor<xpu>(Shape2(nOSize, nISize2), d_zero);
    _gradW2 = NewTensor<xpu>(Shape2(nOSize, nISize2), d_zero);
    _eg2W2 = NewTensor<xpu>(Shape2(nOSize, nISize2), d_zero);
    Copy(_W2, W2);

    _W3 = NewTensor<xpu>(Shape2(nOSize, nISize3), d_zero);
    _gradW3 = NewTensor<xpu>(Shape2(nOSize, nISize3), d_zero);
    _eg2W3 = NewTensor<xpu>(Shape2(nOSize, nISize3), d_zero);
    Copy(_W3, W3);

    _b = NewTensor<xpu>(Shape2(1, nOSize), d_zero);
    _gradb = NewTensor<xpu>(Shape2(1, nOSize), d_zero);
    _eg2b = NewTensor<xpu>(Shape2(1, nOSize), d_zero);

    if (bUseB)
      Copy(_b, b);

    _bUseB = bUseB;
    _funcType = funcType;
  }

  inline void release() {
    FreeSpace(&_W1);
    FreeSpace(&_gradW1);
    FreeSpace(&_eg2W1);
    FreeSpace(&_W2);
    FreeSpace(&_gradW2);
    FreeSpace(&_eg2W2);
    FreeSpace(&_W3);
    FreeSpace(&_gradW3);
    FreeSpace(&_eg2W3);
    FreeSpace(&_b);
    FreeSpace(&_gradb);
    FreeSpace(&_eg2b);
  }

  virtual ~TriLayer() {
    // TODO Auto-generated destructor stub
  }

  inline dtype squarenormAll() {
    dtype result = squarenorm(_gradW1);
    result += squarenorm(_gradW2);
    result += squarenorm(_gradW3);
    if (_bUseB) {
      result += squarenorm(_gradb);
    }

    return result;
  }

  inline void scaleGrad(dtype scale) {
    _gradW1 = _gradW1 * scale;
    _gradW2 = _gradW2 * scale;
    _gradW3 = _gradW3 * scale;
    if (_bUseB) {
      _gradb = _gradb * scale;
    }
  }

public:
  inline void ComputeForwardScore(Tensor<xpu, 2, dtype> x1, Tensor<xpu, 2, dtype> x2, Tensor<xpu, 2, dtype> x3, Tensor<xpu, 2, dtype> y) {
    y = dot(x1, _W1.T());
    y += dot(x2, _W2.T());
    y += dot(x3, _W3.T());
    if (_bUseB)
      y = y + _b;
    if (_funcType == 0)
      y = F<nl_tanh>(y);
    else if (_funcType == 1)
      y = F<nl_sigmoid>(y);
    else if (_funcType == 3)
      y = F<nl_exp>(y);
  }

  inline void ComputeForwardScore(Tensor<xpu, 3, dtype> x1, Tensor<xpu, 3, dtype> x2, Tensor<xpu, 3, dtype> x3, Tensor<xpu, 3, dtype> y) {
    int seq_size = y.size(0);

    for (int id = 0; id < seq_size; id++) {
      y[id] = dot(x1[id], _W1.T());
      y[id] += dot(x2[id], _W2.T());
      y[id] += dot(x3[id], _W3.T());
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

  inline void ComputeForwardScore(const std::vector<Tensor<xpu, 2, dtype> > &x1, const std::vector<Tensor<xpu, 2, dtype> > &x2,
      const std::vector<Tensor<xpu, 2, dtype> > &x3, std::vector<Tensor<xpu, 2, dtype> > &y) {
    int seq_size = y.size();

    for (int id = 0; id < seq_size; id++) {
      y[id] = dot(x1[id], _W1.T());
      y[id] += dot(x2[id], _W2.T());
      y[id] += dot(x3[id], _W3.T());
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
  inline void ComputeBackwardLoss(Tensor<xpu, 2, dtype> x1, Tensor<xpu, 2, dtype> x2, Tensor<xpu, 2, dtype> x3, Tensor<xpu, 2, dtype> y,
      Tensor<xpu, 2, dtype> ly, Tensor<xpu, 2, dtype> lx1, Tensor<xpu, 2, dtype> lx2, Tensor<xpu, 2, dtype> lx3, bool bclear = false) {
    //_gradW
    Tensor<xpu, 2, dtype> deri_yx(Shape2(y.size(0), y.size(1))), cly(Shape2(y.size(0), y.size(1)));
    AllocSpace(&deri_yx);
    AllocSpace(&cly);

    if(bclear) {
      lx1 = 0.0;
      lx2 = 0.0;
      lx3 = 0.0;
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
    //_gradW
    _gradW1 += dot(cly.T(), x1);
    _gradW2 += dot(cly.T(), x2);
    _gradW3 += dot(cly.T(), x3);

    //_gradb
    if (_bUseB)
      _gradb += cly;

    //lx
    lx1 += dot(cly, _W1);
    lx2 += dot(cly, _W2);
    lx3 += dot(cly, _W3);

    FreeSpace(&deri_yx);
    FreeSpace(&cly);
  }


  //please allocate the memory outside here
  inline void ComputeBackwardLoss(Tensor<xpu, 3, dtype> x1, Tensor<xpu, 3, dtype> x2, Tensor<xpu, 3, dtype> x3, Tensor<xpu, 3, dtype> y,
      Tensor<xpu, 3, dtype> ly, Tensor<xpu, 3, dtype> lx1, Tensor<xpu, 3, dtype> lx2, Tensor<xpu, 3, dtype> lx3, bool bclear = false) {
    int seq_size = y.size(0);
    int y_dim1 = y.size(1), y_dim2 = y.size(2);
    //_gradW
    Tensor<xpu, 2, dtype> deri_yx(Shape2(y_dim1, y_dim2)), cly(Shape2(y_dim1, y_dim2));
    AllocSpace(&deri_yx);
    AllocSpace(&cly);
    if(bclear) {
      lx1 = 0.0;
      lx2 = 0.0;
      lx3 = 0.0;
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
      _gradW1 += dot(cly.T(), x1[id]);
      _gradW2 += dot(cly.T(), x2[id]);
      _gradW3 += dot(cly.T(), x3[id]);

      //_gradb
      if (_bUseB)
        _gradb += cly;

      //lx
      lx1[id] += dot(cly, _W1);
      lx2[id] += dot(cly, _W2);
      lx3[id] += dot(cly, _W3);
    }

    FreeSpace(&deri_yx);
    FreeSpace(&cly);
  }


  //please allocate the memory outside here
  inline void ComputeBackwardLoss(const std::vector<Tensor<xpu, 2, dtype> > &x1, const std::vector<Tensor<xpu, 2, dtype> > &x2,
      const std::vector<Tensor<xpu, 2, dtype> > &x3, const std::vector<Tensor<xpu, 2, dtype> > &y,
      const std::vector<Tensor<xpu, 2, dtype> > &ly, std::vector<Tensor<xpu, 2, dtype> > &lx1,
      std::vector<Tensor<xpu, 2, dtype> > &lx2, std::vector<Tensor<xpu, 2, dtype> > &lx3, bool bclear = false) {
    int seq_size = y.size();
    assert(seq_size > 0);
    int y_dim1 = y[0].size(0), y_dim2 = y[0].size(1);
    //_gradW
    Tensor<xpu, 2, dtype> deri_yx(Shape2(y_dim1, y_dim2)), cly(Shape2(y_dim1, y_dim2));
    AllocSpace(&deri_yx);
    AllocSpace(&cly);
    if(bclear) {
      for (int id = 0; id < seq_size; id++) {
        lx1[id] = 0.0;
        lx2[id] = 0.0;
        lx3[id] = 0.0;
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
      _gradW1 += dot(cly.T(), x1[id]);
      _gradW2 += dot(cly.T(), x2[id]);
      _gradW3 += dot(cly.T(), x3[id]);

      //_gradb
      if (_bUseB)
        _gradb += cly;

      //lx
      lx1[id] += dot(cly, _W1);
      lx2[id] += dot(cly, _W2);
      lx3[id] += dot(cly, _W3);
    }

    FreeSpace(&deri_yx);
    FreeSpace(&cly);
  }

  inline void randomprint(int num) {
    static int nOSize, nISize1, nISize2, nISize3;
    nOSize = _W1.size(0);
    nISize1 = _W1.size(1);
    nISize2 = _W2.size(1);
    nISize3 = _W3.size(1);
    int count = 0;
    while (count < num) {
      int idx1 = rand() % nOSize;
      int idy1 = rand() % nISize1;
      int idx2 = rand() % nOSize;
      int idy2 = rand() % nISize2;
      int idx3 = rand() % nOSize;
      int idy3 = rand() % nISize3;

      std::cout << "_W1[" << idx1 << "," << idy1 << "]=" << _W1[idx1][idy1] << " ";
      std::cout << "_W2[" << idx2 << "," << idy2 << "]=" << _W2[idx2][idy2] << " ";
      std::cout << "_W3[" << idx3 << "," << idy3 << "]=" << _W3[idx3][idy3] << " ";

      if (_bUseB) {
        int idz = rand() % nOSize;
        std::cout << "_b[0][" << idz << "]=" << _b[0][idz] << " ";
      }
      count++;
    }

    std::cout << std::endl;
  }

  inline void updateAdaGrad(dtype regularizationWeight, dtype adaAlpha, dtype adaEps) {
    _gradW1 = _gradW1 + _W1 * regularizationWeight;
    _eg2W1 = _eg2W1 + _gradW1 * _gradW1;
    _W1 = _W1 - _gradW1 * adaAlpha / F<nl_sqrt>(_eg2W1 + adaEps);

    _gradW2 = _gradW2 + _W2 * regularizationWeight;
    _eg2W2 = _eg2W2 + _gradW2 * _gradW2;
    _W2 = _W2 - _gradW2 * adaAlpha / F<nl_sqrt>(_eg2W2 + adaEps);

    _gradW3 = _gradW3 + _W3 * regularizationWeight;
    _eg2W3 = _eg2W3 + _gradW3 * _gradW3;
    _W3 = _W3 - _gradW3 * adaAlpha / F<nl_sqrt>(_eg2W3 + adaEps);

    if (_bUseB) {
      _gradb = _gradb + _b * regularizationWeight;
      _eg2b = _eg2b + _gradb * _gradb;
      _b = _b - _gradb * adaAlpha / F<nl_sqrt>(_eg2b + adaEps);
    }

    clearGrad();
  }

  inline void clearGrad() {
    _gradW1 = 0;
    _gradW2 = 0;
    _gradW3 = 0;
    if (_bUseB)
      _gradb = 0;
  }

  void writeModel(LStream &outf) {
    SaveBinary(outf, _W1);
    SaveBinary(outf, _W2);
    SaveBinary(outf, _W3);
    SaveBinary(outf, _b);

    SaveBinary(outf, _gradW1);
    SaveBinary(outf, _gradW2);
    SaveBinary(outf, _gradW3);
    SaveBinary(outf, _gradb);

    SaveBinary(outf, _eg2W1);
    SaveBinary(outf, _eg2W2);
    SaveBinary(outf, _eg2W3);
    SaveBinary(outf, _eg2b);

    WriteBinary(outf, _bUseB);
    WriteBinary(outf, _funcType);
    // cout << "TrilayerLSTM " << _bUseB << _funcType << endl;
    // cout << "TrilayerLSTM value: " << _W3.size(0) << " and " << _W3.size(1) << " value " << _W3[0][1] << endl;


  }

  void loadModel(LStream &inf) {
    LoadBinary(inf, &_W1, false);
    LoadBinary(inf, &_W2, false);
    LoadBinary(inf, &_W3, false);
    LoadBinary(inf, &_b, false);

    LoadBinary(inf, &_gradW1, false);
    LoadBinary(inf, &_gradW2, false);
    LoadBinary(inf, &_gradW3, false);
    LoadBinary(inf, &_gradb, false);

    LoadBinary(inf, &_eg2W1, false);
    LoadBinary(inf, &_eg2W2, false);
    LoadBinary(inf, &_eg2W3, false);
    LoadBinary(inf, &_eg2b, false);

    ReadBinary(inf, _bUseB);
    ReadBinary(inf, _funcType);
    // cout << "TrilayerLSTM " << _bUseB << _funcType << endl;
    // cout << "TrilayerLSTM value: " << _W3.size(0) << " and " << _W3.size(1) << " value " << _W3[0][1]  << endl;
  }
  
};

#endif /* SRC_TriLayer_H_ */
