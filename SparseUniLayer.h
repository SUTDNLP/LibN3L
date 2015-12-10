/*
 * SparseUniLayer.h
 *
 *  Created on: Mar 18, 2015
 *      Author: mszhang
 */

#ifndef SRC_SparseUniLayer_H_
#define SRC_SparseUniLayer_H_
#include "tensor.h"
#include "Utiltensor.h"
#include "MyLib.h"

using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;

// Weight updating process implemented without theory support,
// but recently find an EMNLP 2015 paper "An Empirical Analysis of Optimization for Max-Margin NLP"
// In all my papers that use adagrad for sparse features, I use it for parameter updating.

template<typename xpu>
class SparseUniLayer {

public:

  hash_set<int> _indexers;

  Tensor<xpu, 2, dtype> _W;
  Tensor<xpu, 2, dtype> _b;

  Tensor<xpu, 2, dtype> _gradW;
  Tensor<xpu, 2, dtype> _gradb;

  Tensor<xpu, 2, dtype> _eg2W;
  Tensor<xpu, 2, dtype> _eg2b;

  Tensor<xpu, 2, dtype> _ftW;

  bool _bUseB;

  int _funcType; // 0: tanh, 1: sigmod, 2: f(x)=x, 3: exp

  int _max_update;
  NRVec<int> _last_update;


public:

  SparseUniLayer() {
    _indexers.clear();
  }

  inline void initial(int nOSize, int nISize, bool bUseB = true, int seed = 0, int funcType = 0) {
    dtype bound = sqrt(6.0 / (nOSize + nISize + 1));
    //dtype bound = 0.01;

    _W = NewTensor<xpu>(Shape2(nISize, nOSize), d_zero);
    _gradW = NewTensor<xpu>(Shape2(nISize, nOSize), d_zero);
    _eg2W = NewTensor<xpu>(Shape2(nISize, nOSize), d_zero);
    _ftW = NewTensor<xpu>(Shape2(nISize, nOSize), d_one);

    _b = NewTensor<xpu>(Shape2(1, nOSize), d_zero);
    _gradb = NewTensor<xpu>(Shape2(1, nOSize), d_zero);
    _eg2b = NewTensor<xpu>(Shape2(1, nOSize), d_zero);

    random(_W, -1.0 * bound, 1.0 * bound, seed);
    random(_b, -1.0 * bound, 1.0 * bound, seed + 1);

    _bUseB = bUseB;
    _funcType = funcType;

    _max_update = 0;
    _last_update.resize(nISize);
    _last_update = 0;
  }

  inline void initial(Tensor<xpu, 2, dtype> W, Tensor<xpu, 2, dtype> b, bool bUseB = true, int funcType = 0) {
    static int nOSize, nISize;
    nISize = W.size(0);
    nOSize = W.size(1);

    _W = NewTensor<xpu>(Shape2(nOSize, nISize), d_zero);
    _gradW = NewTensor<xpu>(Shape2(nOSize, nISize), d_zero);
    _eg2W = NewTensor<xpu>(Shape2(nOSize, nISize), d_zero);
    _ftW = NewTensor<xpu>(Shape2(nOSize, nISize), d_one);
    Copy(_W, W);

    _b = NewTensor<xpu>(Shape2(1, nOSize), d_zero);
    _gradb = NewTensor<xpu>(Shape2(1, nOSize), d_zero);
    _eg2b = NewTensor<xpu>(Shape2(1, nOSize), d_zero);

    if (bUseB)
      Copy(_b, b);

    _bUseB = bUseB;
    _funcType = funcType;

    _max_update = 0;
    _last_update.resize(nISize);
    _last_update = 0;
  }

  inline void release() {
    FreeSpace(&_W);
    FreeSpace(&_gradW);
    FreeSpace(&_eg2W);
    FreeSpace(&_ftW);
    FreeSpace(&_b);
    FreeSpace(&_gradb);
    FreeSpace(&_eg2b);
    _indexers.clear();
  }

  virtual ~SparseUniLayer() {
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
  void ComputeForwardScore(const std::vector<int>& x, Tensor<xpu, 2, dtype> y) {
    static long long featNum, featId;
    featNum = x.size();
    y = 0.0;
    for (int idx = 0; idx < featNum; idx++) {
      featId = x[idx];
      updateSparseWeight(featId);
      y[0] += _W[featId];
    }

    if (_bUseB)
      y = y + _b;
    if (_funcType == 0)
      y = F<nl_tanh>(y);
    else if (_funcType == 1)
      y = F<nl_sigmoid>(y);
    else if (_funcType == 3)
      y = F<nl_exp>(y);
  }

  void ComputeForwardScore(const std::vector<std::vector<int> >& x, Tensor<xpu, 3, dtype> y) {
    static long long featNum, featId;

    int seq_size = y.size(0);

    for (int id = 0; id < seq_size; id++) {
      featNum = x[id].size();
      y[id] = 0.0;
      for (int idx = 0; idx < featNum; idx++) {
        featId = x[id][idx];
        updateSparseWeight(featId);
        y[id][0] += _W[featId];
      }

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

  void ComputeForwardScore(const std::vector<std::vector<int> >& x, std::vector<Tensor<xpu, 2, dtype> > &y) {
    static long long featNum, featId;
    int seq_size = y.size();

    for (int id = 0; id < seq_size; id++) {
      featNum = x[id].size();
      y[id] = 0.0;
      for (int idx = 0; idx < featNum; idx++) {
        featId = x[id][idx];
        updateSparseWeight(featId);
        y[id][0] += _W[featId];
      }

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
  // loss is stopped at this layer, since the input is one-hold alike
  void ComputeBackwardLoss(const std::vector<int>& x, Tensor<xpu, 2, dtype> y, Tensor<xpu, 2, dtype> ly) {
    Tensor<xpu, 2, dtype> deri_yx(Shape2(y.size(0), y.size(1))), cly(Shape2(y.size(0), y.size(1)));
    AllocSpace(&deri_yx);
    AllocSpace(&cly);
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
    static long long featNum, featId;
    featNum = x.size();
    for (int idx = 0; idx < featNum; idx++) {
      featId = x[idx];
      _indexers.insert(featId);
      _gradW[featId] += cly[0];
    }

    if (_bUseB)
      _gradb = _gradb + cly;

    FreeSpace(&deri_yx);
    FreeSpace(&cly);
  }

  void ComputeBackwardLoss(const std::vector<std::vector<int> >& x, Tensor<xpu, 3, dtype> y, Tensor<xpu, 3, dtype> ly) {
    int seq_size = y.size(0);
    int y_dim1 = y.size(1), y_dim2 = y.size(2);

    static long long featNum, featId;
    Tensor<xpu, 2, dtype> deri_yx(Shape2(y_dim1, y_dim2)), cly(Shape2(y_dim1, y_dim2));
    AllocSpace(&deri_yx);
    AllocSpace(&cly);

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
      featNum = x[id].size();
      for (int idx = 0; idx < featNum; idx++) {
        featId = x[id][idx];
        _indexers.insert(featId);
        _gradW[featId] += cly[0];
      }

      if (_bUseB)
        _gradb = _gradb + cly;
    }

    FreeSpace(&deri_yx);
    FreeSpace(&cly);
  }

  void ComputeBackwardLoss(const std::vector<std::vector<int> >& x, const std::vector<Tensor<xpu, 2, dtype> > &y,
      const std::vector<Tensor<xpu, 2, dtype> > &ly) {
    int seq_size = y.size();
    assert(seq_size > 0);
    int y_dim1 = y[0].size(0), y_dim2 = y[0].size(1);

    static long long featNum, featId, startPos;
    Tensor<xpu, 2, dtype> deri_yx(Shape2(y_dim1, y_dim2)), cly(Shape2(y_dim1, y_dim2));
    AllocSpace(&deri_yx);
    AllocSpace(&cly);

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
      featNum = x[id].size();
      for (int idx = 0; idx < featNum; idx++) {
        featId = x[id][idx];
        _indexers.insert(featId);
        _gradW[featId] += cly[0];
      }

      if (_bUseB)
        _gradb = _gradb + cly;
    }

    FreeSpace(&deri_yx);
    FreeSpace(&cly);
  }

  void randomprint(int num) {
    static int nOSize, nISize;
    nISize = _W.size(0);
    nOSize = _W.size(1);

    int count = 0;
    while (count < num) {
      int idx = rand() % nOSize;
      int idy = rand() % nISize;

      std::cout << "_W[" << idx << "," << idy << "]=" << _W[idy][idx] << " ";

      if (_bUseB) {
        int idz = rand() % nOSize;
        std::cout << "_b[0][" << idz << "]=" << _b[0][idz] << " ";
      }
      count++;
    }

    std::cout << std::endl;
  }

  void updateAdaGrad(dtype regularizationWeight, dtype adaAlpha, dtype adaEps) {
    static int startPos;

    static hash_set<int>::iterator it;

    _max_update++;

    Tensor<xpu, 1, dtype> sqrt_eg2W = NewTensor<xpu>(Shape1(_W.size(1)), d_zero);

    for (it = _indexers.begin(); it != _indexers.end(); ++it) {
      int index = *it;
      _eg2W[index] = _eg2W[index] + _gradW[index] * _gradW[index];
      sqrt_eg2W = F<nl_sqrt>(_eg2W[index] + adaEps);
      _W[index] = (_W[index] * sqrt_eg2W - _gradW[index] * adaAlpha) / (adaAlpha * regularizationWeight + sqrt_eg2W);
      _ftW[index] = sqrt_eg2W / (adaAlpha * regularizationWeight + sqrt_eg2W);
    }

    FreeSpace(&sqrt_eg2W);

    if (_bUseB) {
      _gradb = _gradb + _b * regularizationWeight;
      _eg2b = _eg2b + _gradb * _gradb;
      _b = _b - _gradb * adaAlpha / F<nl_sqrt>(_eg2b + adaEps);
    }

    clearGrad();
  }

  void clearGrad() {
    static hash_set<int>::iterator it;

    for (it = _indexers.begin(); it != _indexers.end(); ++it) {
      int index = *it;
      _gradW[index] = 0.0;
    }

    _indexers.clear();
    if (_bUseB)
      _gradb = 0.0;
  }

  void updateSparseWeight(long long featId) {
    if (_last_update[featId] < _max_update) {
      int times = _max_update - _last_update[featId];
      _W[featId] = _W[featId] * F<nl_exp>(times * F<nl_log>(_ftW[featId]));
      _last_update[featId] = _max_update;
    }
  }

  void writeModel(LStream &outf) {
    SaveBinary(outf, _W);
    SaveBinary(outf, _b);
    SaveBinary(outf, _gradW);
    SaveBinary(outf, _gradb);
    SaveBinary(outf, _eg2W);
    SaveBinary(outf, _eg2b);
    SaveBinary(outf, _ftW);


    WriteBinary(outf, _bUseB);
    WriteBinary(outf, _funcType);
    WriteBinary(outf, _max_update);
    WriteVector(outf, _last_update);
  }

  void loadModel(LStream &inf) {
    LoadBinary(inf, &_W, false);
    LoadBinary(inf, &_b, false);
    LoadBinary(inf, &_gradW, false);
    LoadBinary(inf, &_gradb, false);
    LoadBinary(inf, &_eg2W, false);
    LoadBinary(inf, &_eg2b, false);
    LoadBinary(inf, &_ftW, false);

    ReadBinary(inf, _bUseB);
    ReadBinary(inf, _funcType);
    ReadBinary(inf, _max_update);
    ReadVector(inf, _last_update);
  }


};

#endif /* SRC_SparseUniLayer_H_ */
