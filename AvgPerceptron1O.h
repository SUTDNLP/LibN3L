/*
 * AvgPerceptron1O.h
 *
 *  Created on: Oct 22, 2015
 *      Author: mszhang
 */

#ifndef AVGPERCEPTRON1O_H_
#define AVGPERCEPTRON1O_H_

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
class AvgPerceptron1O {

public:

  hash_set<int> _indexers;

  Tensor<xpu, 1, dtype> _W;

  Tensor<xpu, 1, dtype> _gradW;

  Tensor<xpu, 1, dtype> _sumW;

  int _max_update;
  NRVec<int> _last_update;

public:

  AvgPerceptron1O() {
    _indexers.clear();
  }

  inline void initial(int nISize, int seed = 0) {
    dtype bound = sqrt(6.0 / (nISize + 1));
    //dtype bound = 0.01;

    _W = NewTensor<xpu>(Shape1(nISize), d_zero);
    _gradW = NewTensor<xpu>(Shape1(nISize), d_zero);
    _sumW = NewTensor<xpu>(Shape1(nISize), d_one);

    _max_update = 0;
    _last_update.resize(nISize);
    _last_update = 0;
  }

  inline void initial(Tensor<xpu, 1, dtype> W) {
    static int nOSize, nISize;
    nISize = W.size(0);

    _W = NewTensor<xpu>(Shape1(nISize), d_zero);
    _gradW = NewTensor<xpu>(Shape1(nISize), d_zero);
    _sumW = NewTensor<xpu>(Shape1(nISize), d_one);
    Copy(_W, W);

    _max_update = 0;
    _last_update.resize(nISize);
    _last_update = 0;
  }

  inline void release() {
    FreeSpace(&_W);
    FreeSpace(&_gradW);
    FreeSpace(&_sumW);
    _indexers.clear();
  }

  virtual ~AvgPerceptron1O() {
    // TODO Auto-generated destructor stub
  }

  inline dtype squarenormAll() {
    dtype result = squarenorm(_gradW);

    return result;
  }

  inline void scaleGrad(dtype scale) {
    _gradW = _gradW * scale;
  }

public:
  void ComputeForwardScore(const std::vector<int>& x, dtype& y, bool bTrain = false) {
    static long long featNum, featId;
    featNum = x.size();
    y = 0.0;
    for (int idx = 0; idx < featNum; idx++) {
      featId = x[idx];
      if (featId >= _W.size(0))
        continue;
      if (bTrain)
        y += _W[featId];
      else
        y += sumWeight(featId);
      //y += _W[featId];
    }
  }

  // loss is stopped at this layer, since the input is one-hold alike
  void ComputeBackwardLoss(const std::vector<int>& x, dtype ly) {
    //_gradW
    static long long featNum, featId;
    featNum = x.size();
    for (int idx = 0; idx < featNum; idx++) {
      featId = x[idx];
      if (featId >= _W.size(0))
        continue;
      _indexers.insert(featId);
      _gradW[featId] += ly;
    }
  }

  void randomprint(int num) {
    static int nISize;
    nISize = _W.size(0);

    int count = 0;
    while (count < num) {
      int idx = rand() % nISize;
      std::cout << "_W[" << idx << "]=" << _W[idx] << " ";
      count++;
    }

    std::cout << std::endl;
  }

  void updateAdaGrad(dtype regularizationWeight, dtype adaAlpha, dtype adaEps) {
    static int startPos;

    static hash_set<int>::iterator it;

    _max_update++;

    for (it = _indexers.begin(); it != _indexers.end(); ++it) {
      int index = *it;
      _sumW[index] += (_max_update - _last_update[index]) * _W[index] - _gradW[index];
      _W[index] = _W[index] - _gradW[index];
      _last_update[index] = _max_update;
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

  }

  dtype sumWeight(int featId) {
    if (_last_update[featId] < _max_update) {
      int times = _max_update - _last_update[featId];
      _sumW[featId] += _W[featId] * times;
      _last_update[featId] = _max_update;
    }

    return _sumW[featId];
  }

  void writeModel(LStream &outf) {
    SaveBinary(outf, _W);
    SaveBinary(outf, _gradW);
    SaveBinary(outf, _sumW);
    WriteBinary(outf, _max_update);
    WriteVector(outf, _last_update);

  }
  
  void loadModel(LStream &inf) {
    LoadBinary(inf, &_W, false);
    LoadBinary(inf, &_gradW, false);
    LoadBinary(inf, &_sumW, false);
    ReadBinary(inf, _max_update);
    ReadVector(inf, _last_update);
  }
};

#endif /* AVGPERCEPTRON1O_H_ */
