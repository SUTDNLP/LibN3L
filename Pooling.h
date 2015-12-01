#ifndef POOL
#define POOL

#include "tensor.h"
#include "MyLib.h"

using namespace std;
using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;


// pooling functions, include sum, std, pro, max, min, avg with all indexes or selected indexes
// We use a poolIndex to record the pooling weights of each input vector
// General formalization for linear pooling: out = \sum \alpha_i input_i,
// which can be extended to attention model if \alpha_i is decided by gates.
// General for non-linear pooling:  out =  (\sum \alpha_i (input_i)^p)^{1/p}


//sum pooling
template<typename xpu>
inline void sumpool_forward(Tensor<xpu, 3, dtype> inputs, Tensor<xpu, 2, dtype> output, Tensor<xpu, 3, dtype> poolIndex) {
  int num = inputs.size(0);
  poolIndex = 1.0;
  output = 0.0;
  for(int idx = 0; idx < num; idx++)
  {
    output = output + inputs[idx] * poolIndex[idx];
  }
}

template<typename xpu>
inline void sumpool_forward(Tensor<xpu, 3, dtype> inputs, Tensor<xpu, 2, dtype> output) {
  int num = inputs.size(0);
  output = 0.0;
  for(int idx = 0; idx < num; idx++)
  {
    output = output + inputs[idx];
  }
}


//std pooling
template<typename xpu>
inline void stdpool_forward(Tensor<xpu, 3, dtype> inputs, Tensor<xpu, 2, dtype> output, Tensor<xpu, 3, dtype> poolIndex) {
  int num = inputs.size(0);
  poolIndex = 0.0;
  output = 1e-20;
  // poolIndex is no sense here.
  for(int idx = 0; idx < num; idx++)
  {
    output = output + inputs[idx] * inputs[idx];
  }
  output = F<nl_sqrt>(output);
  for(int idx = 0; idx < num; idx++) {
    poolIndex[idx] = inputs[idx] / output;
  }
}

//pro pooling
template<typename xpu>
inline void propool_forward(Tensor<xpu, 3, dtype> inputs, Tensor<xpu, 2, dtype> output, Tensor<xpu, 3, dtype> poolIndex) {
  int num = inputs.size(0);
  poolIndex = 0.0;
  output = 1.0;
  // poolIndex is no sense here.
  for(int idx = 0; idx < num; idx++)
  {
    output = output * inputs[idx];
  }
  for(int idx = 0; idx < num; idx++) {
    for(int idy = 0; idy < inputs.size(1); idy++) {
      for(int idz = 0; idz < inputs.size(2); idz++){
        if(abs(output[idy][idz]) < 1e-20 )
        {
          output[idy][idz] = 0.0;
          poolIndex[idx][idy][idz] = 0.0;
        }
        else
        {
          poolIndex[idx][idy][idz] = output[idy][idz] / inputs[idx][idy][idz];
        }
      }
    }
  }
}

//avg pooling
template<typename xpu>
inline void avgpool_forward(Tensor<xpu, 3, dtype> inputs, Tensor<xpu, 2, dtype> output, Tensor<xpu, 3, dtype> poolIndex) {
  int num = inputs.size(0);
  poolIndex = 1.0 / num;
  output = 0.0;
  for(int idx = 0; idx < num; idx++)
  {
    output = output + inputs[idx] * poolIndex[idx];
  }
}

//avg pooling
template<typename xpu>
inline void avgpool_forward(Tensor<xpu, 3, dtype> inputs, Tensor<xpu, 2, dtype> output) {
  int num = inputs.size(0);
  output = 0.0;
  for(int idx = 0; idx < num; idx++)
  {
    output = output + inputs[idx] / num;
  }
}

//max pooling
template<typename xpu>
inline void maxpool_forward(Tensor<xpu, 3, dtype> inputs, Tensor<xpu, 2, dtype> output, Tensor<xpu, 3, dtype> poolIndex) {
  int num = inputs.size(0);
  int dim1 = inputs.size(1);
  int dim2 = inputs.size(2);
  output = 0.0; poolIndex = 0.0;
  for(int idx = 0; idx < dim1; idx++)
  {
    for(int idy = 0; idy < dim2; idy++)
    {
      dtype max = inputs[0][idx][idy];
      int maxId = 0;
      for(int idz = 1; idz < num; idz++)
      {
        if(inputs[idz][idx][idy] > max)
        {
          max = inputs[idz][idx][idy];
          maxId = idz;
        }
      }
      output[idx][idy] = max;
      poolIndex[maxId][idx][idy] = 1.0;
    }
  }
}


//min pooling
template<typename xpu>
inline void minpool_forward(Tensor<xpu, 3, dtype> inputs, Tensor<xpu, 2, dtype> output, Tensor<xpu, 3, dtype> poolIndex) {
  int num = inputs.size(0);
  int dim1 = inputs.size(1);
  int dim2 = inputs.size(2);
  output = 0.0; poolIndex = 0.0;
  for(int idx = 0; idx < dim1; idx++)
  {
    for(int idy = 0; idy < dim2; idy++)
    {
      dtype min = inputs[0][idx][idy];
      int minId = 0;
      for(int idz = 1; idz < num; idz++)
      {
        if(inputs[idz][idx][idy] < min)
        {
          min = inputs[idz][idx][idy];
          minId = idz;
        }
      }
      output[idx][idy] = min;
      poolIndex[minId][idx][idy] = 1.0;
    }
  }
}

//sum pooling, selected indexes
template<typename xpu>
inline void sumpool_forward(Tensor<xpu, 3, dtype> inputs, Tensor<xpu, 2, dtype> output, Tensor<xpu, 3, dtype> poolIndex, const hash_set<int>& indexes) {
  static hash_set<int>::iterator it;
  output = 0.0;
  poolIndex = 0.0;
  for (it = indexes.begin(); it != indexes.end(); ++it)
  {
    poolIndex[*it] = 1.0;
    output = output + inputs[*it] * poolIndex[*it];
  }
}

//std pooling, selected indexes
template<typename xpu>
inline void stdpool_forward(Tensor<xpu, 3, dtype> inputs, Tensor<xpu, 2, dtype> output, Tensor<xpu, 3, dtype> poolIndex, const hash_set<int>& indexes) {
  poolIndex = 0.0;
  output = 1e-20;
  // poolIndex is no sense here.
  static hash_set<int>::iterator it;
  int idx;
  for (it = indexes.begin(); it != indexes.end(); ++it)
  {
    idx = *it;
    output = output + inputs[idx] * inputs[idx];
  }
  output = F<nl_sqrt>(output);
  for (it = indexes.begin(); it != indexes.end(); ++it) {
    idx = *it;
    poolIndex[idx] = inputs[idx] / output;
  }
}

//pro pooling, selected indexes
template<typename xpu>
inline void propool_forward(Tensor<xpu, 3, dtype> inputs, Tensor<xpu, 2, dtype> output, Tensor<xpu, 3, dtype> poolIndex, const hash_set<int>& indexes) {
  poolIndex = 0.0;
  output = 1.0;
  // poolIndex is no sense here.
  static hash_set<int>::iterator it;
  int idx;
  for (it = indexes.begin(); it != indexes.end(); ++it) {
    idx = *it;
    output = output * inputs[idx];
  }
  for (it = indexes.begin(); it != indexes.end(); ++it) {
    idx = *it;
    for(int idy = 0; idy < inputs.size(1); idy++) {
      for(int idz = 0; idz < inputs.size(2); idz++){
        if(abs(output[idy][idz]) < 1e-20 )
        {
          output[idy][idz] = 0.0;
          poolIndex[idx][idy][idz] = 0.0;
        }
        else
        {
          poolIndex[idx][idy][idz] = output[idy][idz] / inputs[idx][idy][idz];
        }
      }
    }
  }
}

//avg pooling, selected indexes
template<typename xpu>
inline void avgpool_forward(Tensor<xpu, 3, dtype> inputs, Tensor<xpu, 2, dtype> output, Tensor<xpu, 3, dtype> poolIndex, const hash_set<int>& indexes) {
  int num = indexes.size();
  poolIndex = 0.0;
  if(num == 0)
  {
    return;
  }
  static hash_set<int>::iterator it;
  output = 0.0;
  for (it = indexes.begin(); it != indexes.end(); ++it)
  {
    poolIndex[*it] = 1.0 / num;
    output = output + inputs[*it] * poolIndex[*it];
  }
}

//max pooling, selected indexes
template<typename xpu>
inline void maxpool_forward(Tensor<xpu, 3, dtype> inputs, Tensor<xpu, 2, dtype> output, Tensor<xpu, 3, dtype> poolIndex, const hash_set<int>& indexes) {
  int dim1 = inputs.size(1);
  int dim2 = inputs.size(2);
  output = 0.0;
  poolIndex = 0.0;
  static hash_set<int>::iterator it;
  for(int idx = 0; idx < dim1; idx++)
  {
    for(int idy = 0; idy < dim2; idy++)
    {
      dtype max = inputs[0][idx][idy];
      int maxId = -1;
      for (it = indexes.begin(); it != indexes.end(); ++it)
      {
        if(maxId == -1 || inputs[*it][idx][idy] > max)
        {
          max = inputs[*it][idx][idy];
          maxId = *it;
        }
      }
      if(maxId != -1)
      {
        output[idx][idy] = max;
        poolIndex[maxId][idx][idy] = 1.0;
      }
    }
  }
}


//min pooling, selected indexes
template<typename xpu>
inline void minpool_forward(Tensor<xpu, 3, dtype> inputs, Tensor<xpu, 2, dtype> output, Tensor<xpu, 3, dtype> poolIndex, const hash_set<int>& indexes) {
  int dim1 = inputs.size(1);
  int dim2 = inputs.size(2);
  output = 0.0;
  poolIndex = 0.0;
  static hash_set<int>::iterator it;
  for(int idx = 0; idx < dim1; idx++)
  {
    for(int idy = 0; idy < dim2; idy++)
    {
      dtype min = inputs[0][idx][idy];
      int minId = -1;
      for (it = indexes.begin(); it != indexes.end(); ++it)
      {
        if(minId == -1 || inputs[*it][idx][idy] < min)
        {
          min = inputs[*it][idx][idy];
          minId = *it;
        }
      }
      if(minId != -1)
      {
        output[idx][idy] = min;
        poolIndex[minId][idx][idy] = 1.0;
      }
    }
  }
}

//vector-style
//sum pooling
template<typename xpu>
inline void sumpool_forward(const vector<Tensor<xpu, 2, dtype> > &inputs, Tensor<xpu, 2, dtype> output, vector<Tensor<xpu, 2, dtype> > &poolIndex) {
  int num = inputs.size();
  poolIndex = 1.0;
  output = 0.0;
  for(int idx = 0; idx < num; idx++)
  {
    output = output + inputs[idx] * poolIndex[idx];
  }
}

template<typename xpu>
inline void sumpool_forward(const vector<Tensor<xpu, 2, dtype> > &inputs, Tensor<xpu, 2, dtype> output) {
  int num = inputs.size();
  output = 0.0;
  for(int idx = 0; idx < num; idx++)
  {
    output = output + inputs[idx];
  }
}


//std pooling
template<typename xpu>
inline void stdpool_forward(const vector<Tensor<xpu, 2, dtype> > &inputs, Tensor<xpu, 2, dtype> output, vector<Tensor<xpu, 2, dtype> > &poolIndex) {
  int num = inputs.size();
  poolIndex = 0.0;
  output = 1e-20;
  // poolIndex is no sense here.
  for(int idx = 0; idx < num; idx++)
  {
    output = output + inputs[idx] * inputs[idx];
  }
  output = F<nl_sqrt>(output);
  for(int idx = 0; idx < num; idx++) {
    poolIndex[idx] = inputs[idx] / output;
  }
}

//pro pooling
template<typename xpu>
inline void propool_forward(const vector<Tensor<xpu, 2, dtype> > &inputs, Tensor<xpu, 2, dtype> output, vector<Tensor<xpu, 2, dtype> > &poolIndex) {
  int num = inputs.size();
  poolIndex = 0.0;
  output = 1.0;
  // poolIndex is no sense here.
  for(int idx = 0; idx < num; idx++)
  {
    output = output * inputs[idx];
  }
  for(int idx = 0; idx < num; idx++) {
    for(int idy = 0; idy < inputs[0].size(0); idy++) {
      for(int idz = 0; idz < inputs[0].size(1); idz++){
        if(abs(output[idy][idz]) < 1e-20 )
        {
          output[idy][idz] = 0.0;
          poolIndex[idx][idy][idz] = 0.0;
        }
        else
        {
          poolIndex[idx][idy][idz] = output[idy][idz] / inputs[idx][idy][idz];
        }
      }
    }
  }
}

//avg pooling
template<typename xpu>
inline void avgpool_forward(const vector<Tensor<xpu, 2, dtype> > &inputs, Tensor<xpu, 2, dtype> output, vector<Tensor<xpu, 2, dtype> > &poolIndex) {
  int num = inputs.size();
  poolIndex = 1.0 / num;
  output = 0.0;
  for(int idx = 0; idx < num; idx++)
  {
    output = output + inputs[idx] * poolIndex[idx];
  }
}

//avg pooling
template<typename xpu>
inline void avgpool_forward(const vector<Tensor<xpu, 2, dtype> > &inputs, Tensor<xpu, 2, dtype> output) {
  int num = inputs.size();
  output = 0.0;
  for(int idx = 0; idx < num; idx++)
  {
    output = output + inputs[idx] / num;
  }
}

//max pooling
template<typename xpu>
inline void maxpool_forward(const vector<Tensor<xpu, 2, dtype> > &inputs, Tensor<xpu, 2, dtype> output, vector<Tensor<xpu, 2, dtype> > &poolIndex) {
  int num = inputs.size();
  int dim1 = inputs[0].size(0);
  int dim2 = inputs[0].size(1);
  output = 0.0; poolIndex = 0.0;
  for(int idx = 0; idx < dim1; idx++)
  {
    for(int idy = 0; idy < dim2; idy++)
    {
      dtype max = inputs[0][idx][idy];
      int maxId = 0;
      for(int idz = 1; idz < num; idz++)
      {
        if(inputs[idz][idx][idy] > max)
        {
          max = inputs[idz][idx][idy];
          maxId = idz;
        }
      }
      output[idx][idy] = max;
      poolIndex[maxId][idx][idy] = 1.0;
    }
  }
}


//min pooling
template<typename xpu>
inline void minpool_forward(const vector<Tensor<xpu, 2, dtype> > &inputs, Tensor<xpu, 2, dtype> output, vector<Tensor<xpu, 2, dtype> > &poolIndex) {
  int num = inputs.size();
  int dim1 = inputs[0].size(0);
  int dim2 = inputs[0].size(1);
  output = 0.0; poolIndex = 0.0;
  for(int idx = 0; idx < dim1; idx++)
  {
    for(int idy = 0; idy < dim2; idy++)
    {
      dtype min = inputs[0][idx][idy];
      int minId = 0;
      for(int idz = 1; idz < num; idz++)
      {
        if(inputs[idz][idx][idy] < min)
        {
          min = inputs[idz][idx][idy];
          minId = idz;
        }
      }
      output[idx][idy] = min;
      poolIndex[minId][idx][idy] = 1.0;
    }
  }
}

//sum pooling, selected indexes
template<typename xpu>
inline void sumpool_forward(const vector<Tensor<xpu, 2, dtype> > &inputs, Tensor<xpu, 2, dtype> output, vector<Tensor<xpu, 2, dtype> > &poolIndex, const hash_set<int>& indexes) {
  static hash_set<int>::iterator it;
  output = 0.0;
  poolIndex = 0.0;
  for (it = indexes.begin(); it != indexes.end(); ++it)
  {
    poolIndex[*it] = 1.0;
    output = output + inputs[*it] * poolIndex[*it];
  }
}

//std pooling, selected indexes
template<typename xpu>
inline void stdpool_forward(const vector<Tensor<xpu, 2, dtype> > &inputs, Tensor<xpu, 2, dtype> output, vector<Tensor<xpu, 2, dtype> > &poolIndex, const hash_set<int>& indexes) {
  poolIndex = 0.0;
  output = 1e-20;
  // poolIndex is no sense here.
  static hash_set<int>::iterator it;
  int idx;
  for (it = indexes.begin(); it != indexes.end(); ++it)
  {
    idx = *it;
    output = output + inputs[idx] * inputs[idx];
  }
  output = F<nl_sqrt>(output);
  for (it = indexes.begin(); it != indexes.end(); ++it) {
    idx = *it;
    poolIndex[idx] = inputs[idx] / output;
  }
}

//pro pooling, selected indexes
template<typename xpu>
inline void propool_forward(const vector<Tensor<xpu, 2, dtype> > &inputs, Tensor<xpu, 2, dtype> output, vector<Tensor<xpu, 2, dtype> > &poolIndex, const hash_set<int>& indexes) {
  poolIndex = 0.0;
  output = 1.0;
  // poolIndex is no sense here.
  static hash_set<int>::iterator it;
  int idx;
  for (it = indexes.begin(); it != indexes.end(); ++it) {
    idx = *it;
    output = output * inputs[idx];
  }
  for (it = indexes.begin(); it != indexes.end(); ++it) {
    idx = *it;
    for(int idy = 0; idy < inputs[0].size(0); idy++) {
      for(int idz = 0; idz < inputs[0].size(1); idz++){
        if(abs(output[idy][idz]) < 1e-20 )
        {
          output[idy][idz] = 0.0;
          poolIndex[idx][idy][idz] = 0.0;
        }
        else
        {
          poolIndex[idx][idy][idz] = output[idy][idz] / inputs[idx][idy][idz];
        }
      }
    }
  }
}

//avg pooling, selected indexes
template<typename xpu>
inline void avgpool_forward(const vector<Tensor<xpu, 2, dtype> > &inputs, Tensor<xpu, 2, dtype> output, vector<Tensor<xpu, 2, dtype> > &poolIndex, const hash_set<int>& indexes) {
  int num = indexes.size();
  poolIndex = 0.0;
  if(num == 0)
  {
    return;
  }
  static hash_set<int>::iterator it;
  output = 0.0;
  for (it = indexes.begin(); it != indexes.end(); ++it)
  {
    poolIndex[*it] = 1.0 / num;
    output = output + inputs[*it] * poolIndex[*it];
  }
}

//max pooling, selected indexes
template<typename xpu>
inline void maxpool_forward(const vector<Tensor<xpu, 2, dtype> > &inputs, Tensor<xpu, 2, dtype> output, vector<Tensor<xpu, 2, dtype> > &poolIndex, const hash_set<int>& indexes) {
  int dim1 = inputs[0].size(0);
  int dim2 = inputs[0].size(1);
  output = 0.0;
  poolIndex = 0.0;
  static hash_set<int>::iterator it;
  for(int idx = 0; idx < dim1; idx++)
  {
    for(int idy = 0; idy < dim2; idy++)
    {
      dtype max = inputs[0][idx][idy];
      int maxId = -1;
      for (it = indexes.begin(); it != indexes.end(); ++it)
      {
        if(maxId == -1 || inputs[*it][idx][idy] > max)
        {
          max = inputs[*it][idx][idy];
          maxId = *it;
        }
      }
      if(maxId != -1)
      {
        output[idx][idy] = max;
        poolIndex[maxId][idx][idy] = 1.0;
      }
    }
  }
}


//min pooling, selected indexes
template<typename xpu>
inline void minpool_forward(const vector<Tensor<xpu, 2, dtype> > &inputs, Tensor<xpu, 2, dtype> output, vector<Tensor<xpu, 2, dtype> > &poolIndex, const hash_set<int>& indexes) {
  int dim1 = inputs[0].size(0);
  int dim2 = inputs[0].size(1);
  output = 0.0;
  poolIndex = 0.0;
  static hash_set<int>::iterator it;
  for(int idx = 0; idx < dim1; idx++)
  {
    for(int idy = 0; idy < dim2; idy++)
    {
      dtype min = inputs[0][idx][idy];
      int minId = -1;
      for (it = indexes.begin(); it != indexes.end(); ++it)
      {
        if(minId == -1 || inputs[*it][idx][idy] < min)
        {
          min = inputs[*it][idx][idy];
          minId = *it;
        }
      }
      if(minId != -1)
      {
        output[idx][idy] = min;
        poolIndex[minId][idx][idy] = 1.0;
      }
    }
  }
}


// pooling, back-propagation
// bclear by false denotes that the losses are accumulated.
template<typename xpu>
inline void pool_backward(Tensor<xpu, 2, dtype> outputLoss, Tensor<xpu, 3, dtype> poolIndex, Tensor<xpu, 3, dtype> inputsLoss, bool bclear = false) {
  int num = inputsLoss.size(0);
  if(bclear)inputsLoss = 0.0;
  for(int idx = 0; idx < num; idx++)
  {
    inputsLoss[idx] += outputLoss * poolIndex[idx];
  }
}

// The indexes of the following two pooling methods are fixed, needs no computation.
template<typename xpu>
inline void sumpool_backward(Tensor<xpu, 2, dtype> outputLoss, Tensor<xpu, 3, dtype> inputsLoss, bool bclear = false) {
  int num = inputsLoss.size(0);
  if(bclear)inputsLoss = 0.0;
  for(int idx = 0; idx < num; idx++)
  {
    inputsLoss[idx] += outputLoss;
  }
}

template<typename xpu>
inline void avgpool_backward(Tensor<xpu, 2, dtype> outputLoss, Tensor<xpu, 3, dtype> inputsLoss, bool bclear = false) {
  int num = inputsLoss.size(0);
  if(bclear)inputsLoss = 0.0;
  for(int idx = 0; idx < num; idx++)
  {
    inputsLoss[idx] += outputLoss / num;
  }
}

// vector-style
// pooling, back-propagation
// bclear by false denotes that the losses are accumulated.
template<typename xpu>
inline void pool_backward(Tensor<xpu, 2, dtype> outputLoss, const vector<Tensor<xpu, 2, dtype> > &poolIndex, vector<Tensor<xpu, 2, dtype> > &inputsLoss, bool bclear = false) {
  int num = inputsLoss.size();
  for(int idx = 0; idx < num; idx++)
  {
    if(bclear)inputsLoss[idx] = 0.0;
    inputsLoss[idx] += outputLoss * poolIndex[idx];
  }
}

// The indexes of the following two pooling methods are fixed, needs no computation.
template<typename xpu>
inline void sumpool_backward(Tensor<xpu, 2, dtype> outputLoss, vector<Tensor<xpu, 2, dtype> > &inputsLoss, bool bclear = false) {
  int num = inputsLoss.size();
  for(int idx = 0; idx < num; idx++)
  {
    if(bclear)inputsLoss[idx] = 0.0;
    inputsLoss[idx] += outputLoss;
  }
}

template<typename xpu>
inline void avgpool_backward(Tensor<xpu, 2, dtype> outputLoss, vector<Tensor<xpu, 2, dtype> > &inputsLoss, bool bclear = false) {
  int num = inputsLoss.size();
  for(int idx = 0; idx < num; idx++)
  {
    if(bclear)inputsLoss[idx] = 0.0;
    inputsLoss[idx] += outputLoss / num;
  }
}

#endif
