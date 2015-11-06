#ifndef SOFTMAXLOSS
#define SOFTMAXLOSS

#include "tensor.h"
#include "MyLib.h"
#include "Metric.h"

using namespace std;
using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;

template<typename xpu>
inline dtype softmax_loss(const vector<Tensor<xpu, 2, dtype> > &output, const vector<vector<int> > &answers, vector<Tensor<xpu, 2, dtype> > &loutput,
    Metric & eval, int batchsize = 1) {
  int seqsize = output.size();
  if (answers.size() != seqsize || seqsize == 0) {
    std::cerr << "softmax_loss error: vector size or context size invalid" << std::endl;
  }

  int dim1 = output[0].size(0), dim2 = output[0].size(1);
  int odim1 = loutput[0].size(0), odim2 = loutput[0].size(1);
  int labelsize = answers[0].size();

  if (labelsize != odim2 || dim2 != odim2 || dim1 != 1 || odim1 != 1) {
    std::cerr << "softmax_loss error: dim size invalid" << std::endl;
  }

  Tensor<xpu, 3, dtype> scores = NewTensor<xpu>(Shape3(seqsize, 1, dim2), d_zero);

  for (int idx = 0; idx < seqsize; idx++) {
    loutput[idx] = 0.0;
  }

  dtype cost = 0.0;
  static int optLabel;
  for (int idx = 0; idx < seqsize; idx++) {
    optLabel = -1;
    for (int i = 0; i < dim2; ++i) {
      if (answers[idx][i] >= 0) {
        if (optLabel < 0 || output[idx][0][i] > output[idx][0][optLabel])
          optLabel = i;
      }
    }

    dtype sum1 = 0.0;
    dtype sum2 = 0.0;
    dtype maxScore = output[idx][0][optLabel];
    for (int i = 0; i < dim2; ++i) {
      scores[idx][0][i] = -1e10;
      if (answers[idx][i] >= 0) {
        scores[idx][0][i] = exp(output[idx][0][i] - maxScore);
        if (answers[idx][i] == 1)
          sum1 += scores[idx][0][i];
        sum2 += scores[idx][0][i];
      }
    }
    cost += (log(sum2) - log(sum1)) / (batchsize * seqsize);
    if (answers[idx][optLabel] == 1)
      eval.correct_label_count++;
    eval.overall_label_count++;

    for (int i = 0; i < dim2; ++i) {
      if (answers[idx][i] >= 0) {
        loutput[idx][0][i] = (scores[idx][0][i] / sum2 - answers[idx][i]) / (batchsize * seqsize);
      }
    }

  }

  FreeSpace(&scores);
  return cost;
}

template<typename xpu>
inline dtype softmax_cost(const vector<Tensor<xpu, 2, dtype> > &output, const vector<vector<int> > &answers) {
  int seqsize = output.size();
  if (answers.size() != seqsize || seqsize == 0) {
    std::cerr << "softmax_cost error: vector size or context size invalid" << std::endl;
  }

  int dim1 = output[0].size(0), dim2 = output[0].size(1);
  int labelsize = answers[0].size();

  if (labelsize != dim2 || dim1 != 1) {
    std::cerr << "softmax_cost error: dim size invalid" << std::endl;
  }

  Tensor<xpu, 3, dtype> scores = NewTensor<xpu>(Shape3(seqsize, 1, dim2), d_zero);

  dtype cost = 0.0;
  static int optLabel;
  for (int idx = 0; idx < seqsize; idx++) {
    optLabel = -1;
    for (int i = 0; i < dim2; ++i) {
      if (answers[idx][i] >= 0) {
        if (optLabel < 0 || output[idx][0][i] > output[idx][0][optLabel])
          optLabel = i;
      }
    }

    dtype sum1 = 0.0;
    dtype sum2 = 0.0;
    dtype maxScore = output[idx][0][optLabel];
    for (int i = 0; i < dim2; ++i) {
      scores[idx][0][i] = -1e10;
      if (answers[idx][i] >= 0) {
        scores[idx][0][i] = exp(output[idx][0][i] - maxScore);
        if (answers[idx][i] == 1)
          sum1 += scores[idx][0][i];
        sum2 += scores[idx][0][i];
      }
    }
    cost += (log(sum2) - log(sum1)) / seqsize;
  }

  FreeSpace(&scores);
  return cost;
}

template<typename xpu>
inline void softmax_predict(const vector<Tensor<xpu, 2, dtype> > &output, vector<int>& results) {
  int seqsize = output.size();
  if (seqsize == 0) {
    std::cerr << "softmax_predict error: vector size or context size invalid" << std::endl;
  }

  int dim1 = output[0].size(0), dim2 = output[0].size(1);
  if (dim1 != 1) {
    std::cerr << "softmax_predict error: dim size invalid" << std::endl;
  }

  results.resize(seqsize);

  static int optLabel;
  for (int idx = 0; idx < seqsize; idx++) {
    optLabel = -1;
    for (int i = 0; i < dim2; ++i) {
      if (optLabel < 0 || output[idx][0][i] > output[idx][0][optLabel])
        optLabel = i;
    }
    results[idx] = optLabel;
  }

}

template<typename xpu>
inline dtype softmax_loss(Tensor<xpu, 3, dtype> output, const vector<vector<int> > &answers, Tensor<xpu, 3, dtype> loutput, Metric & eval, int batchsize = 1) {
  int seqsize = output.size(0);
  if (answers.size() != seqsize || seqsize == 0) {
    std::cerr << "softmax_loss error: vector size or context size invalid" << std::endl;
  }

  int dim1 = output.size(1), dim2 = output.size(2);
  int odim1 = loutput.size(1), odim2 = loutput.size(2);
  int labelsize = answers[0].size();

  if (labelsize != odim2 || dim2 != odim2 || dim1 != 1 || odim1 != 1) {
    std::cerr << "softmax_loss error: dim size invalid" << std::endl;
  }

  Tensor<xpu, 3, dtype> scores = NewTensor<xpu>(Shape3(seqsize, 1, dim2), d_zero);

  loutput = 0.0;
  dtype cost = 0.0;
  static int optLabel;
  for (int idx = 0; idx < seqsize; idx++) {
    optLabel = -1;
    for (int i = 0; i < dim2; ++i) {
      if (answers[idx][i] >= 0) {
        if (optLabel < 0 || output[idx][0][i] > output[idx][0][optLabel])
          optLabel = i;
      }
    }

    dtype sum1 = 0.0;
    dtype sum2 = 0.0;
    dtype maxScore = output[idx][0][optLabel];
    for (int i = 0; i < dim2; ++i) {
      scores[idx][0][i] = -1e10;
      if (answers[idx][i] >= 0) {
        scores[idx][0][i] = exp(output[idx][0][i] - maxScore);
        if (answers[idx][i] == 1)
          sum1 += scores[idx][0][i];
        sum2 += scores[idx][0][i];
      }
    }
    cost += (log(sum2) - log(sum1)) / (batchsize * seqsize);
    if (answers[idx][optLabel] == 1)
      eval.correct_label_count++;
    eval.overall_label_count++;

    for (int i = 0; i < dim2; ++i) {
      if (answers[idx][i] >= 0) {
        loutput[idx][0][i] = (scores[idx][0][i] / sum2 - answers[idx][i]) / (batchsize * seqsize);
      }
    }

  }

  FreeSpace(&scores);
  return cost;
}

template<typename xpu>
inline dtype softmax_cost(Tensor<xpu, 3, dtype> output, const vector<vector<int> > &answers, int batchsize = 1) {
  int seqsize = output.size(0);
  if (answers.size() != seqsize || seqsize == 0) {
    std::cerr << "softmax_cost error: vector size or context size invalid" << std::endl;
  }

  int dim1 = output.size(1), dim2 = output.size(2);
  int labelsize = answers[0].size();

  if (labelsize != dim2 || dim1 != 1) {
    std::cerr << "softmax_cost error: dim size invalid" << std::endl;
  }

  Tensor<xpu, 3, dtype> scores = NewTensor<xpu>(Shape3(seqsize, 1, dim2), d_zero);

  dtype cost = 0.0;
  static int optLabel;
  for (int idx = 0; idx < seqsize; idx++) {
    optLabel = -1;
    for (int i = 0; i < dim2; ++i) {
      if (answers[idx][i] >= 0) {
        if (optLabel < 0 || output[idx][0][i] > output[idx][0][optLabel])
          optLabel = i;
      }
    }

    dtype sum1 = 0.0;
    dtype sum2 = 0.0;
    dtype maxScore = output[idx][0][optLabel];
    for (int i = 0; i < dim2; ++i) {
      scores[idx][0][i] = -1e10;
      if (answers[idx][i] >= 0) {
        scores[idx][0][i] = exp(output[idx][0][i] - maxScore);
        if (answers[idx][i] == 1)
          sum1 += scores[idx][0][i];
        sum2 += scores[idx][0][i];
      }
    }
    cost += (log(sum2) - log(sum1)) / (batchsize * seqsize);
  }

  FreeSpace(&scores);
  return cost;
}

template<typename xpu>
inline void softmax_predict(Tensor<xpu, 3, dtype> output, vector<int>& results) {
  int seqsize = output.size(0);
  if (seqsize == 0) {
    std::cerr << "softmax_predict error: vector size or context size invalid" << std::endl;
  }

  int dim1 = output.size(1), dim2 = output.size(2);
  if (dim1 != 1) {
    std::cerr << "softmax_predict error: dim size invalid" << std::endl;
  }

  results.resize(seqsize);

  static int optLabel;
  for (int idx = 0; idx < seqsize; idx++) {
    optLabel = -1;
    for (int i = 0; i < dim2; ++i) {
      if (optLabel < 0 || output[idx][0][i] > output[idx][0][optLabel])
        optLabel = i;
    }
    results[idx] = optLabel;
  }

}

template<typename xpu>
inline dtype softmax_loss(Tensor<xpu, 2, dtype> output, const vector<int> &answer, Tensor<xpu, 2, dtype> loutput, Metric & eval, int batchsize = 1) {
  int dim1 = output.size(0), dim2 = output.size(1);
  int odim1 = loutput.size(0), odim2 = loutput.size(1);
  int labelsize = answer.size();

  if (labelsize != odim2 || dim2 != odim2 || dim1 != 1 || odim1 != 1) {
    std::cerr << "softmax_loss error: dim size invalid" << std::endl;
  }

  Tensor<xpu, 2, dtype> scores = NewTensor<xpu>(Shape2(1, dim2), d_zero);

  loutput = 0.0;
  dtype cost = 0.0;

  int optLabel = -1;
  for (int i = 0; i < dim2; ++i) {
    if (answer[i] >= 0) {
      if (optLabel < 0 || output[0][i] > output[0][optLabel])
        optLabel = i;
    }
  }

  dtype sum1 = 0.0;
  dtype sum2 = 0.0;
  dtype maxScore = output[0][optLabel];
  for (int i = 0; i < dim2; ++i) {
    scores[0][i] = -1e10;
    if (answer[i] >= 0) {
      scores[0][i] = exp(output[0][i] - maxScore);
      if (answer[i] == 1)
        sum1 += scores[0][i];
      sum2 += scores[0][i];
    }
  }
  cost += (log(sum2) - log(sum1)) / batchsize;
  if (answer[optLabel] == 1)
    eval.correct_label_count++;
  eval.overall_label_count++;

  for (int i = 0; i < dim2; ++i) {
    if (answer[i] >= 0) {
      loutput[0][i] = (scores[0][i] / sum2 - answer[i]) / batchsize;
    }
  }

  FreeSpace(&scores);
  return cost;
}

template<typename xpu>
inline dtype softmax_cost(Tensor<xpu, 2, dtype> output, const vector<int> &answer, int batchsize = 1) {
  int dim1 = output.size(0), dim2 = output.size(1);
  int labelsize = answer.size();

  if (labelsize != dim2 || dim1 != 1) {
    std::cerr << "softmax_cost error: dim size invalid" << std::endl;
  }

  Tensor<xpu, 2, dtype> scores = NewTensor<xpu>(Shape2(1, dim2), d_zero);

  dtype cost = 0.0;

  int optLabel = -1;
  for (int i = 0; i < dim2; ++i) {
    if (answer[i] >= 0) {
      if (optLabel < 0 || output[0][i] > output[0][optLabel])
        optLabel = i;
    }
  }

  dtype sum1 = 0.0;
  dtype sum2 = 0.0;
  dtype maxScore = output[0][optLabel];
  for (int i = 0; i < dim2; ++i) {
    scores[0][i] = -1e10;
    if (answer[i] >= 0) {
      scores[0][i] = exp(output[0][i] - maxScore);
      if (answer[i] == 1)
        sum1 += scores[0][i];
      sum2 += scores[0][i];
    }
  }
  cost += (log(sum2) - log(sum1)) / batchsize;

  FreeSpace(&scores);
  return cost;
}

template<typename xpu>
inline void softmax_predict(Tensor<xpu, 2, dtype> output, int& result) {
  int dim1 = output.size(0), dim2 = output.size(1);
  if (dim1 != 1) {
    std::cerr << "softmax_predict error: dim size invalid" << std::endl;
  }

  int optLabel = -1;
  for (int i = 0; i < dim2; ++i) {
    if (optLabel < 0 || output[0][i] > output[0][optLabel])
      optLabel = i;
  }
  result = optLabel;

}

template<typename xpu>
inline int softmax_predict(Tensor<xpu, 2, dtype> output, vector<dtype>& results) {
  int dim1 = output.size(0), dim2 = output.size(1);
  if (dim1 != 1) {
    std::cerr << "softmax_predict error: dim size invalid" << std::endl;
  }

  int optLabel = -1;
  for (int i = 0; i < dim2; ++i) {
    if (optLabel < 0 || output[0][i] > output[0][optLabel])
      optLabel = i;
  }

  dtype maxScore = output[0][optLabel];
  results.resize(dim2);

  dtype sum = 0.0;
  for (int i = 0; i < dim2; ++i) {
      results[i] = exp(output[0][i] - maxScore);
      sum += results[i];
  }

  for (int i = 0; i < dim2; ++i) {
      results[i] = results[i]/sum;
  }

  return optLabel;

}

#endif
