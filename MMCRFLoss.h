/*
 * MMCRFLoss.h
 *
 *  Created on: Mar 18, 2015
 *      Author: mszhang
 */

#ifndef SRC_MMCRFLoss_H_
#define SRC_MMCRFLoss_H_
#include "tensor.h"
#include "MyLib.h"
#include "Utiltensor.h"

using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;

template<typename xpu>
class MMCRFLoss {

public:

  Tensor<xpu, 2, dtype> _tagBigram;
  Tensor<xpu, 2, dtype> _grad_tagBigram;
  Tensor<xpu, 2, dtype> _eg2_tagBigram;

  dtype _delta;


public:
  MMCRFLoss() {
  }

  inline void initial(int nLabelSize, int seed = 0) {
    dtype bound = sqrt(6.0 / (nLabelSize + nLabelSize + 1));
    //dtype bound = 0.01;

    _tagBigram = NewTensor<xpu>(Shape2(nLabelSize, nLabelSize), d_zero);
    _grad_tagBigram = NewTensor<xpu>(Shape2(nLabelSize, nLabelSize), d_zero);
    _eg2_tagBigram = NewTensor<xpu>(Shape2(nLabelSize, nLabelSize), d_zero);

    random(_tagBigram, -bound, bound, seed);

    _delta = 0.2;
  }


  inline void initial(int nLabelSize, dtype delta, int seed = 0) {
    dtype bound = sqrt(6.0 / (nLabelSize + nLabelSize + 1));
    //dtype bound = 0.01;

    _tagBigram = NewTensor<xpu>(Shape2(nLabelSize, nLabelSize), d_zero);
    _grad_tagBigram = NewTensor<xpu>(Shape2(nLabelSize, nLabelSize), d_zero);
    _eg2_tagBigram = NewTensor<xpu>(Shape2(nLabelSize, nLabelSize), d_zero);

    random(_tagBigram, -bound, bound, seed);

    _delta = delta;
  }

  inline void initial(Tensor<xpu, 2, dtype> W, dtype delta = 0.2) {
    static int nLabelSize;
    nLabelSize = W.size(0);

    _tagBigram = NewTensor<xpu>(Shape2(nLabelSize, nLabelSize), d_zero);
    _grad_tagBigram = NewTensor<xpu>(Shape2(nLabelSize, nLabelSize), d_zero);
    _eg2_tagBigram = NewTensor<xpu>(Shape2(nLabelSize, nLabelSize), d_zero);
    Copy(_tagBigram, W);

    _delta = delta;

  }

  inline void release() {
    FreeSpace(&_tagBigram);
    FreeSpace(&_grad_tagBigram);
    FreeSpace(&_eg2_tagBigram);
  }

  virtual ~MMCRFLoss() {
    // TODO Auto-generated destructor stub
  }

  inline dtype squarenormAll() {
    dtype result = squarenorm(_grad_tagBigram);

    return result;
  }

  inline void scaleGrad(dtype scale) {
    _grad_tagBigram = _grad_tagBigram * scale;
  }

public:

  inline dtype loss(const vector<Tensor<xpu, 2, dtype> > &output, const vector<vector<int> > &answers, vector<Tensor<xpu, 2, dtype> > &loutput,
      Metric & eval, int batchsize = 1) {
    int seq_size = output.size();
    if (answers.size() != seq_size || seq_size == 0) {
      std::cerr << "mlcrf_loss error: vector size or context size invalid" << std::endl;
    }

    int dim1 = output[0].size(0), dim2 = output[0].size(1);
    int odim1 = loutput[0].size(0), odim2 = loutput[0].size(1);
    int labelsize = answers[0].size();
    if (labelsize != odim2 || dim2 != odim2 || dim1 != 1 || odim1 != 1) {
      std::cerr << "mlcrf_loss error: dim size invalid" << std::endl;
    }

    dtype cost = 0.0;
    // get delta for each output
    // viterbi algorithm
    NRVec<int> goldlabels(seq_size);
    goldlabels = -1;
    for (int idx = 0; idx < seq_size; idx++) {
      for (int i = 0; i < labelsize; ++i) {
        if (answers[idx][i] == 1) {
          goldlabels[idx] = i;
        }
      }
    }

    NRMat<dtype> maxscores(seq_size, labelsize);
    NRMat<int> maxlastlabels(seq_size, labelsize);
    dtype goldScore = 0.0;
    for (int idx = 0; idx < seq_size; idx++) {
      if (idx == 0)
        goldScore = output[idx][0][goldlabels[idx]];
      else
        goldScore += output[idx][0][goldlabels[idx]] + _tagBigram[goldlabels[idx - 1]][goldlabels[idx]];

      for (int i = 0; i < labelsize; ++i) {
        // can be changed with probabilities in future work
        if (idx == 0) {
          maxscores[idx][i] = output[idx][0][i];
          if (goldlabels[idx] != i)
            maxscores[idx][i] = maxscores[idx][i] + _delta;
          maxlastlabels[idx][i] = -1;
        } else {
          int maxlastlabel = 0;
          dtype maxscore = _tagBigram[0][i] + output[idx][0][i] + maxscores[idx - 1][0];
          for (int j = 1; j < labelsize; ++j) {
            dtype curscore = _tagBigram[j][i] + output[idx][0][i] + maxscores[idx - 1][j];
            if (curscore > maxscore) {
              maxlastlabel = j;
              maxscore = curscore;
            }
          }
          maxscores[idx][i] = maxscore;
          if (goldlabels[idx] != i)
            maxscores[idx][i] = maxscores[idx][i] + _delta;
          maxlastlabels[idx][i] = maxlastlabel;

        }
      }
    }

    NRVec<int> optLabels(seq_size);
    optLabels = 0;
    dtype maxScore = maxscores[seq_size - 1][0];
    for (int i = 1; i < labelsize; ++i) {
      if (maxscores[seq_size - 1][i] > maxScore) {
        maxScore = maxscores[seq_size - 1][i];
        optLabels[seq_size - 1] = i;
      }
    }

    for (int idx = seq_size - 2; idx >= 0; idx--) {
      optLabels[idx] = maxlastlabels[idx + 1][optLabels[idx + 1]];
    }

    bool bcorrect = true;
    for (int idx = 0; idx < seq_size; idx++) {
      if (goldlabels[idx] == -1)
        continue;
      eval.overall_label_count++;
      if (optLabels[idx] == goldlabels[idx]) {
        eval.correct_label_count++;
      } else {
        bcorrect = false;
      }
    }

    dtype curcost = bcorrect ? 0.0 : 1.0;
    //dtype curcost = maxScore - goldScore;
    curcost = curcost / batchsize;

    for (int idx = 0; idx < seq_size; idx++) {
      if (goldlabels[idx] == -1)
        continue;
      if (optLabels[idx] != goldlabels[idx]) {
        loutput[idx][0][optLabels[idx]] = curcost;
        loutput[idx][0][goldlabels[idx]] = -curcost;
        cost += curcost;
      }
      if (idx > 0 && goldlabels[idx - 1] >= 0) {
        _grad_tagBigram[optLabels[idx - 1]][optLabels[idx]] += curcost;
        _grad_tagBigram[goldlabels[idx - 1]][goldlabels[idx]] -= curcost;
      }
    }

    return cost;

  }


  inline dtype cost(const vector<Tensor<xpu, 2, dtype> > &output, const vector<vector<int> > &answers) {
    int seq_size = output.size();
    if (answers.size() != seq_size || seq_size == 0) {
      std::cerr << "softmax_cost error: vector size or context size invalid" << std::endl;
    }

    int dim1 = output[0].size(0), dim2 = output[0].size(1);
    int labelsize = answers[0].size();

    if (labelsize != dim2 || dim1 != 1) {
      std::cerr << "softmax_cost error: dim size invalid" << std::endl;
    }

    // get delta for each output
    // viterbi algorithm
    NRVec<int> goldlabels(seq_size);
    goldlabels = -1;
    for (int idx = 0; idx < seq_size; idx++) {
      for (int i = 0; i < labelsize; ++i) {
        if (answers[idx][i] == 1) {
          goldlabels[idx] = i;
        }
      }
    }

    NRMat<dtype> maxscores(seq_size, labelsize);
    NRMat<int> maxlastlabels(seq_size, labelsize);
    dtype goldScore = 0.0;
    for (int idx = 0; idx < seq_size; idx++) {
      if (idx == 0)
        goldScore = output[idx][0][goldlabels[idx]];
      else
        goldScore += output[idx][0][goldlabels[idx]] + _tagBigram[goldlabels[idx - 1]][goldlabels[idx]];
      dtype delta = 1.0;
      for (int i = 0; i < labelsize; ++i) {
        // can be changed with probabilities in future work
        if (idx == 0) {
          maxscores[idx][i] = output[idx][0][i];
          if (goldlabels[idx] != i)
            maxscores[idx][i] = maxscores[idx][i] + delta;
          maxlastlabels[idx][i] = -1;
        } else {
          int maxlastlabel = 0;
          dtype maxscore = _tagBigram[0][i] + output[idx][0][i] + maxscores[idx - 1][0];
          for (int j = 1; j < labelsize; ++j) {
            dtype curscore = _tagBigram[j][i] + output[idx][0][i] + maxscores[idx - 1][j];
            if (curscore > maxscore) {
              maxlastlabel = j;
              maxscore = curscore;
            }
          }
          maxscores[idx][i] = maxscore;
          if (goldlabels[idx] != i)
            maxscores[idx][i] = maxscores[idx][i] + delta;
          maxlastlabels[idx][i] = maxlastlabel;

        }
      }
    }

    NRVec<int> optLabels(seq_size);
    optLabels = 0;
    dtype maxScore = maxscores[seq_size - 1][0];
    for (int i = 1; i < labelsize; ++i) {
      if (maxscores[seq_size - 1][i] > maxScore) {
        maxScore = maxscores[seq_size - 1][i];
        optLabels[seq_size - 1] = i;
      }
    }

    for (int idx = seq_size - 2; idx >= 0; idx--) {
      optLabels[idx] = maxlastlabels[idx + 1][optLabels[idx + 1]];
    }

    bool bcorrect = true;
    for (int idx = 0; idx < seq_size; idx++) {
      if (goldlabels[idx] == -1)
        continue;
      if (optLabels[idx] == goldlabels[idx]) {
      } else {
        bcorrect = false;
      }
    }

    dtype cost = bcorrect ? 0.0 : 1.0;

    return cost;
  }


  inline void predict(const vector<Tensor<xpu, 2, dtype> > &output, vector<int>& results) {
    int seq_size = output.size();
    if (seq_size == 0) {
      std::cerr << "softmax_predict error: vector size or context size invalid" << std::endl;
    }

    int dim1 = output[0].size(0), dim2 = output[0].size(1);
    if (dim1 != 1) {
      std::cerr << "softmax_predict error: dim size invalid" << std::endl;
    }

    int labelsize = _tagBigram.size(0);
    // decode algorithm
    // viterbi algorithm
    NRMat<dtype> maxscores(seq_size, labelsize);
    NRMat<int> maxlastlabels(seq_size, labelsize);

    for (int idx = 0; idx < seq_size; idx++) {
      for (int i = 0; i < labelsize; ++i) {
        // can be changed with probabilities in future work
        if (idx == 0) {
          maxscores[idx][i] = output[idx][0][i];
          maxlastlabels[idx][i] = -1;
        } else {
          int maxlastlabel = 0;
          dtype maxscore = _tagBigram[0][i] + output[idx][0][i]
              + maxscores[idx - 1][0];
          for (int j = 1; j < labelsize; ++j) {
            dtype curscore = _tagBigram[j][i] + output[idx][0][i]
                + maxscores[idx - 1][j];
            if (curscore > maxscore) {
              maxlastlabel = j;
              maxscore = curscore;
            }
          }
          maxscores[idx][i] = maxscore;
          maxlastlabels[idx][i] = maxlastlabel;
        }
      }
    }

    results.resize(seq_size);
    dtype maxFinalScore = maxscores[seq_size - 1][0];
    results[seq_size - 1] = 0;
    for (int i = 1; i < labelsize; ++i) {
      if (maxscores[seq_size - 1][i] > maxFinalScore) {
        maxFinalScore = maxscores[seq_size - 1][i];
        results[seq_size - 1] = i;
      }
    }

    for (int idx = seq_size - 2; idx >= 0; idx--) {
      results[idx] = maxlastlabels[idx + 1][results[idx + 1]];
    }

  }


  inline dtype loss(Tensor<xpu, 3, dtype> output, const vector<vector<int> > &answers, Tensor<xpu, 3, dtype> loutput,
      Metric & eval, int batchsize = 1) {
    int seq_size = output.size(0);
    if (answers.size() != seq_size || seq_size == 0) {
      std::cerr << "mlcrf_loss error: vector size or context size invalid" << std::endl;
    }

    int dim1 = output.size(1), dim2 = output.size(2);
    int odim1 = loutput.size(1), odim2 = loutput.size(2);
    int labelsize = answers[0].size();
    if (labelsize != odim2 || dim2 != odim2 || dim1 != 1 || odim1 != 1) {
      std::cerr << "mlcrf_loss error: dim size invalid" << std::endl;
    }

    dtype cost = 0.0;
    // get delta for each output
    // viterbi algorithm
    NRVec<int> goldlabels(seq_size);
    goldlabels = -1;
    for (int idx = 0; idx < seq_size; idx++) {
      for (int i = 0; i < labelsize; ++i) {
        if (answers[idx][i] == 1) {
          goldlabels[idx] = i;
        }
      }
    }

    NRMat<dtype> maxscores(seq_size, labelsize);
    NRMat<int> maxlastlabels(seq_size, labelsize);
    dtype goldScore = 0.0;
    for (int idx = 0; idx < seq_size; idx++) {
      if (idx == 0)
        goldScore = output[idx][0][goldlabels[idx]];
      else
        goldScore += output[idx][0][goldlabels[idx]] + _tagBigram[goldlabels[idx - 1]][goldlabels[idx]];

      for (int i = 0; i < labelsize; ++i) {
        // can be changed with probabilities in future work
        if (idx == 0) {
          maxscores[idx][i] = output[idx][0][i];
          if (goldlabels[idx] != i)
            maxscores[idx][i] = maxscores[idx][i] + _delta;
          maxlastlabels[idx][i] = -1;
        } else {
          int maxlastlabel = 0;
          dtype maxscore = _tagBigram[0][i] + output[idx][0][i] + maxscores[idx - 1][0];
          for (int j = 1; j < labelsize; ++j) {
            dtype curscore = _tagBigram[j][i] + output[idx][0][i] + maxscores[idx - 1][j];
            if (curscore > maxscore) {
              maxlastlabel = j;
              maxscore = curscore;
            }
          }
          maxscores[idx][i] = maxscore;
          if (goldlabels[idx] != i)
            maxscores[idx][i] = maxscores[idx][i] + _delta;
          maxlastlabels[idx][i] = maxlastlabel;

        }
      }
    }

    NRVec<int> optLabels(seq_size);
    optLabels = 0;
    dtype maxScore = maxscores[seq_size - 1][0];
    for (int i = 1; i < labelsize; ++i) {
      if (maxscores[seq_size - 1][i] > maxScore) {
        maxScore = maxscores[seq_size - 1][i];
        optLabels[seq_size - 1] = i;
      }
    }

    for (int idx = seq_size - 2; idx >= 0; idx--) {
      optLabels[idx] = maxlastlabels[idx + 1][optLabels[idx + 1]];
    }

    bool bcorrect = true;
    for (int idx = 0; idx < seq_size; idx++) {
      if (goldlabels[idx] == -1)
        continue;
      eval.overall_label_count++;
      if (optLabels[idx] == goldlabels[idx]) {
        eval.correct_label_count++;
      } else {
        bcorrect = false;
      }
    }

    dtype curcost = bcorrect ? 0.0 : 1.0;
    //dtype curcost = maxScore - goldScore;
    curcost = curcost / batchsize;

    for (int idx = 0; idx < seq_size; idx++) {
      if (goldlabels[idx] == -1)
        continue;
      if (optLabels[idx] != goldlabels[idx]) {
        loutput[idx][0][optLabels[idx]] = curcost;
        loutput[idx][0][goldlabels[idx]] = -curcost;
        cost += curcost;
      }
      if (idx > 0 && goldlabels[idx - 1] >= 0) {
        _grad_tagBigram[optLabels[idx - 1]][optLabels[idx]] += curcost;
        _grad_tagBigram[goldlabels[idx - 1]][goldlabels[idx]] -= curcost;
      }
    }

    return cost;

  }

  inline dtype cost(Tensor<xpu, 3, dtype> output, const vector<vector<int> > &answers) {
    int seq_size = output.size(0);
    if (answers.size() != seq_size || seq_size == 0) {
      std::cerr << "softmax_cost error: vector size or context size invalid" << std::endl;
    }

    int dim1 = output.size(1), dim2 = output.size(2);
    int labelsize = answers[0].size();

    if (labelsize != dim2 || dim1 != 1) {
      std::cerr << "softmax_cost error: dim size invalid" << std::endl;
    }

    // get delta for each output
    // viterbi algorithm
    NRVec<int> goldlabels(seq_size);
    goldlabels = -1;
    for (int idx = 0; idx < seq_size; idx++) {
      for (int i = 0; i < labelsize; ++i) {
        if (answers[idx][i] == 1) {
          goldlabels[idx] = i;
        }
      }
    }

    NRMat<dtype> maxscores(seq_size, labelsize);
    NRMat<int> maxlastlabels(seq_size, labelsize);
    dtype goldScore = 0.0;
    for (int idx = 0; idx < seq_size; idx++) {
      if (idx == 0)
        goldScore = output[idx][0][goldlabels[idx]];
      else
        goldScore += output[idx][0][goldlabels[idx]] + _tagBigram[goldlabels[idx - 1]][goldlabels[idx]];
      dtype delta = 1.0;
      for (int i = 0; i < labelsize; ++i) {
        // can be changed with probabilities in future work
        if (idx == 0) {
          maxscores[idx][i] = output[idx][0][i];
          if (goldlabels[idx] != i)
            maxscores[idx][i] = maxscores[idx][i] + delta;
          maxlastlabels[idx][i] = -1;
        } else {
          int maxlastlabel = 0;
          dtype maxscore = _tagBigram[0][i] + output[idx][0][i] + maxscores[idx - 1][0];
          for (int j = 1; j < labelsize; ++j) {
            dtype curscore = _tagBigram[j][i] + output[idx][0][i] + maxscores[idx - 1][j];
            if (curscore > maxscore) {
              maxlastlabel = j;
              maxscore = curscore;
            }
          }
          maxscores[idx][i] = maxscore;
          if (goldlabels[idx] != i)
            maxscores[idx][i] = maxscores[idx][i] + delta;
          maxlastlabels[idx][i] = maxlastlabel;

        }
      }
    }

    NRVec<int> optLabels(seq_size);
    optLabels = 0;
    dtype maxScore = maxscores[seq_size - 1][0];
    for (int i = 1; i < labelsize; ++i) {
      if (maxscores[seq_size - 1][i] > maxScore) {
        maxScore = maxscores[seq_size - 1][i];
        optLabels[seq_size - 1] = i;
      }
    }

    for (int idx = seq_size - 2; idx >= 0; idx--) {
      optLabels[idx] = maxlastlabels[idx + 1][optLabels[idx + 1]];
    }

    bool bcorrect = true;
    for (int idx = 0; idx < seq_size; idx++) {
      if (goldlabels[idx] == -1)
        continue;
      if (optLabels[idx] == goldlabels[idx]) {
      } else {
        bcorrect = false;
      }
    }

    dtype cost = bcorrect ? 0.0 : 1.0;

    return cost;
  }


  inline void predict(Tensor<xpu, 3, dtype> output, vector<int>& results) {
    int seq_size = output.size(0);
    if (seq_size == 0) {
      std::cerr << "softmax_predict error: vector size or context size invalid" << std::endl;
    }

    int dim1 = output.size(1), dim2 = output.size(2);
    if (dim1 != 1) {
      std::cerr << "softmax_predict error: dim size invalid" << std::endl;
    }

    int labelsize = _tagBigram.size(0);
    // decode algorithm
    // viterbi algorithm
    NRMat<dtype> maxscores(seq_size, labelsize);
    NRMat<int> maxlastlabels(seq_size, labelsize);

    for (int idx = 0; idx < seq_size; idx++) {
      for (int i = 0; i < labelsize; ++i) {
        // can be changed with probabilities in future work
        if (idx == 0) {
          maxscores[idx][i] = output[idx][0][i];
          maxlastlabels[idx][i] = -1;
        } else {
          int maxlastlabel = 0;
          dtype maxscore = _tagBigram[0][i] + output[idx][0][i]
              + maxscores[idx - 1][0];
          for (int j = 1; j < labelsize; ++j) {
            dtype curscore = _tagBigram[j][i] + output[idx][0][i]
                + maxscores[idx - 1][j];
            if (curscore > maxscore) {
              maxlastlabel = j;
              maxscore = curscore;
            }
          }
          maxscores[idx][i] = maxscore;
          maxlastlabels[idx][i] = maxlastlabel;
        }
      }
    }

    results.resize(seq_size);
    dtype maxFinalScore = maxscores[seq_size - 1][0];
    results[seq_size - 1] = 0;
    for (int i = 1; i < labelsize; ++i) {
      if (maxscores[seq_size - 1][i] > maxFinalScore) {
        maxFinalScore = maxscores[seq_size - 1][i];
        results[seq_size - 1] = i;
      }
    }

    for (int idx = seq_size - 2; idx >= 0; idx--) {
      results[idx] = maxlastlabels[idx + 1][results[idx + 1]];
    }

  }


  inline void randomprint(int num) {
    static int nOSize, nISize;
    nOSize = _tagBigram.size(0);
    nISize = _tagBigram.size(1);
    int count = 0;
    while (count < num) {
      int idx = rand() % nOSize;
      int idy = rand() % nISize;

      std::cout << "_tagBigram[" << idx << "," << idy << "]=" << _tagBigram[idx][idy] << " ";

      count++;
    }

    std::cout << std::endl;
  }

  inline void updateAdaGrad(dtype regularizationWeight, dtype adaAlpha, dtype adaEps) {
    _grad_tagBigram = _grad_tagBigram + _tagBigram * regularizationWeight;
    _eg2_tagBigram = _eg2_tagBigram + _grad_tagBigram * _grad_tagBigram;
    _tagBigram = _tagBigram - _grad_tagBigram * adaAlpha / F<nl_sqrt>(_eg2_tagBigram + adaEps);


    clearGrad();
  }

  inline void clearGrad() {
    _grad_tagBigram = 0;
  }

  void writeModel(LStream &outf) {
    SaveBinary(outf, _tagBigram);
    SaveBinary(outf, _grad_tagBigram);
    SaveBinary(outf, _eg2_tagBigram);
    WriteBinary(outf, _delta);
  }

  void loadModel(LStream &inf) {
    LoadBinary(inf, &_tagBigram, false);
    LoadBinary(inf, &_grad_tagBigram, false);
    LoadBinary(inf, &_eg2_tagBigram, false);
    ReadBinary(inf, _delta);
  }
  
};

#endif /* SRC_MMCRFLoss_H_ */
