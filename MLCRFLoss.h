/*
 * MLCRFLoss.h
 *
 *  Created on: Mar 18, 2015
 *      Author: mszhang
 */

#ifndef SRC_MLCRFLoss_H_
#define SRC_MLCRFLoss_H_
#include "tensor.h"
#include "MyLib.h"
#include "Utiltensor.h"

using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;

template<typename xpu>
class MLCRFLoss {

public:

  Tensor<xpu, 2, dtype> _tagBigram;
  Tensor<xpu, 2, dtype> _grad_tagBigram;
  Tensor<xpu, 2, dtype> _eg2_tagBigram;


public:
  MLCRFLoss() {
  }

  inline void initial(int nLabelSize, int seed = 0) {
    dtype bound = sqrt(6.0 / (nLabelSize + nLabelSize + 1));
    //dtype bound = 0.01;

    _tagBigram = NewTensor<xpu>(Shape2(nLabelSize, nLabelSize), d_zero);
    _grad_tagBigram = NewTensor<xpu>(Shape2(nLabelSize, nLabelSize), d_zero);
    _eg2_tagBigram = NewTensor<xpu>(Shape2(nLabelSize, nLabelSize), d_zero);

    random(_tagBigram, -bound, bound, seed);
  }

  inline void initial(Tensor<xpu, 2, dtype> W) {
    static int nLabelSize;
    nLabelSize = W.size(0);

    _tagBigram = NewTensor<xpu>(Shape2(nLabelSize, nLabelSize), d_zero);
    _grad_tagBigram = NewTensor<xpu>(Shape2(nLabelSize, nLabelSize), d_zero);
    _eg2_tagBigram = NewTensor<xpu>(Shape2(nLabelSize, nLabelSize), d_zero);
    Copy(_tagBigram, W);

  }

  inline void release() {
    FreeSpace(&_tagBigram);
    FreeSpace(&_grad_tagBigram);
    FreeSpace(&_eg2_tagBigram);
  }

  virtual ~MLCRFLoss() {
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
    dtype tmp_value = 0.0;
    NRMat<dtype> alpha(seq_size, labelsize);
    NRMat<dtype> alpha_annotated(seq_size, labelsize);
    for (int idx = 0; idx < seq_size; idx++) {
      for (int i = 0; i < labelsize; ++i) {
        // can be changed with probabilities in future work
        if (idx == 0) {
          alpha[idx][i] = output[idx][0][i];
          if (answers[idx][i] == 0) {
            alpha_annotated[idx][i] = minlogvalue;
          } else if (answers[idx][i] == 1) {
            alpha_annotated[idx][i] = output[idx][0][i];
          } else {
            cout << "error label set" << std::endl;
          }
        } else {
          dtype tmp[labelsize];
          for (int j = 0; j < labelsize; ++j) {
            tmp[j] = _tagBigram[j][i] + output[idx][0][i] + alpha[idx - 1][j];
          }
          alpha[idx][i] = logsumexp(tmp, labelsize);

          if (answers[idx][i] == 0) {
            alpha_annotated[idx][i] = minlogvalue;
          } else if (answers[idx][i] == 1) {
            dtype tmp_annoteted[labelsize];
            for (int j = 0; j < labelsize; ++j) {
              if (answers[idx - 1][j] == 1) {
                tmp_annoteted[j] = _tagBigram[j][i] + output[idx][0][i] + alpha_annotated[idx - 1][j];
              } else {
                tmp_annoteted[j] = minlogvalue;
              }
            }
            alpha_annotated[idx][i] = logsumexp(tmp_annoteted, labelsize);
          } else {
            cout << "error label set" << std::endl;
          }
        }
      }
    }

    // backward
    NRMat<dtype> belta(seq_size, labelsize);
    NRMat<dtype> belta_annotated(seq_size, labelsize);

    for (int idx = seq_size - 1; idx >= 0; idx--) {
      for (int i = 0; i < labelsize; ++i) {
        if (idx == seq_size - 1) {
          belta[idx][i] = 0.0;
          if (answers[idx][i] == 0) {
            belta_annotated[idx][i] = minlogvalue;
          } else if (answers[idx][i] == 1) {
            belta_annotated[idx][i] = 0.0;
          } else {
            cout << "error label set" << std::endl;
          }
        } else {
          dtype tmp[labelsize];
          for (int j = 0; j < labelsize; ++j) {
            tmp[j] = _tagBigram[i][j] + output[idx + 1][0][j] + belta[idx + 1][j];
          }
          belta[idx][i] = logsumexp(tmp, labelsize);

          if (answers[idx][i] == 0) {
            belta_annotated[idx][i] = minlogvalue;
          } else if (answers[idx][i] == 1) {
            dtype tmp_annoteted[labelsize];
            for (int j = 0; j < labelsize; ++j) {
              if (answers[idx + 1][j] == 1) {
                tmp_annoteted[j] = _tagBigram[i][j] + output[idx + 1][0][j] + belta_annotated[idx + 1][j];
              } else {
                tmp_annoteted[j] = minlogvalue;
              }
            }
            belta_annotated[idx][i] = logsumexp(tmp_annoteted, labelsize);
          } else {
            cout << "error label set" << std::endl;
          }
        }
      }
    }

    dtype logZ = logsumexp(alpha[seq_size - 1], labelsize);

    dtype logZAnnotated = logsumexp(alpha_annotated[seq_size - 1], labelsize);
    cost += (logZ - logZAnnotated) / batchsize;

    // compute free expectation
    NRMat<dtype> marginalProbXL(seq_size, labelsize);
    NRMat3d<dtype> marginalProbLL(seq_size, labelsize, labelsize);

    for (int idx = 0; idx < seq_size; idx++) {
      dtype sum = 0.0;
      for (int i = 0; i < labelsize; ++i) {
        marginalProbXL[idx][i] = 0.0;
        if (idx == 0) {
          tmp_value = alpha[idx][i] + belta[idx][i] - logZ;
          marginalProbXL[idx][i] = exp(tmp_value);
        } else {
          for (int j = 0; j < labelsize; ++j) {
            tmp_value = alpha[idx - 1][j] + output[idx][0][i] + _tagBigram[j][i] + belta[idx][i] - logZ;
            marginalProbLL[idx][j][i] = exp(tmp_value);
            marginalProbXL[idx][i] += marginalProbLL[idx][j][i];
          }
          tmp_value = alpha[idx][i] + belta[idx][i] - logZ;
          dtype tmpprob = exp(tmp_value);
          if (abs(marginalProbXL[idx][i] - tmpprob) > 1e-20) {
            // System.err.println(String.format("diff: %.18f\t%.18f",
            // marginalProbXL[idx][i], tmpprob));
          }
        }
        sum += marginalProbXL[idx][i];
      }
      //if (abs(sum - 1) > 1e-6)
      //  std::cout << "prob unconstrained sum: " << sum << std::endl;
    }

    // compute constrained expectation
    NRMat<dtype> marginalAnnotatedProbXL(seq_size, labelsize);
    NRMat3d<dtype> marginalAnnotatedProbLL(seq_size, labelsize, labelsize);
    for (int idx = 0; idx < seq_size; idx++) {
      dtype sum = 0;
      for (int i = 0; i < labelsize; ++i) {
        marginalAnnotatedProbXL[idx][i] = 0.0;
        if (idx == 0) {
          if (answers[idx][i] == 1) {
            tmp_value = alpha_annotated[idx][i] + belta_annotated[idx][i] - logZAnnotated;
            marginalAnnotatedProbXL[idx][i] = exp(tmp_value);
          }
        } else {
          for (int j = 0; j < labelsize; ++j) {
            marginalAnnotatedProbLL[idx][j][i] = 0.0;
            if (answers[idx - 1][j] == 1 && answers[idx][i] == 1) {
              tmp_value = alpha_annotated[idx - 1][j] + output[idx][0][i] + _tagBigram[j][i] + belta_annotated[idx][i] - logZAnnotated;
              marginalAnnotatedProbLL[idx][j][i] = exp(tmp_value);
            }
            marginalAnnotatedProbXL[idx][i] += marginalAnnotatedProbLL[idx][j][i];
          }
        }
        sum += marginalAnnotatedProbXL[idx][i];
      }
      //if (abs(sum - 1) > 1e-6)
      //  std::cout << "prob constrained sum: " << sum << std::endl;
    }

    // compute _tagBigram grad
    for (int idx = 1; idx < seq_size; idx++) {
      for (int i = 0; i < labelsize; ++i) {
        for (int j = 0; j < labelsize; ++j) {
          _grad_tagBigram[i][j] += marginalProbLL[idx][i][j] - marginalAnnotatedProbLL[idx][i][j];
        }
      }
    }

    // get delta for each output
    eval.overall_label_count += seq_size;
    for (int idx = 0; idx < seq_size; idx++) {
      dtype predict_best = -1.0;
      int predict_labelid = -1;
      dtype annotated_best = -1.0;
      int annotated_labelid = -1;
      for (int i = 0; i < labelsize; ++i) {
        loutput[idx][0][i] = (marginalProbXL[idx][i] - marginalAnnotatedProbXL[idx][i]) / batchsize;
        if (marginalProbXL[idx][i] > predict_best) {
          predict_best = marginalProbXL[idx][i];
          predict_labelid = i;
        }
        if (marginalAnnotatedProbXL[idx][i] > annotated_best) {
          annotated_best = marginalAnnotatedProbXL[idx][i];
          annotated_labelid = i;
        }
      }

      if (annotated_labelid != -1 && annotated_labelid == predict_labelid)
        eval.correct_label_count++;
      if (annotated_labelid == -1)
        std::cout << "error, please debug" << std::endl;

    }

    return cost;

  }


  inline dtype cost(const vector<Tensor<xpu, 2, dtype> > &output, const vector<vector<int> > &answers) {
    int seq_size = output.size();
    if (answers.size() != seq_size || seq_size == 0) {
      std::cerr << "mlcrf cost error: vector size or context size invalid" << std::endl;
    }

    int dim1 = output[0].size(0), dim2 = output[0].size(1);
    int labelsize = answers[0].size();

    if (labelsize != dim2 || dim1 != 1) {
      std::cerr << "mlcrf cost error: dim size invalid" << std::endl;
    }

    dtype tmp_value = 0.0;
    NRMat<dtype> alpha(seq_size, labelsize);
    NRMat<dtype> alpha_annotated(seq_size, labelsize);
    for (int idx = 0; idx < seq_size; idx++) {
      for (int i = 0; i < labelsize; ++i) {
        // can be changed with probabilities in future work
        if (idx == 0) {
          alpha[idx][i] = output[idx][0][i];
          if (answers[idx][i] == 0) {
            alpha_annotated[idx][i] = minlogvalue;
          } else if (answers[idx][i] == 1) {
            alpha_annotated[idx][i] = output[idx][0][i];
          } else {
            cout << "error label set" << std::endl;
          }
        } else {
          dtype tmp[labelsize];
          for (int j = 0; j < labelsize; ++j) {
            tmp[j] = _tagBigram[j][i] + output[idx][0][i] + alpha[idx - 1][j];
          }
          alpha[idx][i] = logsumexp(tmp, labelsize);

          if (answers[idx][i] == 0) {
            alpha_annotated[idx][i] = minlogvalue;
          } else if (answers[idx][i] == 1) {
            dtype tmp_annoteted[labelsize];
            for (int j = 0; j < labelsize; ++j) {
              if (answers[idx - 1][j] == 1) {
                tmp_annoteted[j] = _tagBigram[j][i] + output[idx][0][i] + alpha_annotated[idx - 1][j];
              } else {
                tmp_annoteted[j] = minlogvalue;
              }
            }
            alpha_annotated[idx][i] = logsumexp(tmp_annoteted, labelsize);
          } else {
            cout << "error label set" << std::endl;
          }
        }
      }
    }

    // backward
    NRMat<dtype> belta(seq_size, labelsize);
    NRMat<dtype> belta_annotated(seq_size, labelsize);

    for (int idx = seq_size - 1; idx >= 0; idx--) {
      for (int i = 0; i < labelsize; ++i) {
        if (idx == seq_size - 1) {
          belta[idx][i] = 0.0;
          if (answers[idx][i] == 0) {
            belta_annotated[idx][i] = minlogvalue;
          } else if (answers[idx][i] == 1) {
            belta_annotated[idx][i] = 0.0;
          } else {
            cout << "error label set" << std::endl;
          }
        } else {
          dtype tmp[labelsize];
          for (int j = 0; j < labelsize; ++j) {
            tmp[j] = _tagBigram[i][j] + output[idx + 1][0][j] + belta[idx + 1][j];
          }
          belta[idx][i] = logsumexp(tmp, labelsize);

          if (answers[idx][i] == 0) {
            belta_annotated[idx][i] = minlogvalue;
          } else if (answers[idx][i] == 1) {
            dtype tmp_annoteted[labelsize];
            for (int j = 0; j < labelsize; ++j) {
              if (answers[idx + 1][j] == 1) {
                tmp_annoteted[j] = _tagBigram[i][j] + output[idx + 1][0][j] + belta_annotated[idx + 1][j];
              } else {
                tmp_annoteted[j] = minlogvalue;
              }
            }
            belta_annotated[idx][i] = logsumexp(tmp_annoteted, labelsize);
          } else {
            cout << "error label set" << std::endl;
          }
        }
      }
    }

    dtype logZ = logsumexp(alpha[seq_size - 1], labelsize);

    dtype logZAnnotated = logsumexp(alpha_annotated[seq_size - 1], labelsize);

    return logZ - logZAnnotated;
  }


  inline void predict(const vector<Tensor<xpu, 2, dtype> > &output, vector<int>& results) {
    int seq_size = output.size();
    if (seq_size == 0) {
      std::cerr << "mlcrf predict error: vector size or context size invalid" << std::endl;
    }

    int dim1 = output[0].size(0), dim2 = output[0].size(1);
    if (dim1 != 1) {
      std::cerr << "mlcrf predict error: dim size invalid" << std::endl;
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
    dtype tmp_value = 0.0;
    NRMat<dtype> alpha(seq_size, labelsize);
    NRMat<dtype> alpha_annotated(seq_size, labelsize);
    for (int idx = 0; idx < seq_size; idx++) {
      for (int i = 0; i < labelsize; ++i) {
        // can be changed with probabilities in future work
        if (idx == 0) {
          alpha[idx][i] = output[idx][0][i];
          if (answers[idx][i] == 0) {
            alpha_annotated[idx][i] = minlogvalue;
          } else if (answers[idx][i] == 1) {
            alpha_annotated[idx][i] = output[idx][0][i];
          } else {
            cout << "error label set" << std::endl;
          }
        } else {
          dtype tmp[labelsize];
          for (int j = 0; j < labelsize; ++j) {
            tmp[j] = _tagBigram[j][i] + output[idx][0][i] + alpha[idx - 1][j];
          }
          alpha[idx][i] = logsumexp(tmp, labelsize);

          if (answers[idx][i] == 0) {
            alpha_annotated[idx][i] = minlogvalue;
          } else if (answers[idx][i] == 1) {
            dtype tmp_annoteted[labelsize];
            for (int j = 0; j < labelsize; ++j) {
              if (answers[idx - 1][j] == 1) {
                tmp_annoteted[j] = _tagBigram[j][i] + output[idx][0][i] + alpha_annotated[idx - 1][j];
              } else {
                tmp_annoteted[j] = minlogvalue;
              }
            }
            alpha_annotated[idx][i] = logsumexp(tmp_annoteted, labelsize);
          } else {
            cout << "error label set" << std::endl;
          }
        }
      }
    }

    // backward
    NRMat<dtype> belta(seq_size, labelsize);
    NRMat<dtype> belta_annotated(seq_size, labelsize);

    for (int idx = seq_size - 1; idx >= 0; idx--) {
      for (int i = 0; i < labelsize; ++i) {
        if (idx == seq_size - 1) {
          belta[idx][i] = 0.0;
          if (answers[idx][i] == 0) {
            belta_annotated[idx][i] = minlogvalue;
          } else if (answers[idx][i] == 1) {
            belta_annotated[idx][i] = 0.0;
          } else {
            cout << "error label set" << std::endl;
          }
        } else {
          dtype tmp[labelsize];
          for (int j = 0; j < labelsize; ++j) {
            tmp[j] = _tagBigram[i][j] + output[idx + 1][0][j] + belta[idx + 1][j];
          }
          belta[idx][i] = logsumexp(tmp, labelsize);

          if (answers[idx][i] == 0) {
            belta_annotated[idx][i] = minlogvalue;
          } else if (answers[idx][i] == 1) {
            dtype tmp_annoteted[labelsize];
            for (int j = 0; j < labelsize; ++j) {
              if (answers[idx + 1][j] == 1) {
                tmp_annoteted[j] = _tagBigram[i][j] + output[idx + 1][0][j] + belta_annotated[idx + 1][j];
              } else {
                tmp_annoteted[j] = minlogvalue;
              }
            }
            belta_annotated[idx][i] = logsumexp(tmp_annoteted, labelsize);
          } else {
            cout << "error label set" << std::endl;
          }
        }
      }
    }

    dtype logZ = logsumexp(alpha[seq_size - 1], labelsize);

    dtype logZAnnotated = logsumexp(alpha_annotated[seq_size - 1], labelsize);
    cost += (logZ - logZAnnotated) / batchsize;

    // compute free expectation
    NRMat<dtype> marginalProbXL(seq_size, labelsize);
    NRMat3d<dtype> marginalProbLL(seq_size, labelsize, labelsize);

    for (int idx = 0; idx < seq_size; idx++) {
      dtype sum = 0.0;
      for (int i = 0; i < labelsize; ++i) {
        marginalProbXL[idx][i] = 0.0;
        if (idx == 0) {
          tmp_value = alpha[idx][i] + belta[idx][i] - logZ;
          marginalProbXL[idx][i] = exp(tmp_value);
        } else {
          for (int j = 0; j < labelsize; ++j) {
            tmp_value = alpha[idx - 1][j] + output[idx][0][i] + _tagBigram[j][i] + belta[idx][i] - logZ;
            marginalProbLL[idx][j][i] = exp(tmp_value);
            marginalProbXL[idx][i] += marginalProbLL[idx][j][i];
          }
          tmp_value = alpha[idx][i] + belta[idx][i] - logZ;
          dtype tmpprob = exp(tmp_value);
          if (abs(marginalProbXL[idx][i] - tmpprob) > 1e-20) {
            // System.err.println(String.format("diff: %.18f\t%.18f",
            // marginalProbXL[idx][i], tmpprob));
          }
        }
        sum += marginalProbXL[idx][i];
      }
      //if (abs(sum - 1) > 1e-6)
      //  std::cout << "prob unconstrained sum: " << sum << std::endl;
    }

    // compute constrained expectation
    NRMat<dtype> marginalAnnotatedProbXL(seq_size, labelsize);
    NRMat3d<dtype> marginalAnnotatedProbLL(seq_size, labelsize, labelsize);
    for (int idx = 0; idx < seq_size; idx++) {
      dtype sum = 0;
      for (int i = 0; i < labelsize; ++i) {
        marginalAnnotatedProbXL[idx][i] = 0.0;
        if (idx == 0) {
          if (answers[idx][i] == 1) {
            tmp_value = alpha_annotated[idx][i] + belta_annotated[idx][i] - logZAnnotated;
            marginalAnnotatedProbXL[idx][i] = exp(tmp_value);
          }
        } else {
          for (int j = 0; j < labelsize; ++j) {
            marginalAnnotatedProbLL[idx][j][i] = 0.0;
            if (answers[idx - 1][j] == 1 && answers[idx][i] == 1) {
              tmp_value = alpha_annotated[idx - 1][j] + output[idx][0][i] + _tagBigram[j][i] + belta_annotated[idx][i] - logZAnnotated;
              marginalAnnotatedProbLL[idx][j][i] = exp(tmp_value);
            }
            marginalAnnotatedProbXL[idx][i] += marginalAnnotatedProbLL[idx][j][i];
          }
        }
        sum += marginalAnnotatedProbXL[idx][i];
      }
      //if (abs(sum - 1) > 1e-6)
      //  std::cout << "prob constrained sum: " << sum << std::endl;
    }

    // compute _tagBigram grad
    for (int idx = 1; idx < seq_size; idx++) {
      for (int i = 0; i < labelsize; ++i) {
        for (int j = 0; j < labelsize; ++j) {
          _grad_tagBigram[i][j] += marginalProbLL[idx][i][j] - marginalAnnotatedProbLL[idx][i][j];
        }
      }
    }

    // get delta for each output
    eval.overall_label_count += seq_size;
    for (int idx = 0; idx < seq_size; idx++) {
      dtype predict_best = -1.0;
      int predict_labelid = -1;
      dtype annotated_best = -1.0;
      int annotated_labelid = -1;
      for (int i = 0; i < labelsize; ++i) {
        loutput[idx][0][i] = (marginalProbXL[idx][i] - marginalAnnotatedProbXL[idx][i]) / batchsize;
        if (marginalProbXL[idx][i] > predict_best) {
          predict_best = marginalProbXL[idx][i];
          predict_labelid = i;
        }
        if (marginalAnnotatedProbXL[idx][i] > annotated_best) {
          annotated_best = marginalAnnotatedProbXL[idx][i];
          annotated_labelid = i;
        }
      }

      if (annotated_labelid != -1 && annotated_labelid == predict_labelid)
        eval.correct_label_count++;
      if (annotated_labelid == -1)
        std::cout << "error, please debug" << std::endl;

    }

    return cost;

  }


  inline dtype cost(Tensor<xpu, 3, dtype> output, const vector<vector<int> > &answers) {
    int seq_size = output.size(0);
    if (answers.size() != seq_size || seq_size == 0) {
      std::cerr << "mlcrf cost error: vector size or context size invalid" << std::endl;
    }

    int dim1 = output.size(1), dim2 = output.size(2);
    int labelsize = answers[0].size();

    if (labelsize != dim2 || dim1 != 1) {
      std::cerr << "mlcrf cost error: dim size invalid" << std::endl;
    }

    dtype tmp_value = 0.0;
    NRMat<dtype> alpha(seq_size, labelsize);
    NRMat<dtype> alpha_annotated(seq_size, labelsize);
    for (int idx = 0; idx < seq_size; idx++) {
      for (int i = 0; i < labelsize; ++i) {
        // can be changed with probabilities in future work
        if (idx == 0) {
          alpha[idx][i] = output[idx][0][i];
          if (answers[idx][i] == 0) {
            alpha_annotated[idx][i] = minlogvalue;
          } else if (answers[idx][i] == 1) {
            alpha_annotated[idx][i] = output[idx][0][i];
          } else {
            cout << "error label set" << std::endl;
          }
        } else {
          dtype tmp[labelsize];
          for (int j = 0; j < labelsize; ++j) {
            tmp[j] = _tagBigram[j][i] + output[idx][0][i] + alpha[idx - 1][j];
          }
          alpha[idx][i] = logsumexp(tmp, labelsize);

          if (answers[idx][i] == 0) {
            alpha_annotated[idx][i] = minlogvalue;
          } else if (answers[idx][i] == 1) {
            dtype tmp_annoteted[labelsize];
            for (int j = 0; j < labelsize; ++j) {
              if (answers[idx - 1][j] == 1) {
                tmp_annoteted[j] = _tagBigram[j][i] + output[idx][0][i] + alpha_annotated[idx - 1][j];
              } else {
                tmp_annoteted[j] = minlogvalue;
              }
            }
            alpha_annotated[idx][i] = logsumexp(tmp_annoteted, labelsize);
          } else {
            cout << "error label set" << std::endl;
          }
        }
      }
    }

    // backward
    NRMat<dtype> belta(seq_size, labelsize);
    NRMat<dtype> belta_annotated(seq_size, labelsize);

    for (int idx = seq_size - 1; idx >= 0; idx--) {
      for (int i = 0; i < labelsize; ++i) {
        if (idx == seq_size - 1) {
          belta[idx][i] = 0.0;
          if (answers[idx][i] == 0) {
            belta_annotated[idx][i] = minlogvalue;
          } else if (answers[idx][i] == 1) {
            belta_annotated[idx][i] = 0.0;
          } else {
            cout << "error label set" << std::endl;
          }
        } else {
          dtype tmp[labelsize];
          for (int j = 0; j < labelsize; ++j) {
            tmp[j] = _tagBigram[i][j] + output[idx + 1][0][j] + belta[idx + 1][j];
          }
          belta[idx][i] = logsumexp(tmp, labelsize);

          if (answers[idx][i] == 0) {
            belta_annotated[idx][i] = minlogvalue;
          } else if (answers[idx][i] == 1) {
            dtype tmp_annoteted[labelsize];
            for (int j = 0; j < labelsize; ++j) {
              if (answers[idx + 1][j] == 1) {
                tmp_annoteted[j] = _tagBigram[i][j] + output[idx + 1][0][j] + belta_annotated[idx + 1][j];
              } else {
                tmp_annoteted[j] = minlogvalue;
              }
            }
            belta_annotated[idx][i] = logsumexp(tmp_annoteted, labelsize);
          } else {
            cout << "error label set" << std::endl;
          }
        }
      }
    }

    dtype logZ = logsumexp(alpha[seq_size - 1], labelsize);

    dtype logZAnnotated = logsumexp(alpha_annotated[seq_size - 1], labelsize);

    return logZ - logZAnnotated;
  }


  inline void predict(Tensor<xpu, 3, dtype> output, vector<int>& results) {
    int seq_size = output.size(0);
    if (seq_size == 0) {
      std::cerr << "mlcrf predict error: vector size or context size invalid" << std::endl;
    }

    int dim1 = output.size(1), dim2 = output.size(2);
    if (dim1 != 1) {
      std::cerr << "mlcrf predict error: dim size invalid" << std::endl;
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
  }

  void loadModel(LStream &inf) {
    LoadBinary(inf, &_tagBigram, false);
    LoadBinary(inf, &_grad_tagBigram, false);
    LoadBinary(inf, &_eg2_tagBigram, false);
  }

};

#endif /* SRC_MLCRFLoss_H_ */
