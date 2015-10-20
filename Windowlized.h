#ifndef WINDOWLIZED
#define WINDOWLIZED

#include "tensor.h"
#include "MyLib.h"


using namespace std;
using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;


template<typename xpu>
inline void windowlized(const vector<Tensor<xpu, 2, dtype> > &wi, vector<Tensor<xpu, 2, dtype> > &wo, int context)
{
  int seqsize = wo.size();
  if (wi.size() != seqsize || seqsize == 0 || context < 0) {
    std::cerr << "windowlized error: vector size or context size invalid" << std::endl;
  }

  int dim1 = wi[0].size(0), dim2 = wi[0].size(1);
  int odim1 = wo[0].size(0), odim2 = wo[0].size(1);
  int computeddim2 = (2 * context + 1) * dim2;
  if(computeddim2 != odim2 || dim1 != 1 || odim1 != 1){
    std::cerr << "windowlized error: dim size invalid" << std::endl;
  }

  static int offset;
  for (int idx = 0; idx < seqsize; idx++) {
    wo[idx] = 0.0;
    offset = 0;
    for (int idp = idx - context; idp <= idx + context; idp++) {
      if (idp < 0 || idp >= seqsize) {
        offset += dim2;
      } else {
        for (int idy = 0; idy < dim2; idy++) {
          wo[idx][0][offset] = wi[idp][0][idy];
          offset++;
        }
      }
    }
    assert(offset == odim2);
  }

}


template<typename xpu>
inline void windowlized(Tensor<xpu, 3, dtype> wi, Tensor<xpu, 3, dtype> wo, int context)
{
  int seqsize = wo.size(0);
  if (wi.size(0) != seqsize || seqsize == 0 || context < 0) {
    std::cerr << "windowlized error: vector size or context size invalid" << std::endl;
  }

  int dim1 = wi.size(1), dim2 = wi.size(2);
  int odim1 = wo.size(1), odim2 = wo.size(2);
  int computeddim2 = (2 * context + 1) * dim2;
  if(computeddim2 != odim2 || dim1 != 1 || odim1 != 1){
    std::cerr << "windowlized error: dim size invalid" << std::endl;
  }

  wo = 0.0;
  static int offset;
  for (int idx = 0; idx < seqsize; idx++) {
    offset = 0;
    for (int idp = idx - context; idp <= idx + context; idp++) {
      if (idp < 0 || idp >= seqsize) {
        offset += dim2;
      } else {
        for (int idy = 0; idy < dim2; idy++) {
          wo[idx][0][offset] = wi[idp][0][idy];
          offset++;
        }
      }
    }
    assert(offset == odim2);
  }

}


template<typename xpu>
inline void windowlized_backward(vector<Tensor<xpu, 2, dtype> > &lwi, const vector<Tensor<xpu, 2, dtype> > &lwo, int context, bool bclear = false)
{
  int seqsize = lwo.size();
  if (lwi.size() != seqsize || seqsize == 0 || context < 0) {
    std::cerr << "windowlized error: vector size or context size invalid" << std::endl;
  }

  int dim1 = lwi[0].size(0), dim2 = lwi[0].size(1);
  int odim1 = lwo[0].size(0), odim2 = lwo[0].size(1);
  int computeddim2 = (2 * context + 1) * dim2;
  if(computeddim2 != odim2 || dim1 != 1 || odim1 != 1){
    std::cerr << "windowlized error: dim size invalid" << std::endl;
  }

  if(bclear){
    for (int idx = 0; idx < seqsize; idx++) {
      lwi[idx] = 0.0;
    }
  }
  static int offset;
  for (int idx = 0; idx < seqsize; idx++) {
    offset = 0;
    for (int idp = idx - context; idp <= idx + context; idp++) {
      if (idp < 0 || idp >= seqsize) {
        offset += dim2;
      } else {
        for (int idy = 0; idy < dim2; idy++) {
          lwi[idp][0][idy] += lwo[idx][0][offset];
          offset++;
        }
      }
    }
    assert(offset == odim2);
  }

}


template<typename xpu>
inline void windowlized_backward(Tensor<xpu, 3, dtype> lwi, Tensor<xpu, 3, dtype> lwo, int context, bool bclear = false)
{
  int seqsize = lwo.size(0);
  if (lwi.size(0) != seqsize || seqsize == 0 || context < 0) {
    std::cerr << "windowlized error: vector size or context size invalid" << std::endl;
  }

  int dim1 = lwi.size(1), dim2 = lwi.size(2);
  int odim1 = lwo.size(1), odim2 = lwo.size(2);
  int computeddim2 = (2 * context + 1) * dim2;
  if(computeddim2 != odim2 || dim1 != 1 || odim1 != 1){
    std::cerr << "windowlized error: dim size invalid" << std::endl;
  }

  if(bclear) lwi = 0.0;
  static int offset;
  for (int idx = 0; idx < seqsize; idx++) {
    offset = 0;
    for (int idp = idx - context; idp <= idx + context; idp++) {
      if (idp < 0 || idp >= seqsize) {
        offset += dim2;
      } else {
        for (int idy = 0; idy < dim2; idy++) {
          lwi[idp][0][idy] += lwo[idx][0][offset];
          offset++;
        }
      }
    }
    assert(offset == odim2);
  }

}


#endif
