#ifndef CONCAT
#define CONCAT

#include "tensor.h"
#include "MyLib.h"

using namespace std;
using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;

// For all the contact-related functions,
// we add the related values into the targeted matrices.
// for uncat functions, bclear by false denotes that the losses are accumulated.

//only applicable on Shape2(1,x), notice that we add the value to the target
template<typename xpu>
inline void concat(Tensor<xpu, 2, dtype> w1, Tensor<xpu, 2, dtype> w2, Tensor<xpu, 2, dtype> w) {
  if (w1.size(0) != 1 || w2.size(0) != 1 || w.size(0) != 1) {
    std::cerr << "concat error, only support Shape2(1,x)" << std::endl;
    return;
  }
  int row = w.size(0);
  int col = w.size(1);
  int col1 = w1.size(1);
  int col2 = w2.size(1);
  if (col1 + col2 != col) {
    std::cerr << "col check error!" << std::endl;
    return;
  }
  int offset;
  w = 0.0;
  for (int idx = 0; idx < row; idx++) {
    offset = 0;
    for (int idy = 0; idy < col1; idy++) {
      w[idx][offset] += w1[idx][idy];
      offset++;
    }
    for (int idy = 0; idy < col2; idy++) {
      w[idx][offset] += w2[idx][idy];
      offset++;
    }
  }
  return;
}

template<typename xpu>
inline void concat(Tensor<xpu, 2, dtype> w1, Tensor<xpu, 2, dtype> w2, Tensor<xpu, 2, dtype> w3, Tensor<xpu, 2, dtype> w) {
  if (w1.size(0) != 1 || w2.size(0) != 1 || w3.size(0) != 1 || w.size(0) != 1) {
    std::cerr << "concat error, only support Shape2(1,x)" << std::endl;
    return;
  }
  int row = w.size(0);
  int col = w.size(1);
  int col1 = w1.size(1);
  int col2 = w2.size(1);
  int col3 = w3.size(1);
  if (col1 + col2 + col3 != col) {
    std::cerr << "col check error!" << std::endl;
    return;
  }
  int offset;
  w = 0.0;
  for (int idx = 0; idx < row; idx++) {
    offset = 0;
    for (int idy = 0; idy < col1; idy++) {
      w[idx][offset] += w1[idx][idy];
      offset++;
    }
    for (int idy = 0; idy < col2; idy++) {
      w[idx][offset] += w2[idx][idy];
      offset++;
    }
    for (int idy = 0; idy < col3; idy++) {
      w[idx][offset] += w3[idx][idy];
      offset++;
    }
  }
  return;
}

template<typename xpu>
inline void concat(Tensor<xpu, 2, dtype> w1, Tensor<xpu, 2, dtype> w2, Tensor<xpu, 2, dtype> w3, Tensor<xpu, 2, dtype> w4, Tensor<xpu, 2, dtype> w) {
  if (w1.size(0) != 1 || w2.size(0) != 1 || w3.size(0) != 1 || w4.size(0) != 1 || w.size(0) != 1) {
    std::cerr << "concat error, only support Shape2(1,x)" << std::endl;
    return;
  }
  int row = w.size(0);
  int col = w.size(1);
  int col1 = w1.size(1);
  int col2 = w2.size(1);
  int col3 = w3.size(1);
  int col4 = w4.size(1);
  if (col1 + col2 + col3 + col4 != col) {
    std::cerr << "col check error!" << std::endl;
    return;
  }
  int offset;
  w = 0.0;
  for (int idx = 0; idx < row; idx++) {
    offset = 0;
    for (int idy = 0; idy < col1; idy++) {
      w[idx][offset] += w1[idx][idy];
      offset++;
    }
    for (int idy = 0; idy < col2; idy++) {
      w[idx][offset] += w2[idx][idy];
      offset++;
    }
    for (int idy = 0; idy < col3; idy++) {
      w[idx][offset] += w3[idx][idy];
      offset++;
    }
    for (int idy = 0; idy < col4; idy++) {
      w[idx][offset] += w4[idx][idy];
      offset++;
    }
  }
  return;
}

template<typename xpu>
inline void concat(Tensor<xpu, 2, dtype> w1, Tensor<xpu, 2, dtype> w2, Tensor<xpu, 2, dtype> w3, Tensor<xpu, 2, dtype> w4, Tensor<xpu, 2, dtype> w5,
    Tensor<xpu, 2, dtype> w) {
  if (w1.size(0) != 1 || w2.size(0) != 1 || w3.size(0) != 1 || w4.size(0) != 1 || w5.size(0) != 1 || w.size(0) != 1) {
    std::cerr << "concat error, only support Shape2(1,x)" << std::endl;
    return;
  }
  int row = w.size(0);
  int col = w.size(1);
  int col1 = w1.size(1);
  int col2 = w2.size(1);
  int col3 = w3.size(1);
  int col4 = w4.size(1);
  int col5 = w5.size(1);
  if (col1 + col2 + col3 + col4 + col5 != col) {
    std::cerr << "col check error!" << std::endl;
    return;
  }
  int offset;
  w = 0.0;
  for (int idx = 0; idx < row; idx++) {
    offset = 0;
    for (int idy = 0; idy < col1; idy++) {
      w[idx][offset] += w1[idx][idy];
      offset++;
    }
    for (int idy = 0; idy < col2; idy++) {
      w[idx][offset] += w2[idx][idy];
      offset++;
    }
    for (int idy = 0; idy < col3; idy++) {
      w[idx][offset] += w3[idx][idy];
      offset++;
    }
    for (int idy = 0; idy < col4; idy++) {
      w[idx][offset] += w4[idx][idy];
      offset++;
    }
    for (int idy = 0; idy < col5; idy++) {
      w[idx][offset] += w5[idx][idy];
      offset++;
    }
  }
  return;
}

template<typename xpu>
inline void concat(Tensor<xpu, 2, dtype> w1, Tensor<xpu, 2, dtype> w2, Tensor<xpu, 2, dtype> w3, Tensor<xpu, 2, dtype> w4, Tensor<xpu, 2, dtype> w5,
    Tensor<xpu, 2, dtype> w6, Tensor<xpu, 2, dtype> w) {
  if (w1.size(0) != 1 || w2.size(0) != 1 || w3.size(0) != 1 || w4.size(0) != 1 || w5.size(0) != 1 || w6.size(0) != 1 || w.size(0) != 1) {
    std::cerr << "concat error, only support Shape2(1,x)" << std::endl;
    return;
  }
  int row = w.size(0);
  int col = w.size(1);
  int col1 = w1.size(1);
  int col2 = w2.size(1);
  int col3 = w3.size(1);
  int col4 = w4.size(1);
  int col5 = w5.size(1);
  int col6 = w6.size(1);
  if (col1 + col2 + col3 + col4 + col5 + col6 != col) {
    std::cerr << "col check error!" << std::endl;
    return;
  }
  int offset;
  w = 0.0;
  for (int idx = 0; idx < row; idx++) {
    offset = 0;
    for (int idy = 0; idy < col1; idy++) {
      w[idx][offset] += w1[idx][idy];
      offset++;
    }
    for (int idy = 0; idy < col2; idy++) {
      w[idx][offset] += w2[idx][idy];
      offset++;
    }
    for (int idy = 0; idy < col3; idy++) {
      w[idx][offset] += w3[idx][idy];
      offset++;
    }
    for (int idy = 0; idy < col4; idy++) {
      w[idx][offset] += w4[idx][idy];
      offset++;
    }
    for (int idy = 0; idy < col5; idy++) {
      w[idx][offset] += w5[idx][idy];
      offset++;
    }
    for (int idy = 0; idy < col6; idy++) {
      w[idx][offset] += w6[idx][idy];
      offset++;
    }
  }
  return;
}

template<typename xpu>
inline void concat(vector<Tensor<xpu, 2, dtype> > wi, Tensor<xpu, 2, dtype> w, int distance = 0) {
  // distance denotes right shift position, if less than zero,
  // denotes right (-distance) values are filled with zeors
  for (int num = 0; num < wi.size(); num++) {
    if (wi[num].size(0) != 1) {
      std::cerr << "concat error, only support Shape2(1,x)" << std::endl;
      return;
    }
  }
  int row = w.size(0);
  int col = w.size(1);
  int sumcol = (distance >= 0) ? distance : -distance;
  for (int num = 0; num < wi.size(); num++) {
    sumcol += wi[num].size(1);
  }

  if (sumcol != col) {
    std::cerr << "col check error!" << std::endl;
    return;
  }
  int offset;
  w = 0.0;
  for (int idx = 0; idx < row; idx++) {
    offset = (distance >= 0) ? distance : 0;
    for (int num = 0; num < wi.size(); num++) {
      for (int idy = 0; idy < wi[num].size(1); idy++) {
        w[idx][offset] += wi[num][idx][idy];
        offset++;
      }
    }
  }
  return;
}

template<typename xpu>
inline void concat(Tensor<xpu, 3, dtype> wi, Tensor<xpu, 2, dtype> w, int distance = 0) {
  // distance denotes right shift position, if less than zero,
  // denotes right (-distance) values are filled with zeors
  for (int num = 0; num < wi.size(0); num++) {
    if (wi[num].size(0) != 1) {
      std::cerr << "concat error, only support Shape2(1,x)" << std::endl;
      return;
    }
  }
  int row = w.size(0);
  int col = w.size(1);
  int sumcol = (distance >= 0) ? distance : -distance;
  for (int num = 0; num < wi.size(0); num++) {
    sumcol += wi[num].size(1);
  }

  if (sumcol != col) {
    std::cerr << "col check error!" << std::endl;
    return;
  }
  int offset;
  w = 0.0;
  for (int idx = 0; idx < row; idx++) {
    offset = (distance >= 0) ? distance : 0;
    for (int num = 0; num < wi.size(0); num++) {
      for (int idy = 0; idy < wi[num].size(1); idy++) {
        w[idx][offset] += wi[num][idx][idy];
        offset++;
      }
    }
  }
  return;
}

//only applicable on Shape2(1,x), notice that we add the value to the target

template<typename xpu>
inline void unconcat(Tensor<xpu, 2, dtype> w1, Tensor<xpu, 2, dtype> w2, Tensor<xpu, 2, dtype> w, bool bclear = false) {
  if (w1.size(0) != 1 || w2.size(0) != 1 || w.size(0) != 1) {
    std::cerr << "unconcat error, only spport Shape2(1,x)" << std::endl;
    return;
  }
  int row = w.size(0);
  int col = w.size(1);
  int col1 = w1.size(1);
  int col2 = w2.size(1);
  if (col1 + col2 != col) {
    std::cerr << "col check error!" << std::endl;
    return;
  }
  int offset;
  if (bclear) {
    w1 = 0.0;
    w2 = 0.0;
  }
  for (int idx = 0; idx < row; idx++) {
    offset = 0;
    for (int idy = 0; idy < col1; idy++) {
      w1[idx][idy] += w[idx][offset];
      offset++;
    }
    for (int idy = 0; idy < col2; idy++) {
      w2[idx][idy] += w[idx][offset];
      offset++;
    }
  }
  return;
}

template<typename xpu>
inline void unconcat(Tensor<xpu, 2, dtype> w1, Tensor<xpu, 2, dtype> w2, Tensor<xpu, 2, dtype> w3, Tensor<xpu, 2, dtype> w, bool bclear = false) {
  if (w1.size(0) != 1 || w2.size(0) != 1 || w3.size(0) != 1 || w.size(0) != 1) {
    std::cerr << "unconcat error, only spport Shape2(1,x)" << std::endl;
    return;
  }
  int row = w.size(0);
  int col = w.size(1);
  int col1 = w1.size(1);
  int col2 = w2.size(1);
  int col3 = w3.size(1);
  if (col1 + col2 + col3 != col) {
    std::cerr << "col check error!" << std::endl;
    return;
  }
  int offset;
  if (bclear) {
    w1 = 0.0;
    w2 = 0.0;
    w3 = 0.0;
  }
  for (int idx = 0; idx < row; idx++) {
    offset = 0;
    for (int idy = 0; idy < col1; idy++) {
      w1[idx][idy] += w[idx][offset];
      offset++;
    }
    for (int idy = 0; idy < col2; idy++) {
      w2[idx][idy] += w[idx][offset];
      offset++;
    }
    for (int idy = 0; idy < col3; idy++) {
      w3[idx][idy] += w[idx][offset];
      offset++;
    }
  }
  return;
}

template<typename xpu>
inline void unconcat(Tensor<xpu, 2, dtype> w1, Tensor<xpu, 2, dtype> w2, Tensor<xpu, 2, dtype> w3, Tensor<xpu, 2, dtype> w4, Tensor<xpu, 2, dtype> w,
    bool bclear = false) {
  if (w1.size(0) != 1 || w2.size(0) != 1 || w3.size(0) != 1 || w4.size(0) != 1 || w.size(0) != 1) {
    std::cerr << "unconcat error, only spport Shape2(1,x)" << std::endl;
    return;
  }
  int row = w.size(0);
  int col = w.size(1);
  int col1 = w1.size(1);
  int col2 = w2.size(1);
  int col3 = w3.size(1);
  int col4 = w4.size(1);
  if (col1 + col2 + col3 + col4 != col) {
    std::cerr << "col check error!" << std::endl;
    return;
  }
  int offset;
  if (bclear) {
    w1 = 0.0;
    w2 = 0.0;
    w3 = 0.0;
    w4 = 0.0;
  }
  for (int idx = 0; idx < row; idx++) {
    offset = 0;
    for (int idy = 0; idy < col1; idy++) {
      w1[idx][idy] += w[idx][offset];
      offset++;
    }
    for (int idy = 0; idy < col2; idy++) {
      w2[idx][idy] += w[idx][offset];
      offset++;
    }
    for (int idy = 0; idy < col3; idy++) {
      w3[idx][idy] += w[idx][offset];
      offset++;
    }
    for (int idy = 0; idy < col4; idy++) {
      w4[idx][idy] += w[idx][offset];
      offset++;
    }
  }
  return;
}

template<typename xpu>
inline void unconcat(Tensor<xpu, 2, dtype> w1, Tensor<xpu, 2, dtype> w2, Tensor<xpu, 2, dtype> w3, Tensor<xpu, 2, dtype> w4, Tensor<xpu, 2, dtype> w5,
    Tensor<xpu, 2, dtype> w, bool bclear = false) {
  if (w1.size(0) != 1 || w2.size(0) != 1 || w3.size(0) != 1 || w4.size(0) != 1 || w5.size(0) != 1 || w.size(0) != 1) {
    std::cerr << "unconcat error, only spport Shape2(1,x)" << std::endl;
    return;
  }
  int row = w.size(0);
  int col = w.size(1);
  int col1 = w1.size(1);
  int col2 = w2.size(1);
  int col3 = w3.size(1);
  int col4 = w4.size(1);
  int col5 = w5.size(1);
  if (col1 + col2 + col3 + col4 + col5 != col) {
    std::cerr << "col check error!" << std::endl;
    return;
  }
  int offset;
  if (bclear) {
    w1 = 0.0;
    w2 = 0.0;
    w3 = 0.0;
    w4 = 0.0;
    w5 = 0.0;
  }
  for (int idx = 0; idx < row; idx++) {
    offset = 0;
    for (int idy = 0; idy < col1; idy++) {
      w1[idx][idy] += w[idx][offset];
      offset++;
    }
    for (int idy = 0; idy < col2; idy++) {
      w2[idx][idy] += w[idx][offset];
      offset++;
    }
    for (int idy = 0; idy < col3; idy++) {
      w3[idx][idy] += w[idx][offset];
      offset++;
    }
    for (int idy = 0; idy < col4; idy++) {
      w4[idx][idy] += w[idx][offset];
      offset++;
    }
    for (int idy = 0; idy < col5; idy++) {
      w5[idx][idy] += w[idx][offset];
      offset++;
    }
  }
  return;
}

template<typename xpu>
inline void unconcat(Tensor<xpu, 2, dtype> w1, Tensor<xpu, 2, dtype> w2, Tensor<xpu, 2, dtype> w3, Tensor<xpu, 2, dtype> w4, Tensor<xpu, 2, dtype> w5,
    Tensor<xpu, 2, dtype> w6, Tensor<xpu, 2, dtype> w, bool bclear = false) {
  if (w1.size(0) != 1 || w2.size(0) != 1 || w3.size(0) != 1 || w4.size(0) != 1 || w5.size(0) != 1 || w6.size(0) != 1 || w.size(0) != 1) {
    std::cerr << "unconcat error, only spport Shape2(1,x)" << std::endl;
    return;
  }
  int row = w.size(0);
  int col = w.size(1);
  int col1 = w1.size(1);
  int col2 = w2.size(1);
  int col3 = w3.size(1);
  int col4 = w4.size(1);
  int col5 = w5.size(1);
  int col6 = w6.size(1);
  if (col1 + col2 + col3 + col4 + col5 + col6 != col) {
    std::cerr << "col check error!" << std::endl;
    return;
  }
  int offset;
  if (bclear) {
    w1 = 0.0;
    w2 = 0.0;
    w3 = 0.0;
    w4 = 0.0;
    w5 = 0.0;
    w6 = 0.0;
  }
  for (int idx = 0; idx < row; idx++) {
    offset = 0;
    for (int idy = 0; idy < col1; idy++) {
      w1[idx][idy] += w[idx][offset];
      offset++;
    }
    for (int idy = 0; idy < col2; idy++) {
      w2[idx][idy] += w[idx][offset];
      offset++;
    }
    for (int idy = 0; idy < col3; idy++) {
      w3[idx][idy] += w[idx][offset];
      offset++;
    }
    for (int idy = 0; idy < col4; idy++) {
      w4[idx][idy] += w[idx][offset];
      offset++;
    }
    for (int idy = 0; idy < col5; idy++) {
      w5[idx][idy] += w[idx][offset];
      offset++;
    }
    for (int idy = 0; idy < col6; idy++) {
      w6[idx][idy] += w[idx][offset];
      offset++;
    }
  }
  return;
}

template<typename xpu>
inline void unconcat(vector<Tensor<xpu, 2, dtype> > wi, Tensor<xpu, 2, dtype> w, int distance = 0, bool bclear = false) {
  for (int num = 0; num < wi.size(); num++) {
    if (wi[num].size(0) != 1) {
      std::cerr << "concat error, only support Shape2(1,x)" << std::endl;
      return;
    }
  }
  int row = w.size(0);
  int col = w.size(1);
  int sumcol = (distance >= 0) ? distance : -distance;
  for (int num = 0; num < wi.size(); num++) {
    sumcol += wi[num].size(1);
  }

  if (sumcol != col) {
    std::cerr << "col check error!" << std::endl;
    return;
  }
  int offset;
  if (bclear) {
    for (int num = 0; num < wi.size(); num++) {
      wi[num] = 0.0;
    }
  }
  for (int idx = 0; idx < row; idx++) {
    offset = (distance >= 0) ? distance : 0;
    for (int num = 0; num < wi.size(); num++) {
      for (int idy = 0; idy < wi[num].size(1); idy++) {
        wi[num][idx][idy] += w[idx][offset];
        offset++;
      }
    }
  }
  return;
}

template<typename xpu>
inline void unconcat(Tensor<xpu, 3, dtype> wi, Tensor<xpu, 2, dtype> w, int distance = 0, bool bclear = false) {
  for (int num = 0; num < wi.size(0); num++) {
    if (wi[num].size(0) != 1) {
      std::cerr << "concat error, only support Shape2(1,x)" << std::endl;
      return;
    }
  }
  int row = w.size(0);
  int col = w.size(1);
  int sumcol = (distance >= 0) ? distance : -distance;
  for (int num = 0; num < wi.size(0); num++) {
    sumcol += wi[num].size(1);
  }

  if (sumcol != col) {
    std::cerr << "col check error!" << std::endl;
    return;
  }
  int offset;
  if (bclear) {
    wi = 0.0;
  }
  for (int idx = 0; idx < row; idx++) {
    offset = (distance >= 0) ? distance : 0;
    for (int num = 0; num < wi.size(0); num++) {
      for (int idy = 0; idy < wi[num].size(1); idy++) {
        wi[num][idx][idy] += w[idx][offset];
        offset++;
      }
    }
  }
  return;
}

#endif
