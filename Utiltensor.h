#ifndef UTILTENSOR
#define UTILTENSOR

#include "tensor.h"
#include "MyLib.h"

using namespace std;
using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;
using namespace nr;

// define tanh operation
struct nl_tanh {
	MSHADOW_XINLINE static dtype Map(dtype a) {
//    	return a>0?a:0;
		return tanh(a);
	}
};
struct nl_dtanh {
	MSHADOW_XINLINE static dtype Map(dtype a) {
//    	return a>0?1:0;
		return (1.0 - a) * (1.0 + a);
	}
};
struct nl_sigmoid {
	MSHADOW_XINLINE static dtype Map(dtype a) {
//    	return a>0?a:0;
		return 1.0 / (1.0 + exp(-a));
	}
};
struct nl_dsigmoid {
	MSHADOW_XINLINE static dtype Map(dtype a) {
//    	return a>0?1:0;
		return (1.0 - a) * a;
	}
};
struct nl_relu {
	MSHADOW_XINLINE static dtype Map(dtype a) {
		return a > 0 ? a : 0;
	}
};
struct nl_drelu {
	MSHADOW_XINLINE static dtype Map(dtype a) {
		return a > 0 ? 1 : 0;
	}
};
struct nl_exp {
	MSHADOW_XINLINE static dtype Map(dtype a) {
//    	return a>0?a:0;
		return exp(a);
	}
};
struct nl_log {
	MSHADOW_XINLINE static dtype Map(dtype a) {
//      return a>0?a:0;
		return log(a);
	}
};
struct xe_dx {
	MSHADOW_XINLINE static dtype Map(dtype a, dtype b) {
		return (b - a) / (a * (1.0 - a) + 1e-6);
	}
};
struct xe_ll {
	MSHADOW_XINLINE static dtype Map(dtype a, dtype b) {
		return b > 0.5f ? log(a + 1e-10) : log(1.0 - a + 1e-10);
	}
};
struct square {
	MSHADOW_XINLINE static dtype Map(dtype a) {
		return a * a;

	}
};
struct clip {
	MSHADOW_XINLINE static dtype Map(dtype a) {
		return a > 10.0 ? 10.0 : (a < -10.0 ? -10.0 : a);

	}
};
struct inv_sqrt {
	MSHADOW_XINLINE static dtype Map(dtype a, dtype b) {
		return a / (sqrt(b) + 0.0001);
	}
};

struct nl_sqrt {
	MSHADOW_XINLINE static dtype Map(dtype a) {
		return sqrt(a);
	}
};

struct dropout {
	// p: prob to dropout
	MSHADOW_XINLINE static dtype Map(dtype p, dtype r) {
		if (p > r)
			return 0.0;
		else
			return 1.0 / (1.0 - p);
	}
};

// \sum x_{ijk}^2
template<typename xpu>
inline dtype squarenorm(Tensor<xpu, 1, dtype> w) {
	dtype result = 0;
	for (int idx = 0; idx < w.size(0); idx++) {
		result += w[idx] * w[idx];
	}
	return result;
}

template<typename xpu>
inline dtype squarenorm(Tensor<xpu, 2, dtype> w) {
	dtype result = 0;
	for (int idx = 0; idx < w.size(0); idx++) {
		for (int idy = 0; idy < w.size(1); idy++) {
			result += w[idx][idy] * w[idx][idy];
		}
	}
	return result;
}

template<typename xpu>
inline dtype squarenorm(Tensor<xpu, 3, dtype> w) {
	dtype result = 0;
	for (int idx = 0; idx < w.size(0); idx++) {
		for (int idy = 0; idy < w.size(1); idy++) {
			for (int idz = 0; idz < w.size(2); idz++) {
				result += w[idx][idy][idz] * w[idx][idy][idz];
			}
		}
	}
	return result;
}

template<typename xpu>
inline void assign(Tensor<xpu, 1, dtype> w, const NRVec<dtype>& wnr) {
	int dim = wnr.size();
	for (int idx = 0; idx < dim; idx++) {
		w[idx] = wnr[idx];
	}
}

template<typename xpu>
inline void assign(Tensor<xpu, 2, dtype> w, const NRMat<dtype>& wnr) {
	int dim1 = wnr.nrows();
	int dim2 = wnr.ncols();
	for (int idx = 0; idx < dim1; idx++) {
		for (int idy = 0; idy < dim2; idy++) {
			w[idx][idy] = wnr[idx][idy];
		}
	}
}

template<typename xpu>
inline void assign(Tensor<xpu, 3, dtype> w, const NRMat3d<dtype>& wnr) {
	int dim1 = wnr.dim1();
	int dim2 = wnr.dim2();
	int dim3 = wnr.dim3();
	for (int idx = 0; idx < dim1; idx++) {
		for (int idy = 0; idy < dim2; idy++) {
			for (int idz = 0; idz < dim3; idz++) {
				w[idx][idy][idz] = wnr[idx][idy][idz];
			}
		}
	}
}

template<typename xpu>
inline void assign(vector<Tensor<xpu, 1, dtype> > &w, dtype value) {
	int dim = w.size();
	for (int idx = 0; idx < dim; idx++) {
		w[idx] = value;
	}
}

template<typename xpu>
inline void assign(vector<Tensor<xpu, 2, dtype> > &w, dtype value) {
	int dim = w.size();
	for (int idx = 0; idx < dim; idx++) {
		w[idx] = value;
	}
}

template<typename xpu>
inline void assign(vector<Tensor<xpu, 3, dtype> > &w, dtype value) {
	int dim = w.size();
	for (int idx = 0; idx < dim; idx++) {
		w[idx] = value;
	}
}

template<typename xpu>
inline void norm2one(Tensor<xpu, 2, dtype> w, int idx) {
	dtype sum = 0.000001;
	for (int idy = 0; idy < w.size(1); idy++) {
		sum += w[idx][idy] * w[idx][idy];
	}
	dtype scale = sqrt(sum);
	for (int idy = 0; idy < w.size(1); idy++)
		w[idx][idy] = w[idx][idy] / scale;
}

template<typename xpu>
inline void random(Tensor<xpu, 1, dtype> w, dtype min = 0.0, dtype max = 1.0, int seed = 0) {
	srand(seed);
	int dim = w.size(0);
	for (int idx = 0; idx < dim; idx++) {
		w[idx] = min + (max - min) * (1.0 * rand() / RAND_MAX);
	}
}

template<typename xpu>
inline void random(Tensor<xpu, 2, dtype> w, dtype min = 0.0, dtype max = 1.0, int seed = 0) {
	srand(seed);
	int dim1 = w.size(0);
	int dim2 = w.size(1);
	for (int idx = 0; idx < dim1; idx++) {
		for (int idy = 0; idy < dim2; idy++) {
			w[idx][idy] = min + (max - min) * (1.0 * rand() / RAND_MAX);
		}
	}
}

template<typename xpu>
inline void random(Tensor<xpu, 3, dtype> w, dtype min = 0.0, dtype max = 1.0, int seed = 0) {
	srand(seed);
	int dim1 = w.size(0);
	int dim2 = w.size(1);
	int dim3 = w.size(2);
	for (int idx = 0; idx < dim1; idx++) {
		for (int idy = 0; idy < dim2; idy++) {
			for (int idz = 0; idz < dim3; idz++) {
				w[idx][idy][idz] = min + (max - min) * (1.0 * rand() / RAND_MAX);
			}
		}
	}
}

/*
template<typename xpu>
inline void tcopy(const Tensor<xpu, 3, dtype>& from, Tensor<xpu, 3, dtype>& to, bool bAllocated = true) {
	if (bAllocated) {
		if (to.size(0) != from.size(0) || to.size(1) != from.size(1) || to.size(2) != from.size(2)) {
			FreeSpace(&to);
			to = NewTensor<xpu>(Shape3(from.size(0), from.size(1), from.size(2)), d_zero);
		}
	} else {
		to = NewTensor<xpu>(Shape3(from.size(0), from.size(1), from.size(2)), d_zero);
	}

	Copy(to, from);
}

template<typename xpu>
inline void tcopy(const Tensor<xpu, 2, dtype>& from, Tensor<xpu, 2, dtype>& to, bool bAllocated = true) {
	if (bAllocated) {
		if (to.size(0) != from.size(0) || to.size(1) != from.size(1)) {
			FreeSpace(&to);
			to = NewTensor<xpu>(Shape2(from.size(0), from.size(1)), d_zero);
		}
	} else {
		to = NewTensor<xpu>(Shape2(from.size(0), from.size(1)), d_zero);
	}
	Copy(to, from);
}

template<typename xpu>
inline void tcopy(const Tensor<xpu, 1, dtype>&from, Tensor<xpu, 1, dtype>& to, bool bAllocated = true) {
	if (bAllocated) {
		if (to.size(0) != from.size(0)) {
			FreeSpace(&to);
			to = NewTensor<xpu>(Shape1(from.size(0)), d_zero);
		}
	} else {
		to = NewTensor<xpu>(Shape1(from.size(0)), d_zero);
	}
	Copy(to, from);
}
*/
#endif
