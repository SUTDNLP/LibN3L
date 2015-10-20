#ifndef DROPOUT
#define DROPOUT

#include "tensor.h"
#include "MyLib.h"


using namespace std;
using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;


template<typename xpu>
inline void dropoutcol(Tensor<xpu, 2, dtype> w, dtype dropOut)
{
	w = 1.0;
	std::vector<int> indexes;
  for (int i = 0; i < w.size(1); ++i)
    indexes.push_back(i);
  int dropNum =   (int) (w.size(1) * dropOut);
  
	for(int idx = 0; idx < w.size(0); idx++)
	{
		random_shuffle(indexes.begin(), indexes.end());
		for(int idy = 0; idy < dropNum; idy++)
		{
			w[idx][indexes[idy]] = 0.0;
		}
	}
}


template<typename xpu>
inline void dropoutrow(Tensor<xpu, 2, dtype> w, dtype dropOut)
{
	w = 1.0;
	std::vector<int> indexes;
  for (int i = 0; i < w.size(0); ++i)
    indexes.push_back(i);
  int dropNum = (int) (w.size(0) * dropOut);
  
	for(int idx = 0; idx < w.size(1); idx++)
	{
		random_shuffle(indexes.begin(), indexes.end());
		for(int idy = 0; idy < dropNum; idy++)
		{
			w[indexes[idy]][idx] = 0.0;
		}
	}
}


#endif
