/*
 * CheckGrad.h
 *
 *  Created on: Dec 4, 2015
 *      Author: mason
 */

#ifndef BASIC_CHECKGRAD_H_
#define BASIC_CHECKGRAD_H_

#include <iostream>
#include "tensor.h"
#include "MyLib.h"

using namespace nr;
using namespace std;
using namespace mshadow;
using namespace mshadow::expr;
using namespace mshadow::utils;

template<typename xpu, typename Example, typename Classifier>
void checkgrad(Classifier* classifier, const vector<Example>& examples, Tensor<xpu, 2, double>& Wd,
		const Tensor<xpu, 2, double>& gradWd, const string& mark, int iter) {
	int charseed = mark.length();
	for (int i = 0; i < mark.length(); i++) {
		charseed = (int) (mark[i]) * 5 + charseed;
	}
	srand(iter + charseed);
	std::vector<int> idRows, idCols;
	idRows.clear();
	idCols.clear();
	for (int i = 0; i < Wd.size(0); ++i)
		idRows.push_back(i);
	for (int idx = 0; idx < Wd.size(1); idx++)
		idCols.push_back(idx);

	random_shuffle(idRows.begin(), idRows.end());
	random_shuffle(idCols.begin(), idCols.end());

	int check_i = idRows[0], check_j = idCols[0];

	dtype orginValue = Wd[check_i][check_j];

	Wd[check_i][check_j] = orginValue + 0.001;
	dtype lossAdd = 0.0;
	for (int i = 0; i < examples.size(); i++) {
		Example oneExam = examples[i];
		lossAdd += classifier->computeScore(oneExam);
	}

	Wd[check_i][check_j] = orginValue - 0.001;
	dtype lossPlus = 0.0;
	for (int i = 0; i < examples.size(); i++) {
		Example oneExam = examples[i];
		lossPlus += classifier->computeScore(oneExam);
	}

	dtype mockGrad = (lossAdd - lossPlus) / 0.002;
	mockGrad = mockGrad / examples.size();
	dtype computeGrad = gradWd[check_i][check_j];

	printf("Iteration %d, Checking gradient for %s[%d][%d]:\t", iter,
			mark.c_str(), check_i, check_j);
	printf("mock grad = %.18f, computed grad = %.18f\n", mockGrad, computeGrad);

	Wd[check_i][check_j] = orginValue;
}

template<typename xpu, typename Example, typename Classifier>
void checkgrad(Classifier* classifier, const vector<Example>& examples, Tensor<xpu, 2, double>& Wd,
		const Tensor<xpu, 2, double>& gradWd, const string& mark, int iter,
		const hash_set<int>& indexes, bool bRow = true) {
	if (indexes.size() == 0)
		return;
	int charseed = mark.length();
	for (int i = 0; i < mark.length(); i++) {
		charseed = (int) (mark[i]) * 5 + charseed;
	}
	srand(iter + charseed);
	std::vector<int> idRows, idCols;
	idRows.clear();
	idCols.clear();
	static hash_set<int>::iterator it;
	if (bRow) {
		for (it = indexes.begin(); it != indexes.end(); ++it)
			idRows.push_back(*it);
		for (int idx = 0; idx < Wd.size(1); idx++)
			idCols.push_back(idx);
	} else {
		for (it = indexes.begin(); it != indexes.end(); ++it)
			idCols.push_back(*it);
		for (int idx = 0; idx < Wd.size(0); idx++)
			idRows.push_back(idx);
	}

	random_shuffle(idRows.begin(), idRows.end());
	random_shuffle(idCols.begin(), idCols.end());

	int check_i = idRows[0], check_j = idCols[0];

	dtype orginValue = Wd[check_i][check_j];

	Wd[check_i][check_j] = orginValue + 0.001;
	dtype lossAdd = 0.0;
	for (int i = 0; i < examples.size(); i++) {
		Example oneExam = examples[i];
		lossAdd += classifier->computeScore(oneExam);
	}

	Wd[check_i][check_j] = orginValue - 0.001;
	dtype lossPlus = 0.0;
	for (int i = 0; i < examples.size(); i++) {
		Example oneExam = examples[i];
		lossPlus += classifier->computeScore(oneExam);
	}

	dtype mockGrad = (lossAdd - lossPlus) / 0.002;
	mockGrad = mockGrad / examples.size();
	dtype computeGrad = gradWd[check_i][check_j];

	printf("Iteration %d, Checking gradient for %s[%d][%d]:\t", iter,
			mark.c_str(), check_i, check_j);
	printf("mock grad = %.18f, computed grad = %.18f\n", mockGrad, computeGrad);

	Wd[check_i][check_j] = orginValue;

}


#endif /* BASIC_CHECKGRAD_H_ */
