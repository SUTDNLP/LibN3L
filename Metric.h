/*
 * Metric.h
 *
 *  Created on: Mar 17, 2015
 *      Author: mszhang
 */

#ifndef SRC_METRIC_H_
#define SRC_METRIC_H_


using namespace std;

class Metric {

public:
  int overall_label_count;
  int correct_label_count;
  int predicated_label_count;

public:
  Metric()
  {
    overall_label_count = 0;
    correct_label_count = 0;
    predicated_label_count = 0;
  }

  ~Metric(){}

  void reset()
  {
    overall_label_count = 0;
    correct_label_count = 0;
    predicated_label_count = 0;
  }

  bool bIdentical()
  {
    if(predicated_label_count == 0)
    {
      if(overall_label_count == correct_label_count)
      {
        return true;
      }
      return false;
    }
    else
    {
      if(overall_label_count == correct_label_count && predicated_label_count == correct_label_count)
      {
        return true;
      }
      return false;
    }
  }

  double getAccuracy()
  {
    if(predicated_label_count == 0)
    {
      return correct_label_count*1.0/overall_label_count;
    }
    else
    {
      return correct_label_count*2.0/(overall_label_count + predicated_label_count);
    }
  }


  void print()
  {
    if(predicated_label_count == 0)
    {
      std::cout << "Accuracy:\tP=" << correct_label_count << "/" << overall_label_count
          << "=" << correct_label_count*1.0/overall_label_count << std::endl;
    }
    else
    {
      std::cout << "Recall:\tP=" << correct_label_count << "/" << overall_label_count << "=" << correct_label_count*1.0/overall_label_count
        << ", " << "Accuracy:\tP=" << correct_label_count << "/" << predicated_label_count << "=" << correct_label_count*1.0/predicated_label_count
        << ", " << "Fmeasure:\t" << correct_label_count*2.0/(overall_label_count + predicated_label_count) << std::endl;
    }
  }


};

#endif /* SRC_EXAMPLE_H_ */
