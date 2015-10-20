/////////////////////////////////////////////////////////////////////////////////////
// File Name   : MyLib.h
// Project Name: IRLAS
// Author      : Huipeng Zhang (zhp@ir.hit.edu.cn)
// Environment : Microsoft Visual C++ 6.0
// Description : some utility functions
// Time        : 2005.9
// History     :
// CopyRight   : HIT-IRLab (c) 2001-2005, all rights reserved.
/////////////////////////////////////////////////////////////////////////////////////
#ifndef _MYLIB_H_
#define _MYLIB_H_

#include <string>
#include <vector>
#include <fstream>
#include <cassert>
#include <deque>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <cmath>
#include <ctime>
#include <cfloat>
#include <cstring>
#include <sstream>


#include "Hash_map.hpp"
#include "NRMat.h"

using namespace nr;
using namespace std;



typedef double dtype;

const static dtype minlogvalue = -1000;
const static dtype d_zero = 0.0;
const static dtype d_one = 1.0;

typedef std::vector<std::string> CStringVector;

typedef std::vector<std::pair<std::string, std::string> > CTwoStringVector;

class string_less {
public:
  bool operator()(const string &str1, const string &str2) const {
    int ret = strcmp(str1.c_str(), str2.c_str());
    if (ret < 0)
      return true;
    else
      return false;
  }
};

class LabelScore {
public:
  int labelId;
  dtype score;

public:
  LabelScore() {
    labelId = -1;
    score = 0.0;
  }
  LabelScore(int id, dtype value) {
    labelId = id;
    score = value;
  }
};

class LabelScore_Compare {
public:
  bool operator()(const LabelScore &o1, const LabelScore &o2) const {

    if (o1.score < o2.score)
      return -1;
    else if (o1.score > o2.score)
      return 1;
    else
      return 0;
  }
};

/*==============================================================
 *
 * CSentenceTemplate
 *
 *==============================================================*/

template<typename CSentenceNode>
class CSentenceTemplate: public std::vector<CSentenceNode> {

public:
  CSentenceTemplate() {
  }
  virtual ~CSentenceTemplate() {
  }
};

//==============================================================

template<typename CSentenceNode>
inline std::istream & operator >>(std::istream &is, CSentenceTemplate<CSentenceNode> &sent) {
  sent.clear();
  std::string line;
  while (is && line.empty())
    getline(is, line);

  //getline(is, line);

  while (is && !line.empty()) {
    CSentenceNode node;
    std::istringstream iss(line);
    iss >> node;
    sent.push_back(node);
    getline(is, line);
  }
  return is;
}

template<typename CSentenceNode>
inline std::ostream & operator <<(std::ostream &os, const CSentenceTemplate<CSentenceNode> &sent) {
  for (unsigned i = 0; i < sent.size(); ++i)
    os << sent.at(i) << std::endl;
  os << std::endl;
  return os;
}

inline void print_time() {

  time_t lt = time(NULL);
  cout << ctime(&lt) << endl;

}

inline char* mystrcat(char *dst, const char *src) {
  int n = (dst != 0 ? strlen(dst) : 0);
  dst = (char*) realloc(dst, n + strlen(src) + 1);
  strcat(dst, src);
  return dst;
}

inline char* mystrdup(const char *src) {
  char *dst = (char*) malloc(strlen(src) + 1);
  if (dst != NULL) {
    strcpy(dst, src);
  }
  return dst;
}

inline int message_callback(void *instance, const char *format, va_list args) {
  vfprintf(stdout, format, args);
  fflush(stdout);
  return 0;
}



inline void Free(dtype** p) {
  if (*p != NULL)
    free(*p);
  *p = NULL;
}

//(-scale,scale)
inline void randomMatAssign(dtype* p, int length, dtype scale = 1.0, int seed = 0) {
  srand(seed);
  for (int idx = 0; idx < length; idx++) {
    p[idx] = 2.0 * rand() * scale / RAND_MAX - scale;
  }
}


inline int mod(int v1, int v2) {
  if (v1 < 0 || v2 <= 0)
    return -1;
  else {
    return v1 % v2;
  }
}

inline void ones(dtype* p, int length) {
  for (int idx = 0; idx < length; idx++) {
    p[idx] = 1.0;
  }
}

inline void zeros(dtype* p, int length) {
  for (int idx = 0; idx < length; idx++) {
    p[idx] = 0.0;
  }
}


inline void scaleMat(dtype* p, dtype scale, int length) {
  for (int idx = 0; idx < length; idx++) {
    p[idx] = p[idx] * scale;
  }
}

inline void elemMulMat(dtype* p, dtype* q, int length) {
  for (int idx = 0; idx < length; idx++) {
    p[idx] = p[idx] * q[idx];
  }
}

inline void elemMulMat(dtype* p, dtype* q, dtype *t, int length) {
  for (int idx = 0; idx < length; idx++) {
    t[idx] = p[idx] * q[idx];
  }
}

inline void normalize_mat_onerow(dtype* p, int row, int rowSize, int colSize) {
  dtype sum = 0.000001;
  int start_pos = row * colSize;
  int end_pos = start_pos + colSize;
  for (int idx = start_pos; idx < end_pos; idx++)
    sum = sum + p[idx] * p[idx];
  dtype norm = sqrt(sum);
  for (int idx = start_pos; idx < end_pos; idx++)
    p[idx] = p[idx] / norm;
}

//shift to avg = 0, and then norm = 1
inline void normalize_mat_onecol(dtype* p, int col, int rowSize, int colSize) {
  dtype sum = 0.0;
  int maxLength = rowSize * colSize;
  for (int idx = col; idx < maxLength; idx += rowSize) {
    sum += p[idx];
  }
  dtype avg = sum / colSize;

  sum = 0.000001;
  for (int idx = col; idx < maxLength; idx += rowSize) {
    p[idx] = p[idx] - avg;
    sum += p[idx] * p[idx];
  }

  dtype norm = sqrt(sum);
  for (int idx = col; idx < maxLength; idx += rowSize) {
    p[idx] = p[idx] / norm;
  }
}


inline dtype logsumexp(dtype a[], int length) {
  dtype max = a[0];
  for (int idx = 1; idx < length; idx++) {
    if (a[idx] > max)
      max = a[idx];
  }

  dtype sum = 0;
  for (int idx = 0; idx < length; idx++) {
    if (a[idx] > minlogvalue + 1) {
      sum += exp(a[idx] - max);
    }
  }

  dtype result = max + log(sum);

  if (isnan(result) || isinf(result)) {
    std::cout << "sum = " << sum << ", max = " << max << ", result = " << result << std::endl;
    for (int idx = 0; idx < length; idx++)
      std::cout << a[idx] << " ";
    std::cout << std::endl;
  }

  return max + log(sum);
}

inline bool isPunc(std::string thePostag) {

  if (thePostag.compare("PU") == 0 || thePostag.compare("``") == 0 || thePostag.compare("''") == 0 || thePostag.compare(",") == 0 || thePostag.compare(".") == 0
      || thePostag.compare(":") == 0 || thePostag.compare("-LRB-") == 0 || thePostag.compare("-RRB-") == 0 || thePostag.compare("$") == 0
      || thePostag.compare("#") == 0) {
    return true;
  } else {
    return false;
  }
}

// start some assumptions, "-*-" is a invalid label.
inline bool validlabels(const string& curLabel) {
  if (curLabel[0] == '-' && curLabel[curLabel.length() - 1] == '-') {
    return false;
  }

  return true;
}

inline string cleanLabel(const string& curLabel) {
  if (curLabel.length() > 2 && curLabel[1] == '-') {
    if (curLabel[0] == 'B' || curLabel[0] == 'b' || curLabel[0] == 'M' || curLabel[0] == 'm' || curLabel[0] == 'E' || curLabel[0] == 'e' || curLabel[0] == 'S'
        || curLabel[0] == 's' || curLabel[0] == 'I' || curLabel[0] == 'i') {
      return curLabel.substr(2);
    }
  }

  return curLabel;
}

inline bool is_start_label(const string& label) {
  if (label.length() < 3)
    return false;
  return (label[0] == 'b' || label[0] == 'B' || label[0] == 's' || label[0] == 'S') && label[1] == '-';
}

inline bool is_continue_label(const string& label, const string& startlabel, int distance) {
  if(distance == 0) return true;
  if (label.length() < 3)
    return false;
  if(distance != 0 && is_start_label(label))
    return false;
  if( (startlabel[0] == 's' || startlabel[0] == 'S') && startlabel[1] == '-')
    return false;
  string curcleanlabel = cleanLabel(label);
  string startcleanlabel = cleanLabel(startlabel);
  if(curcleanlabel.compare(startcleanlabel) != 0)
    return false;

  return true;
}

// end some assumptions

inline int cmpPairByValue(const pair<int, int> &x, const pair<int, int> &y) {
  return x.second > y.second;
}

inline void sortMapbyValue(const hash_map<int, int> &t_map, vector<pair<int, int> > &t_vec) {
  t_vec.clear();

  for (hash_map<int, int>::const_iterator iter = t_map.begin(); iter != t_map.end(); iter++) {
    t_vec.push_back(make_pair(iter->first, iter->second));
  }
  std::sort(t_vec.begin(), t_vec.end(), cmpPairByValue);
}

template<typename T>
T min(T const& a, T const& b, T const& c) {
  return std::min(std::min(a, b), c);
}

inline int edit_distance(const string& A, const string& B) {
  int NA = A.size();
  int NB = B.size();

  vector<vector<int> > M(NA + 1, vector<int>(NB + 1));

  for (int a = 0; a <= NA; ++a)
    M[a][0] = a;

  for (int b = 0; b <= NB; ++b)
    M[0][b] = b;

  for (int a = 1; a <= NA; ++a)
    for (int b = 1; b <= NB; ++b) {
      int x = M[a - 1][b] + 1;
      int y = M[a][b - 1] + 1;
      int z = M[a - 1][b - 1] + (A[a - 1] == B[b - 1] ? 0 : 1);
      M[a][b] = min(x, y, z);
    }

  return M[A.size()][B.size()];
}

inline void replace_char_by_char(string &str, char c1, char c2) {
  string::size_type pos = 0;
  for (; pos < str.size(); ++pos) {
    if (str[pos] == c1) {
      str[pos] = c2;
    }
  }
}

inline void split_bychars(const string& str, vector<string> & vec, const char *sep = " ") { //assert(vec.empty());
  vec.clear();
  string::size_type pos1 = 0, pos2 = 0;
  string word;
  while ((pos2 = str.find_first_of(sep, pos1)) != string::npos) {
    word = str.substr(pos1, pos2 - pos1);
    pos1 = pos2 + 1;
    if (!word.empty())
      vec.push_back(word);
  }
  word = str.substr(pos1);
  if (!word.empty())
    vec.push_back(word);
}

// remove the blanks at the begin and end of string
inline void clean_str(string &str) {
  string blank = " \t\r\n";
  string::size_type pos1 = str.find_first_not_of(blank);
  string::size_type pos2 = str.find_last_not_of(blank);
  if (pos1 == string::npos) {
    str = "";
  } else {
    str = str.substr(pos1, pos2 - pos1 + 1);
  }
}

inline bool my_getline(ifstream &inf, string &line) {
  if (!getline(inf, line))
    return false;
  int end = line.size() - 1;
  while (end >= 0 && (line[end] == '\r' || line[end] == '\n')) {
    line.erase(end--);
  }

  return true;
}

inline void str2uint_vec(const vector<string> &vecStr, vector<unsigned int> &vecInt) {
  vecInt.resize(vecStr.size());
  int i = 0;
  for (; i < vecStr.size(); ++i) {
    vecInt[i] = atoi(vecStr[i].c_str());
  }
}

inline void str2int_vec(const vector<string> &vecStr, vector<int> &vecInt) {
  vecInt.resize(vecStr.size());
  int i = 0;
  for (; i < vecStr.size(); ++i) {
    vecInt[i] = atoi(vecStr[i].c_str());
  }
}

inline void int2str_vec(const vector<int> &vecInt, vector<string> &vecStr) {
  vecStr.resize(vecInt.size());
  int i = 0;
  for (; i < vecInt.size(); ++i) {
    ostringstream out;
    out << vecInt[i];
    vecStr[i] = out.str();
  }
}

inline void join_bystr(const vector<string> &vec, string &str, const string &sep) {
  str = "";
  if (vec.empty())
    return;
  str = vec[0];
  int i = 1;
  for (; i < vec.size(); ++i) {
    str += sep + vec[i];
  }
}

inline void split_bystr(const string &str, vector<string> &vec, const string &sep) {
  vec.clear();
  string::size_type pos1 = 0, pos2 = 0;
  string word;
  while ((pos2 = str.find(sep, pos1)) != string::npos) {
    word = str.substr(pos1, pos2 - pos1);
    pos1 = pos2 + sep.size();
    if (!word.empty())
      vec.push_back(word);
  }
  word = str.substr(pos1);
  if (!word.empty())
    vec.push_back(word);
}

inline void split_pair_vector(const vector<pair<int, string> > &vecPair, vector<int> &vecInt, vector<string> &vecStr) {
  int i = 0;
  vecInt.resize(vecPair.size());
  vecStr.resize(vecPair.size());
  for (; i < vecPair.size(); ++i) {
    vecInt[i] = vecPair[i].first;
    vecStr[i] = vecPair[i].second;
  }
}

inline void split_bychar(const string& str, vector<string>& vec, const char separator = ' ') {
  //assert(vec.empty());
  vec.clear();
  string::size_type pos1 = 0, pos2 = 0;
  string word;
  while ((pos2 = str.find_first_of(separator, pos1)) != string::npos) {
    word = str.substr(pos1, pos2 - pos1);
    pos1 = pos2 + 1;
    if (!word.empty())
      vec.push_back(word);
  }
  word = str.substr(pos1);
  if (!word.empty())
    vec.push_back(word);
}

inline void string2pair(const string& str, pair<string, string>& pairStr, const char separator = '/') {
  string::size_type pos = str.find_last_of(separator);
  if (pos == string::npos) {
    string tmp = str + "";
    clean_str(tmp);
    pairStr.first = tmp;
    pairStr.second = "";
  } else {
    string tmp = str.substr(0, pos);
    clean_str(tmp);
    pairStr.first = tmp;
    tmp = str.substr(pos + 1);
    clean_str(tmp);
    pairStr.second = tmp;
  }
}

inline void convert_to_pair(vector<string>& vecString, vector<pair<string, string> >& vecPair) {
  assert(vecPair.empty());
  int size = vecString.size();
  string::size_type cur;
  string strWord, strPos;
  for (int i = 0; i < size; ++i) {
    cur = vecString[i].find('/');

    if (cur == string::npos) {
      strWord = vecString[i].substr(0);
      strPos = "";
    } else if (cur == vecString[i].size() - 1) {
      strWord = vecString[i].substr(0, cur);
      strPos = "";
    } else {
      strWord = vecString[i].substr(0, cur);
      strPos = vecString[i].substr(cur + 1);
    }

    vecPair.push_back(pair<string, string>(strWord, strPos));
  }
}

inline void split_to_pair(const string& str, vector<pair<string, string> >& vecPair) {
  assert(vecPair.empty());
  vector<string> vec;
  split_bychar(str, vec);
  convert_to_pair(vec, vecPair);
}

inline void chomp(string& str) {
  string white = " \t\n";
  string::size_type pos1 = str.find_first_not_of(white);
  string::size_type pos2 = str.find_last_not_of(white);
  if (pos1 == string::npos || pos2 == string::npos) {
    str = "";
  } else {
    str = str.substr(pos1, pos2 - pos1 + 1);
  }
}

inline int common_substr_len(string str1, string str2) {
  string::size_type minLen;
  if (str1.length() < str2.length()) {
    minLen = str1.length();
  } else {
    minLen = str2.length();
    str1.swap(str2); //make str1 the shorter string
  }

  string::size_type maxSubstrLen = 0;
  string::size_type posBeg;
  string::size_type substrLen;
  string sub;
  for (posBeg = 0; posBeg < minLen; posBeg++) {
    for (substrLen = minLen - posBeg; substrLen > 0; substrLen--) {
      sub = str1.substr(posBeg, substrLen);
      if (str2.find(sub) != string::npos) {
        if (maxSubstrLen < substrLen) {
          maxSubstrLen = substrLen;
        }

        if (maxSubstrLen >= minLen - posBeg - 1) {
          return maxSubstrLen;
        }
      }
    }
  }
  return 0;
}

inline int get_char_index(string& str) {
  assert(str.size() == 2);
  return ((unsigned char) str[0] - 176) * 94 + (unsigned char) str[1] - 161;
}

inline bool is_chinese_char(string& str) {
  if (str.size() != 2) {
    return false;
  }
  int index = ((unsigned char) str[0] - 176) * 94 + (unsigned char) str[1] - 161;
  if (index >= 0 && index < 6768) {
    return true;
  } else {
    return false;
  }
}

inline int find_GB_char(const string& str, string wideChar, int begPos) {
  assert(wideChar.size() == 2 && wideChar[0] < 0); //is a GB char
  int strLen = str.size();

  if (begPos >= strLen) {
    return -1;
  }

  string GBchar;
  for (int i = begPos; i < strLen - 1; i++) {
    if (str[i] < 0) //is a GB char
        {
      GBchar = str.substr(i, 2);
      if (GBchar == wideChar)
        return i;
      else
        i++;
    }
  }
  return -1;
}

inline void split_by_separator(const string& str, vector<string>& vec, const string separator) {
  assert(vec.empty());
  string::size_type pos1 = 0, pos2 = 0;
  string word;

  while ((pos2 = find_GB_char(str, separator, pos1)) != -1) {
    word = str.substr(pos1, pos2 - pos1);
    pos1 = pos2 + separator.size();
    if (!word.empty())
      vec.push_back(word);
  }
  word = str.substr(pos1);
  if (!word.empty())
    vec.push_back(word);
}

//inline void compute_time()
//{
//  clock_t tick = clock();
//  dtype t = (dtype)tick / CLK_TCK;
//  cout << endl << "The time used: " << t << " seconds." << endl;
//}

inline string word(string& word_pos) {
  return word_pos.substr(0, word_pos.find("/"));
}

inline bool is_ascii_string(string& word) {
  for (unsigned int i = 0; i < word.size(); i++) {
    if (word[i] < 0) {
      return false;
    }
  }
  return true;
}

inline bool is_startwith(const string& word, const string& prefix) {
  if (word.size() < prefix.size())
    return false;
  for (unsigned int i = 0; i < prefix.size(); i++) {
    if (word[i] != prefix[i]) {
      return false;
    }
  }
  return true;
}


inline void remove_beg_end_spaces(string &str) {
  clean_str(str);
}

inline void split_bystr(const string &str, vector<string> &vec, const char *sep) {
  split_bystr(str, vec, string(sep));
}

#endif

