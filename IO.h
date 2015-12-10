/*!
 * \file IO.h
 * \brief definitions of I/O functions for LibN3L
 * \author Jie
 */
#ifndef LIBN3L_IO_H_
#define LIBN3L_IO_H_
 #include <stdio.h>
#include "tensor.h"
#include "io.h"
#include "Utiltensor.h"
#include "Utils.h"


class LStream : public IStream {
 public:
  FILE *fp_;
  size_t sz_;

  public:
  LStream(const string &fname, const char *mode) {
    const char *newname = &fname[0];
    Open(newname, mode);
  }
  void Open(const char *fname, const char *mode) {
    fp_ =  FopenCheck(fname, mode);
    fseek(fp_, 0L, SEEK_END);
    sz_ = ftell(fp_);
    fseek(fp_, 0L, SEEK_SET);
  }
  size_t Read(void *ptr, size_t size) {
    return fread(ptr, size, 1, fp_);
  }
  void Write(const void *ptr, size_t size) {
    fwrite(ptr, size, 1, fp_);
  }

  // size_t StringRead(string &sentence) {
  //   char buff[100];
  //   return fread(ptr, size, 1, fp_);
  // }
  // void StringWrite(const string &sentence) {
  //   fputs(sentence,fp_);
  // }

  inline void Close(void) {
    if (fp_ != NULL){
      fclose(fp_); fp_ = NULL;
    }
  }
  inline size_t Size() {
    return sz_;
  }
  virtual ~LStream(void) {
    this->Close();
  }

  inline std::FILE *FopenCheck(const char *fname, const char *flag) {
    std::FILE *fp = fopen(fname, flag);
    Check(fp != NULL, "can not open file \"%s\"\n", fname);
    return fp;
  }


};


template<typename DType, typename TStream>
inline void WriteBinary(TStream &fo, const DType &target) { 
  fo.Write(&target, sizeof(target));
}

template<typename DType, typename TStream>
inline void ReadBinary(TStream &fo, DType &target) { 
  fo.Read(&target, sizeof(DType));
}



template<typename TStream>
inline void WriteString(TStream &fo, const string &target) { 
  int string_size = target.size();
  fo.Write(&string_size, sizeof(string_size));
  if (string_size > 0) {
    int char_size = sizeof(target[0]);
    fo.Write(&char_size, sizeof(char_size));
    for (int idx = 0; idx < string_size; idx++) {
      fo.Write(&target[idx], sizeof(target[idx]));
    }
  }
}

template<typename TStream>
inline void ReadString(TStream &fo, string &target) { 
  int string_size;
  fo.Read(&string_size, sizeof(int));
  if (string_size > 0) {
    int char_size;
    fo.Read(&char_size, sizeof(int));
    char character[string_size];
    for (int idx = 0; idx < string_size; idx++) {
      fo.Read(&character[idx], char_size); 
    }    
    target = string(character, string_size);
    assert(target.size()==string_size);
  }
}


template<typename DType, typename TStream>
inline void WriteVector(TStream &fo, vector<DType> &target) { 
  int vector_size = target.size();
  fo.Write(&vector_size, sizeof(vector_size));
  if (vector_size > 0) {
    int element_size = sizeof(target[0]);
    fo.Write(&element_size, sizeof(element_size));
    for (int idx = 0; idx < vector_size; idx++) {
      fo.Write(&target[idx], sizeof(target[idx]));
      // cout << target[idx] << endl;
    }
  }
}

template<typename DType, typename TStream>
inline void ReadVector(TStream &fo, vector<DType> &target) { 
  int vector_size;
  fo.Read(&vector_size, sizeof(int));
  if (vector_size > 0) {
    int element_size;
    fo.Read(&element_size, sizeof(int));
    target.resize(vector_size);
    for (int idx = 0; idx < vector_size; idx++) {
      fo.Read(&target[idx], element_size); 
      // cout << target[idx] << endl;
    }    
    assert(target.size()== vector_size);
  }
}

template<typename TStream>
inline void WriteVector(TStream &fo, vector<string> &target) { 
  int vector_size = target.size();
  fo.Write(&vector_size, sizeof(vector_size));
  if (vector_size > 0) {
    for (int idx = 0; idx < vector_size; idx++) {
      WriteString(fo, target[idx]);
      // cout << target[idx] << endl;
    }
  }
}

template<typename TStream>
inline void ReadVector(TStream &fo, vector<string> &target) { 
  target.clear();
  int vector_size;
  string tmp_target;
  fo.Read(&vector_size, sizeof(int));
  // cout << "vector_size " << vector_size << endl;
  if (vector_size > 0) {
    for (int idx = 0; idx < vector_size; idx++) {
      ReadString(fo, tmp_target); 
      target.push_back(tmp_target);
      // cout << target[idx] << endl;
    }    
  }
  assert(target.size()== vector_size);
}


template<typename DType, typename TStream>
inline void WriteVector(TStream &fo, NRVec<DType> &target) { 
  int vector_size = target.size();
  WriteBinary(fo, vector_size);
  if (vector_size > 0) {
    for (int idx = 0; idx < vector_size; idx++) {
      WriteBinary(fo, target[idx]);
    }
  }
}

template<typename DType, typename TStream>
inline void ReadVector(TStream &fo, NRVec<DType> &target) { 
  int vector_size;
  ReadBinary(fo, vector_size);
  if (vector_size > 0) {
    target.resize(vector_size);
    for (int idx = 0; idx < vector_size; idx++) {
      ReadBinary(fo, target[idx]);
    }    
    assert(target.size()== vector_size);
  }
}



#endif  // LIBN3L_IO_H_
