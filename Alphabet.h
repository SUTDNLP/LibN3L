#ifndef _ALPHABET_
#define _ALPHABET_

#include "MyLib.h"
#include "Hash_map.hpp"
#include "IO.h"

/*
	This class serializes feature from string to int.
	Index starts from 0.
*/

/**
 * The basic class of quark class.
 *  @param  std::string        String class name to be used.
 *  @param  int         ID class name to be used.
 *  @author Naoaki Okazaki
 */
class basic_quark {
protected:
  typedef hash_map<std::string, int> StringToId;
  typedef std::vector<std::string> IdToString;

  StringToId m_string_to_id;
  IdToString m_id_to_string;
  bool m_b_fixed;
  int m_size;

public:
  /**
   * Construct.
   */
  basic_quark()
  {
    clear();
  }

  /**
   * Destruct.
   */
  virtual ~basic_quark()
  {
  }

  /**
   * Map a string to its associated ID.
   *  If string-to-integer association does not exist, allocate a new ID.
   *  @param  str         String value.
   *  @return           Associated ID for the string value.
   */
  int operator[](const std::string& str)
  {
    typename StringToId::const_iterator it = m_string_to_id.find(str);
    if (it != m_string_to_id.end()) {
      return it->second;
    } else if (!m_b_fixed){
      int newid = m_size;
      m_id_to_string.push_back(str);
      m_string_to_id.insert(std::pair<std::string, int>(str, newid));
      m_size++;
      return newid;
    }
    else
    {
      return -1;
    }
  }


  /**
   * Convert ID value into the associated string value.
   *  @param  qid         ID.
   *  @param  def         Default value if the ID was out of range.
   *  @return           String value associated with the ID.
   */
  const std::string& from_id(const int& qid, const std::string& def = "") const
  {
    if (qid < 0 || m_size <= qid) {
      return def;
    } else {
      return m_id_to_string[qid];
    }
  }



  /**
   * Convert string value into the associated ID value.
   *  @param  str         String value.
   *  @return           ID if any, otherwise -1.
   */
  int from_string(const std::string& str)
  {
    typename StringToId::const_iterator it = m_string_to_id.find(str);
    if (it != m_string_to_id.end()) {
      return it->second;
    } else if (!m_b_fixed){
      int newid = m_size;
      m_id_to_string.push_back(str);
      m_string_to_id.insert(std::pair<std::string, int>(str, newid));
      m_size++;
      return newid;
    }
    else
    {
      return -1;
    }
  }

  void clear()
  {
    m_string_to_id.clear();
    m_id_to_string.clear();
    m_b_fixed = false;
    m_size = 0;
  }

  void set_fixed_flag(bool bfixed)
  {
    m_b_fixed = bfixed;
  }

  /**
   * Get the number of string-to-id associations.
   *  @return           The number of association.
   */
  size_t size() const
  {
    return m_size;
  }


  void read(std::ifstream &inf)
  {
    clear();
    static string tmp;
    my_getline(inf, tmp);
    chomp(tmp);
    m_size = atoi(tmp.c_str());
    std::vector<std::string> featids;
    for (int i = 0; i < m_size; ++i) {

      my_getline(inf, tmp);
      split_bychars(tmp, featids);
      m_string_to_id[featids[0]] = i;
      assert(atoi(featids[1].c_str()) == i);
    }
  }

  void write(std::ofstream &outf) const
  {
    outf << m_size << std::endl;
    for (int i=0; i<m_size; i++)
    {
      outf << m_id_to_string[i] << i << std::endl;
    }
  }


  void loadModel(LStream &inf)
  { 
    clear();
    string tmp_string;
    int ID;
    ReadBinary(inf, m_size);
    ReadBinary(inf, m_b_fixed);
    for (int i=0; i<m_size; i++)
    { 
      ReadString(inf, tmp_string);
      ReadBinary(inf, ID);
      m_string_to_id[tmp_string] = i;
      m_id_to_string.push_back(tmp_string);
      // cout << tmp_string << " is " << ID << " and " << i << std::endl;
      // cout << m_id_to_string[i] << " is " << ID << " and " << i << std::endl;
      assert(ID == i);
    }

  }

  void writeModel(LStream &outf) const
  { 
    WriteBinary(outf, m_size);
    WriteBinary(outf, m_b_fixed);
    for (int i=0; i<m_size; i++)
    { 
      // cout << m_id_to_string[i] << " is " << i << std::endl;
      WriteString(outf, m_id_to_string[i]);
      WriteBinary(outf, i);
    }
  }
  
};

typedef basic_quark Alphabet;

#endif

