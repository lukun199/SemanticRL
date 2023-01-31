/* File: example.i */
%module CiderCpp
 
%{
#include "Cider.h"
%}

%include "std_vector.i"
%include "std_string.i"
%include "std_unordered_map.i"

namespace std {
  %template(IntVector) vector<int>;
  %template(IntVectorVector) vector<vector<int>>;
  %template(FloatVector) vector<float>;
  %template(FloatVectorVector) vector<vector<float>>;
  %template(StringVector) vector<string>;
  %template(StringVectorVector) vector<vector<string>>;
  %template(ResultStringPair) std::pair<float, std::vector<float>>;
}

%include "Cider.h"