#pragma once


#include <tuple>
#include <vector>
#include <string>
#include <set>
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>

class Cider {
public:
  typedef std::vector<std::string> SV_type;
  typedef std::vector<std::vector<std::string>> SVV_type;

  Cider(int n, int sigma);
  std::pair<float, std::vector<float>> compute_score(SVV_type gt, SV_type res);
	
private:
	float ref_len=0;
	int n=4;
	float sigma=6.0;
	std::unordered_map<std::string, int> document_frequency;
	std::vector<std::vector<std::unordered_map<std::string, int>>> crefs;
	std::vector<std::unordered_map<std::string, int>> ctest;


	std::unordered_map<std::string, int> precook(std::string s);
	std::unordered_map<std::string, int> cook_test(std::string test);
	std::vector<std::unordered_map<std::string, int>> cook_refs(SV_type refs);
	void compute_doc_freq(void);
	void cook_append(std::string test, SV_type refs);
	void clear(void);
	std::tuple<std::vector<std::unordered_map<std::string, float>>, std::vector<float>, float> counts2vec(std::unordered_map<std::string, int> cnts);
	std::vector<float> sim(std::vector<std::unordered_map<std::string, float>> vec_hyp, std::vector<std::unordered_map<std::string, float>> vec_ref, std::vector<float> norm_hyp, std::vector<float> norm_ref, float length_hyp, float length_ref);
	std::vector<float> compute_cider(void);
	
	
};

void element_add_(std::vector<float>& a, std::vector<float> b);
std::vector<float> element_add(std::vector<float> a, std::vector<float> b);
std::vector<float> element_minus(std::vector<float> a, std::vector<float> b);
float element_mean(std::vector<float> a);
void element_sqrt_(std::vector<float>& a);
