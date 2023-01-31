/* 
From https://github.com/lukun199/SemanticRL
lukun199@gmail.com
*/

#include "Cider.h"

	
Cider::Cider(int n, int sigma): n(n), sigma(sigma) {
}


std::unordered_map<std::string, int> Cider::precook(std::string s){
	std::unordered_map<std::string, int> ret;
	
	// get n-gram
	std::vector<int> milestones_blank{0};
	for(int i=0;i<s.size();i++){
		if(s[i]==' ') milestones_blank.push_back(i);
	}
	milestones_blank.push_back(s.size());

	if(milestones_blank.size()==2){ // no blank
		ret[s]=1;
	}else if(milestones_blank.size()==3){ // one blank
		ret[s.substr(0, milestones_blank[1])]+=1;
		ret[s.substr(milestones_blank[1]+1, milestones_blank[2]-milestones_blank[1]-1)]+=1;
		ret[s]+=1;
	}else{
		for(int n_=1;n_<=4;n_++){
			for(int j=0;j<milestones_blank.size()-n_;j++){
				int start=milestones_blank[j], end=milestones_blank[j+n_];			
				std::string tmp;
				if(start==0) tmp=s.substr(start, end-start);
				else tmp=s.substr(start+1, end-start-1);
				ret[tmp]+=1;	
			}
		}
	}
	
	return ret;
}


std::unordered_map<std::string, int> Cider::cook_test(std::string test){
	return precook(test);
}

std::vector<std::unordered_map<std::string, int>> Cider::cook_refs(SV_type refs){
	std::vector<std::unordered_map<std::string, int>> ret;
	for(std::string s:refs){
		ret.push_back(precook(s));
	}
	return ret;
}


void Cider::compute_doc_freq(void){
	for(std::vector<std::unordered_map<std::string, int>> refs: crefs){
		std::set<std::string> set_;
		for(std::unordered_map<std::string, int> ref:refs){ // dict
			for(std::pair<std::string, int> p: ref)
				set_.insert(p.first);
		}
		for(std::string s:set_) document_frequency[s]+=1;
	}
}

void Cider::cook_append(std::string test, SV_type refs){
	crefs.push_back(cook_refs(refs));
	ctest.push_back(cook_test(test));
}


void Cider::clear(void){
	crefs.clear();
	ctest.clear();
	document_frequency.clear();
}


void element_add_(std::vector<float>& a, std::vector<float> b){
	for(int i=0;i<a.size();i++)
		a[i]+=b[i];
}

std::vector<float> element_add(std::vector<float> a, std::vector<float> b){
	for(int i=0;i<a.size();i++)
		a[i]+=b[i];
	return a;
}

std::vector<float> element_minus(std::vector<float> a, std::vector<float> b){
	for(int i=0;i<a.size();i++)
		a[i]-=b[i];
	return a;
}

float element_mean(std::vector<float> a){
	return accumulate(a.begin(), a.end(), 0.0)/float(a.size());
}

void element_sqrt_(std::vector<float>& a){
	for(int i=0;i<a.size();i++) a[i]=sqrt(a[i]);
}


std::tuple<std::vector<std::unordered_map<std::string, float>>, std::vector<float>, float> Cider::counts2vec(std::unordered_map<std::string, int> cnts){
	// tf-idf
	std::vector<std::unordered_map<std::string, float>> vec(4, std::unordered_map<std::string, float>());
	float length=0;
	std::vector<float> norm(4);
	for(std::pair<std::string, int> p:cnts){
		std::string ngram=p.first;
		int term_freq = p.second;
		float df=log(std::max(float(1.0), float(document_frequency[ngram])));
		int n_=std::count(ngram.begin(), ngram.end(), ' ');
		vec[n_][ngram] = term_freq*(ref_len-df);
		norm[n_]+=pow(vec[n_][ngram], 2);

		if(n_==1) length+=term_freq;
	}
	element_sqrt_(norm);
	return make_tuple(vec, norm, length);
}



std::vector<float> Cider::sim(std::vector<std::unordered_map<std::string, float>> vec_hyp, std::vector<std::unordered_map<std::string, float>> vec_ref, std::vector<float> norm_hyp, std::vector<float> norm_ref, float length_hyp, float length_ref){
	float delta=length_hyp-length_ref;
	std::vector<float> val(4, 0);
	for(int n_=0;n_<4;n_++){
		for(std::pair<std::string, int> p: vec_hyp[n_]){
			std::string ngram=p.first;
			int count = p.second;
			val[n_]+=std::min(vec_hyp[n_][ngram], vec_ref[n_][ngram]) * vec_ref[n_][ngram];
		}
		if(norm_hyp[n_]!=0 && norm_ref[n_] != 0){
			val[n_] /= (norm_hyp[n_]*norm_ref[n_]);
		}
		val[n_] *= exp(-(delta*delta)/(2*sigma*sigma));
	}
	return val;
	
}


std::pair<float, std::vector<float>> Cider::compute_score(SVV_type gt, SV_type res){
	clear();
	for(int i=0;i<gt.size();i++){
		cook_append(res[i], gt[i]);
	}
	
	compute_doc_freq();
	std::vector<float> score;
	score = compute_cider();
	return make_pair(element_mean(score), score);
}

std::vector<float> Cider::compute_cider(void){
	// crefs: list-list-dict
	// ctest: list-dict
	ref_len = log(crefs.size());
	if(crefs.size()==1) ref_len=1.0;
	std::vector<float> scores;
	for(int i=0;i<ctest.size();i++){
		auto test = ctest[i];
		auto refs = crefs[i];
		
		std::tuple<std::vector<std::unordered_map<std::string, float>>, std::vector<float>, float> test_data = counts2vec(test);
		std::vector<float> score(n, 0);
		for(auto ref:refs){  // string
			std::tuple<std::vector<std::unordered_map<std::string, float>>, std::vector<float>, float> ref_data = counts2vec(ref);
			element_add_(score, sim(std::get<0>(test_data), std::get<0>(ref_data), std::get<1>(test_data), std::get<1>(ref_data), std::get<2>(test_data), std::get<2>(ref_data)));
		}
	
		float score_avg=element_mean(score);
		score_avg/=refs.size();
		score_avg*=10;
		scores.push_back(score_avg);
	}
	
	return scores;
}
