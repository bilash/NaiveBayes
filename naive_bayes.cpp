#include "naive_bayes.h"

#include <float.h>
#include <math.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <set> // unordered_set will be more efficient here
#include <sstream>

/*
Bayes theorem:

P(Ci|X) = P(X|Ci)P(Ci) / P(X)

A naive bayes classifier returns a class (C = Ci) for which P(Ci|X) is maximum.

Since P(X) is constant for all classes we maximize only the numerator P(X|Ci)P(Ci) 
to obtain the highest probability for a class (P(Ci|X)).

P(X|Ci) = PRODUCT P(Xk|Ci) for k from 1 to n = P(x1|Ci) * P(x2|Ci) * ... * P(xn|Ci)
The above assumes class conditional independence.

P(Xi|Ci) = Count(Xi AND Ci) / Count(Ci)

*/

void NaiveBayes::train()
{
/*
Given training data (attributes value matrix of instances and a vector of
classes for each instance), compute P(Ci) and P(X|Ci).

Assuming input attribute/feature data in the following format (2-D array):

Instance1: Attribute 1, Attribute 2, ... Attribute N
Instance2: Attribute 1, Attribute 2, ... Attribute N
.
.
.
InstanceM: Attribute 1, Attribute 2, ... Attribute N

There is a class vector containing a class for each instance row.

*/

  get_training_data();

  StringVector::const_iterator vci;

  for (vci = classes.begin(); vci != classes.end(); ++vci) {
    if (class_counts.find(*vci) == class_counts.end()) {
      class_counts[*vci] = 1;
    }
    else {
      class_counts[*vci]++;
    }
  }

  // Calculate P(Ci)  
  int total_classes = classes.size();
  for (MapSD::const_iterator map_it = class_counts.begin(); map_it != class_counts.end(); ++map_it) {
    class_probs[map_it->first] = (double) map_it->second / (double) total_classes;
    //std::cout << map_it->second << ": " << total_classes << std::endl;
    //std::cout << "Prob for class " << map_it->first << " is " << class_probs[map_it->first] << std::endl;
  }

  // Save the counts of co-occurence of an attribute with a class
  /*
    class_attr_cooccurence is a map of maps:

    C1: Count(X1), ..., Count(Xn)
    .
    Cm: Count(X1), ..., Count(Xn)
  */
  total_unique_attr_count = 0;
  MapSI attr_counts;
  for (int j = 0; j < attributes.size(); ++j) {
    StringVector sv = attributes[j];
    StringVector::const_iterator it;
    // We only count a co-occurence once per instance
    // current_pairs is a temp map used to indicate if a cooccurence has already
    // been found for the current instance
    MapSI current_pairs;
    for (it = sv.begin(); it != sv.end(); ++it) {
      current_pairs[*it] = 0;
    }

    for (it = sv.begin(); it != sv.end(); ++it) {
      if (class_attr_cooccurence.find(classes[j]) == class_attr_cooccurence.end()) {
        class_attr_cooccurence[classes[j]] = new MapSD; //  TODO: cleanup when no longer needed
        (*class_attr_cooccurence[classes[j]])[*it] = 1;
        current_pairs[*it] = 1;
        //std::cout << "Inserted map and word: " << *it << std::endl;
      }
      else { // The map is there, insert/increment the word count
        MapSD* cur_map = class_attr_cooccurence[classes[j]];
        if ((*cur_map).find(*it) == (*cur_map).end()) {
          (*cur_map)[*it] = 1;
          //std::cout << "Inserted word: " << *it << std::endl;
        }
        if (current_pairs[*it] == 0) {
          // Only increment cooccurence when found in a different instance (row of attributes)
          (*cur_map)[*it]++;
          current_pairs[*it] = 1;
          //std::cout << "Incremented word: " << *it << std::endl;
        }
      }
      if (attr_counts.find(*it) == attr_counts.end()){
        attr_counts[*it] = 1;
        total_unique_attr_count++;
      }
    }
  }

  std::cout << "Total unique attributes count: " << total_unique_attr_count << std::endl;

  // Count unique pair (Ci, attr)
  // Some duplication of code from above, but makes things cleaner
  MapOfMaps class_attr_pair_counts;
  for (int j = 0; j < attributes.size(); ++j) {
    // Count unique pair (Ci, attr)
    if (class_attr_pair_counts.find(classes[j]) == class_attr_pair_counts.end()) {
      class_attr_pair_counts[classes[j]] = new MapSD; // TODO: free when no longer needed
    }

    MapSD* msd = class_attr_pair_counts[classes[j]];
    StringVector sv = attributes[j];
    StringVector::const_iterator it;
    for (it = sv.begin(); it != sv.end(); ++it) {
      if ((*msd).find(*it) == (*msd).end()) {
        (*msd)[*it] = 1;
      }
    }
  }

  // Count unique pair (Ci, attr)
  for (MapOfMaps::const_iterator mci = class_attr_pair_counts.begin();
       mci != class_attr_pair_counts.end(); ++mci) {
    MapSD* msd = mci->second;
    double pair_count = 0;
    for (MapSD::const_iterator ci = (*msd).begin(); ci != (*msd).end(); ++ci) {
      pair_count += ci->second;
    }
    unique_class_attr_counts[mci->first] = pair_count;
    std::cout << mci->first << ": " << pair_count << std::endl;
  }
}

std::string NaiveBayes::classify(const std::string& input_attr)
{
  std::istringstream iss(input_attr);
  std::string token;
  StringVector attr;
  int i = 0;
  while (std::getline(iss, token, DELIM)) {
    attr.push_back(token);
  }

  std::set<std::string> class_labels;
  StringVector::const_iterator vci;
  double max_prob = -DBL_MAX;
  std::string max_class;
  for (vci = classes.begin(); vci != classes.end(); ++vci) {
    if (class_labels.find(*vci) == class_labels.end()) {
      double prob = get_prob_for_class(attr, *vci);
      std::cout << "Log probability for class " << *vci << " and message " <<
        input_attr << " is " << prob << std::endl;
      if (prob > max_prob) {
        max_prob = prob;
        max_class = *vci;
      }
      class_labels.insert(*vci); // This class is done
    }
  }
   
  return max_class;
}

double NaiveBayes::get_prob_for_class(const StringVector& input_attr, const std::string& class_label)
{
  // Calculate P(X|Ci)
  // To calculate P(X|Ci) we need to calculate P(Xi|Ci) first
  // We will use MLE to calculate P(Xi|Ci)
  // P(Xi|Ci) = Count(Xi AND Ci) / Count(Ci AND Yj), Yj = any word in ALL documents
  // Ci = class_label here
  double log_prob_x_given_ci = 0.0; // TODO: We will find log probabilities to avoid underflow
  for (int i = 0; i < input_attr.size(); ++i) {
    std::string attr = input_attr[i];
    double count_xi_ci = 0;
    MapSD* instance_map = class_attr_cooccurence[class_label];
    MapSD::const_iterator mci = instance_map->find(attr);
    if (mci != instance_map->end()) {
      count_xi_ci = mci->second;
    }
    
    // Use Maximum Likelihood Estimate to calculate P(Xi|Ci).
    // Added 1 to numerator and total_attr_count to denominator for Laplace smoothing.
    double temp = (double) unique_class_attr_counts[class_label];
    double prob_xi_given_ci = (double) (count_xi_ci + 1) / (double) (temp + total_unique_attr_count);
    log_prob_x_given_ci +=  log(prob_xi_given_ci);
  }

  return (log(class_probs[class_label]) + log_prob_x_given_ci);
}

void NaiveBayes::get_training_data()
{
  std::ifstream file(training_data_filepath.c_str());
  std::string line;
  while (std::getline(file, line)) {
    std::istringstream iss(line);
    std::string attr;
    StringVector sv;
    while (std::getline(iss, attr, DELIM)) {
      sv.push_back(attr);
    }
    classes.push_back(sv.back()); // The last entry is the class name
    sv.pop_back(); // Remove the class name
    attributes.push_back(sv);
  }
}

void NaiveBayes::set_training_data_file(std::string filepath)
{
  training_data_filepath = filepath;
}

void NaiveBayes::add_training_data(std::string data)
{
  std::ofstream file;
  file.open(training_data_filepath.c_str(), std::ios_base::app); // append mode
  file << data << "\n"; 
  file.close();
}
