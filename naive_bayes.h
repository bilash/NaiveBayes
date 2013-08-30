#ifndef _H_NAIVE_BAYES
#define _H_NAIVE_BAYES

#include <map>
#include <string>
#include <vector>

typedef std::vector<std::string> StringVector;
typedef std::vector<StringVector> TwoDVector;
typedef std::map<std::string, double> MapSD;
typedef std::map<std::string, int> MapSI;
typedef std::map<std::string, MapSD*> MapOfMaps;

const char DELIM = ' ';

class NaiveBayes
{
public:
  NaiveBayes() {}
  ~NaiveBayes() {}
  void set_training_data_file(std::string filepath);
  void add_training_data(std::string data);
  void get_training_data();
  void train();
  std::string classify(const std::string& input_attr);

private:
  double get_prob_for_class(const StringVector& attr, const std::string& class_label);

  std::string training_data_filepath;
  TwoDVector attributes; // attributes data for each input instance
  StringVector classes; // class for each input instance
  MapSD class_probs; // probabilities for each class
  MapSD class_counts; // counts for each class
  MapSI unique_class_attr_counts; // Number of unique pair (class, attr) counts for each class
  int total_unique_class_attr_pairs;
  MapOfMaps class_attr_cooccurence;
  int total_unique_attr_count; // Total individual attribute count
};

#endif /* _H_NAIVE_BAYES */
