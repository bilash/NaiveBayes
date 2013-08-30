#include "naive_bayes.h"

#include <iostream>

int main(int argc, char* argv[])
{
  NaiveBayes nb;

  nb.set_training_data_file(std::string("training.dat"));

  nb.add_training_data("Buy cheap viagra SPAM");
  nb.add_training_data("Buy cheap airlines airlines tickets HAM");
  nb.add_training_data("Dear friend I am the king of Persia king SPAM");
  nb.add_training_data("Hello friend I am from Persia you must be from New York HAM");
  nb.add_training_data("Hi friend how are you doing I love you HAM");
  nb.add_training_data("New York is a big city HAM");

  nb.train();

  std::string class_ = nb.classify(std::string("Buy cheap viagra tickets"));
  std::cout << "Your message is " << class_ << std::endl;

  class_ = nb.classify(std::string("Hello friend how are you"));
  std::cout << "Your message is " << class_ << std::endl;

  class_ = nb.classify(std::string("Dhaka is a big city"));
  std::cout << "Your message is " << class_ << std::endl;

  return 0;
}
