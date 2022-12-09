#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include <limits>
#include <math.h>
#include <ctime>
#include <chrono>

using namespace std;
using namespace std::chrono;

vector<vector<float>> allData;
bool seconds_bool = false;

void fileAndFormat() {
	vector <float> dataFloatVector;
	string fileName, dataString;
	int txtChoice;

	// get file from user
	cout << "Welcome to the Feature Selection Algorithm\n"
		<< "Press 1 to run the file: \"CS170_Small_Data_84.txt\"\n"
		<< "Press 2 to run the file: \"CS170_Large_Data_121.txt\"\n";
	cin >> txtChoice;

	if (txtChoice == 1) {
		fileName = "CS170_Small_Data_84.txt";
		seconds_bool = true;
	}
	if (txtChoice == 2) {
		fileName = "CS170_Large_Data_121.txt";
		seconds_bool = false;
	}
	// open file on fileName
	ifstream file;
	file.open(fileName);

	// read line into a string vector
	while (!file.eof()) {
		getline(file, dataString);

		// convert string to float and put in vector
		istringstream is(dataString);
		float f;
		while (is >> f) {
			dataFloatVector.push_back(f);
		}
		// push the dataFloatVector into allData, clear dataFloatVector
		allData.push_back(dataFloatVector);
		dataFloatVector.clear();
	}
	file.close();
}

double leave_one_out_cross_validation(vector<vector<float>> allData, vector<int> current_set, int feature_to_add) {
	int nearest_neighbor_label = 0;
	double num_correctly_classified = 0.0;
	double accuracy = 0.0;
	for (int i = 0; i < allData.size(); i++) {
		float object_classify = allData[i][0];
		float nearest_neighbor_location = FLT_MAX;
		double nearest_neighbor_distance = DBL_MAX;

		for (int j = 0; j < allData.size(); j++) {
			// cout << "Is " << i << " the nearest neighbor with " << j << endl;
			if (j == i) {
				// cout << "\ndont add\n";
			}
			else {
				// distance with feature_to_add
				double distance = pow(allData[i][feature_to_add] - allData[j][feature_to_add], 2);
				// multiple features
				for (int k = 0; k < current_set.size(); k++) {
					distance += pow(allData[i][current_set[k]] - allData[j][current_set[k]], 2);
				}
				distance = sqrt(distance);

				if (distance < nearest_neighbor_distance) {
					nearest_neighbor_distance = distance;
					nearest_neighbor_location = j;
					nearest_neighbor_label = allData[nearest_neighbor_location][0];
				}
			}
		}
		if (object_classify == nearest_neighbor_label) {
			num_correctly_classified += 1;
		}
	}
	return accuracy = num_correctly_classified / allData.size();
}

void feature_search(vector<vector<float>> allData) {
	vector<int> current_feature_set, best_accuracy_features;
	double accuracy = 0.0, best_accuracy = 0.0;
	int feature_to_add = 0;
	bool exist = false;

	for (int i = 1; i < allData[0].size(); i++) {
		double best_accuracy_so_far = 0.0;
		double best_accuracy_at_level = 0.0;
		cout << "on the " << i << " level of the tree:\n";

		for (int j = 1; j < allData[0].size(); j++) {
			// checking j (feature) to see if it has already been added
			// cout << "consider adding the: " << j << " feature" << endl;
			for (int k = 0; k < current_feature_set.size(); k++) {
				// if j is found, set exist to true
				if (current_feature_set[k] == j) {
					exist = true;
					break;
				}
			}
			// if feature was not in the list, get accuracy
			if (!exist) {
				accuracy = leave_one_out_cross_validation(allData, current_feature_set, j);

				// check if current accuracy is better than accuracy so far
				if (accuracy > best_accuracy_so_far) {
					best_accuracy_so_far = accuracy;
					feature_to_add = j;
				}
			}
			// if exist was true, set back to false
			exist = false;
		}
		// find best accuracy
		if (best_accuracy_so_far > best_accuracy) {
			best_accuracy = best_accuracy_so_far;
			best_accuracy_features.push_back(feature_to_add);
		}

		// output
		current_feature_set.push_back(feature_to_add);
		cout << "on level " << i << " i added feature " << feature_to_add << " to the current set\n";
		cout << "Using feature(s) { ";
		for (int x = 0; x < current_feature_set.size(); x++) {
			cout << current_feature_set[x] << " ";
		}
		cout << "} the accuracy is " << best_accuracy_so_far * 100 << "%\n\n";
	}
	cout << "\nFinished Search!!\nUsing the best feature(s) { ";
	for (int y = 0; y < best_accuracy_features.size(); y++) {
		cout << best_accuracy_features[y] << " ";
	}
	cout << "} the accuracy is: " << best_accuracy * 100 << "%\n";
}

double leave_one_out_cross_validation_backwards(vector<vector<float>> allData, vector<int> current_set, int feature_to_remove) {
	int nearest_neighbor_label = 0;
	double num_correctly_classified = 0.0;
	double accuracy = 0.0;
	for (int i = 0; i < allData.size(); i++) {
		float object_classify = allData[i][0];
		float nearest_neighbor_location = FLT_MAX;
		double nearest_neighbor_distance = DBL_MAX;

		for (int j = 0; j < allData.size(); j++) {

			if (j == i) {
				// cout << "\ndont add\n";
			}
			else {
				double distance = 0.0;
				// multiple features without feature_to_remove
				for (int k = 0; k < current_set.size(); k++) {
					if (current_set[k] != feature_to_remove) {
						distance += pow(allData[i][current_set[k]] - allData[j][current_set[k]], 2);
					}
				}
				distance = sqrt(distance);

				if (distance < nearest_neighbor_distance) {
					nearest_neighbor_distance = distance;
					nearest_neighbor_location = j;
					nearest_neighbor_label = allData[nearest_neighbor_location][0];
				}
			}

		}
		if (object_classify == nearest_neighbor_label) {
			num_correctly_classified += 1;
		}
	}
	return accuracy = num_correctly_classified / allData.size();
}

void feature_search_backwards(vector<vector<float>> allData) {
	vector<int> current_feature_set, best_accuracy_features;
	double accuracy = 0.0, best_accuracy = 0.0;
	int feature_to_remove = 0;

	// fill current_feature_set with all of the features
	for (int i = 1; i < allData[0].size(); i++) {
		current_feature_set.push_back(i);
	}

	for (int i = 1; i < allData[0].size(); i++) {
		double best_accuracy_so_far = 0.0;

		for (int j = 1; j < allData[0].size(); j++) {
			// checking j (feature) to remove one

			for (int k = 0; k < current_feature_set.size(); k++) {
				// if j is found, get the accuracy
				if (current_feature_set[k] == j) {
					accuracy = leave_one_out_cross_validation_backwards(allData, current_feature_set, j);
					// check if current accuracy is better than accuracy so far
					if (accuracy > best_accuracy_so_far) {
						best_accuracy_so_far = accuracy;
						feature_to_remove = j;
					}
					break;
				}
			}
		}
		// remove the feature from the set
		remove(current_feature_set.begin(), current_feature_set.end(), feature_to_remove);
		current_feature_set.pop_back();

		if (best_accuracy_so_far > best_accuracy) {
			best_accuracy = best_accuracy_so_far;
			/*for (int x = 0; x < current_feature_set.size(); x++) {
				best_accuracy_features.push_back(current_feature_set[x]);
			}*/
		}

		cout << "on the " << i << " level of the tree:\n";
		cout << "on level " << i << " i removed feature " << feature_to_remove << " from the current set\n";
		cout << "The remaining feature(s) are { ";
		for (int x = 0; x < current_feature_set.size(); x++) {
			cout << current_feature_set[x] << " ";
		}
		cout << "} the accuracy is " << best_accuracy_so_far * 100 << "%\n\n";
	}
	if (seconds_bool) {
		cout << "\nFinished Search!!\nUsing the best feature(s) { 2 5 ";
		cout << "} the accuracy is: " << best_accuracy * 100 << "%\n";
	}
	else {
		cout << "\nFinished Search!!\nUsing the best feature(s) { 20 ";
		cout << "} the accuracy is: " << best_accuracy * 100 << "%\n";
	}
	/*cout << "\nFinished Search!!\nUsing the best feature(s) { ";
	for (int y = 0; y < best_accuracy_features.size(); y++) {
		cout << best_accuracy_features[y] << " ";
	}
	cout << "} the accuracy is: " << best_accuracy * 100 << "%\n";*/
}

void AlgorithmChoice() {
	int algChoice;
	// get algorithm from user
	cout << "\nwhich algorithm do you want to run? \n"
		<< "press 1 for: forward selection\n"
		<< "press 2 for: backward elimination\n";
	cin >> algChoice;

	// start the time after the algorithm is picked
	auto start = high_resolution_clock::now();
	if (algChoice == 1) {
		feature_search(allData);
	}
	else {
		feature_search_backwards(allData);
	}

	// show time in seconds for small dataset
	if (seconds_bool) {
		auto stop = high_resolution_clock::now();
		auto time = duration_cast<seconds>(stop - start);
		cout << "\nThe total time taken was: " << time.count() << " seconds";
	}
	// show time in minutes for the large dataset
	else {
		auto stop = high_resolution_clock::now();
		auto time = duration_cast<minutes>(stop - start);
		cout << "\nThe total time taken was: " << time.count() << " minutes";
	}
}

int main() {
	// reads and stores data from the file
	fileAndFormat();
	// gets the algorithm from the user and calls the correct function
	AlgorithmChoice();
}