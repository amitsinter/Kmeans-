// Authors: Idan Abergel, Amit Sinter

#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <list>
#include <chrono>

#define MAX_ITERATIONS 100
#define ENABLE_DEBUG false
using namespace std;

class Point {

private:
    int pointId, clusterId;
    int dimensions;
    vector<double> values;

public:
    Point(int id, string line) {
        dimensions = 0;
        pointId = id;
        values = getLineAsVector(line);
        dimensions = values.size();
        clusterId = 0; //Initially not assigned to any cluster
    }

    // point in dimensions 'dimensions' and all values are 0.
    Point(int dimensions) {
        pointId = -1;
        clusterId = -1;

        this->dimensions = dimensions;
        for (int i = 0; i < dimensions; i++) {
            values.push_back(0.0);
        }
    }

    vector<double> getLineAsVector(string line) {
        std::vector<double> vect;
        std::stringstream ss(line);
        double readedNumber;
        while (ss >> readedNumber) {
            vect.push_back(readedNumber);    // push element to the end of the vector
            if (ss.peek() == ',')            // Returns the next character in the input sequence, without extracting it
                ss.ignore();                // Extracts characters from the input sequence and discards them
        }

        return vect;
    }

    double getDistanceFromPoint(Point p2) {
        double dist = 0;

        if (this->getDimensions() != p2.getDimensions())
            return -1;

        for (int i = 0; i < this->getDimensions(); i++) {
            dist += pow(this->getVal(i) - p2.getVal(i), 2);
        }
        return sqrt(dist);
    }

    vector<double> getValuesVector() {
        return values;
    }

    int getDimensions() {
        return dimensions;
    }

    int getCluster() {
        return clusterId;
    }

    int getID() {
        return pointId;
    }

    void setCluster(int val) {
        clusterId = val;
    }

    double getVal(int pos) {
        return values[pos];
    }

    void setValue(int pos, double val) {
        this->values[pos] = val;
    }
};

class KMeans {
private:
    int K;
    int dimensions{};
    int total_points{};
    double silhouette_score_avg{};

private:

    void writePointstoFile(vector<Point> &dataPoints) {
        if (!ENABLE_DEBUG)
            return;
        ofstream outfile;
        outfile.open("clusters.txt");
        if (outfile.is_open()) {

            for (int i = 0; i < total_points; i++) {
                for (int d = 0; d < dimensions; d++) {
                    outfile << dataPoints[i].getVal(d) << ",";  //Output to file
                }
                outfile << dataPoints[i].getCluster();  //Output to file
                outfile << endl;
            }
            outfile.close();
        } else {
            cout << "Error: Unable to write to clusters.txt";
        }
    }

    void writeCentroidstoFile(vector<Point> &centroids) {
        if (!ENABLE_DEBUG)
            return;
        ofstream outfile;
        outfile.open("Centroids.txt");
        if (outfile.is_open()) {

            for (int i = 0; i < K; i++) {
                for (int d = 0; d < dimensions; d++) {
                    outfile << centroids[i].getVal(d);  //Output to file
                    if (d != dimensions - 1) {
                        outfile << ",";
                    }
                }
                outfile << endl;
            }
            outfile.close();
        } else {
            cout << "Error: Unable to write to clusters.txt";
        }
    }

    void printCentroids(vector<Point> &centroids) {
        if (!ENABLE_DEBUG)
            return;
        for (int k = 0; k < K; k++) {
            cout << "{";
            for (int d = 0; d < dimensions; d++) {
                cout << centroids[k].getVal(d) << ",";
            }
            cout << "}" << endl;
        }

        cout << "------------------------" << endl;
    }

    /* Initialize Centroid by K-Means++ */
    vector<Point> initialize(vector<Point> &data, int k) {
        //Point point, next_centroid;
        double d = 0, temp_dist = 0, max_dist = 0;
        int index = 0, temp_index = 0;
        list<double> distances;

        // Randomly initialize the first centroids
        vector<Point> centroids;
        centroids.push_back(data[rand() % data.size()]);

        //compute remaining k - 1 centroids
        for (int c_id = 0; c_id < k - 1; c_id++) {
            // initialize a list to store distances of data
            // points from nearest centroid
            distances.clear(); // clear distances list
            for (int i = 0; i < data.size(); i++) {
                d = LLONG_MAX;
                // In order to check how point closed to the centroids,
                // get the minimum distance of the point from all centroids
                for (int j = 0; j < centroids.size(); j++) {
                    temp_dist = data[i].getDistanceFromPoint(centroids[j]);
                    d = min(d, temp_dist);
                }
                distances.push_back(d);
            }
            // select data point with maximum distance as our next centroid
            max_dist = 0;
            index = 0;
            temp_index = 0;
            for (double dis : distances) {
                if (max_dist < dis) {
                    max_dist = dis;
                    index = temp_index;
                }
                temp_index++;
            }
            //next_centroid = data[index];
            centroids.push_back(data[index]);
            //plot(data, np.array(centroids))
        }
        return centroids;
    }

    /* Iterate over all points and assign its cluster to the closest centroid.*/
    void assignPointsToCluster(vector<Point> &dataPoints, vector<Point> &centroids) {
        double temp_dist;

        vector<int> pointsInCluster;

        for (int p_id = 0; p_id < total_points; p_id++) {
            double min = LLONG_MAX;
            int tempCluster = -1;

            for (int c_id = 0; c_id < K; c_id++) {
                temp_dist = dataPoints[p_id].getDistanceFromPoint(centroids[c_id]);
                if (temp_dist < min) {
                    min = temp_dist;
                    tempCluster = c_id;
                }
            }
            dataPoints[p_id].setCluster(tempCluster);
        }

    }

    vector<Point> calculateNewCentroids(vector<Point> &dataPoints, const vector<Point> &centroids) {
        vector<Point> temp_centroids;
        // create #centroids zero vectors
        for (int c_id = 0; c_id < K; c_id++) {
            // vector<double> zeroVector(dimensions, 0.0);
            // temp_centroids.push_back(zeroVector);
            auto p = Point(dimensions);
            temp_centroids.push_back(p);
        }

        vector<int> pointsInCluster(K, 0);

        for (int p_id = 0; p_id < total_points; p_id++) {
            auto pointCluster = dataPoints[p_id].getCluster();
            for (int d_id = 0; d_id < dimensions; d_id++) {
                auto tempCentroid = &temp_centroids[pointCluster];
                (*tempCentroid).setValue(d_id,
                                         (*tempCentroid).getVal(d_id) + dataPoints[p_id].getVal(d_id));
            }
            pointsInCluster[pointCluster]++;

        }

        for (int c_id = 0; c_id < K; c_id++) {
            auto tempCentroid = &temp_centroids[c_id];
            for (int d_id = 0; d_id < dimensions; d_id++) {
                auto sum = (*tempCentroid).getVal(d_id);
                (*tempCentroid).setValue(d_id, sum / pointsInCluster[c_id]);
            }
        }

        return temp_centroids;
    }

    vector<double> silhouette(vector<Point> &dataPoints, int k) {
        vector<double> point_silhouette_score(total_points, 0);
        int p_idx = 0;
        for (auto p : dataPoints) {

            // Calculate a
            int cluster = p.getCluster();
            double temp_sum = 0;
            int temp_counter = 0;
            for (auto p_j : dataPoints) {
                // Calc only points in its cluster
                if (p_j.getCluster() != cluster)
                    continue;
                temp_sum += p.getDistanceFromPoint(p_j);
                temp_counter++;
            }
            if (temp_counter == 1) {
                point_silhouette_score[p_idx] = 0;
                continue;
            }
            double a = temp_sum / temp_counter;

            // Calculate b
            double b = LLONG_MAX;
            for (int t_k = 0; t_k < K; t_k++) {
                for (auto p_j : dataPoints) {
                    // Calc only points that are not in its cluster
                    if (p_j.getCluster() == cluster)
                        continue;
                    temp_sum += p.getDistanceFromPoint(p_j);
                    temp_counter++;
                }
                b = min(b, temp_sum / temp_counter);
            }

            if (a < b)
                point_silhouette_score[p_idx] = 1 - a / b;
            else if (a == b)
                point_silhouette_score[p_idx] = 0;
            else if (a > b)
                point_silhouette_score[p_idx] = b / a - 1;
            // cout << "s(" << p_idx << ") = " << point_silhouette_score[p_idx] << endl;
            p_idx++;
        } // for each point

        return point_silhouette_score;
    }

    void kmeansPP(vector<Point> &dataPoints) {
        int iterationNumber = 1;
        auto begin = std::chrono::high_resolution_clock::now();

        // Initializing Centroids
        vector<Point> centroids = initialize(dataPoints, K);
        bool converged = false;

        // Assign each point to its cluster
        while (!converged && iterationNumber++ <= MAX_ITERATIONS) {
            vector<Point> oldCentroids = centroids;
            // printCentroids(oldCentroids);
            assignPointsToCluster(dataPoints, centroids);
            centroids = calculateNewCentroids(dataPoints, centroids);

            if (ENABLE_DEBUG) {
//                writePointstoFile(dataPoints);
//                writeCentroidstoFile(centroids);
//                printCentroids(centroids);
            }

            // Check if centroid changed.
            converged = true;
            for (int c_id = 0; c_id < K; c_id++) {
                if (centroids[c_id].getValuesVector() != oldCentroids[c_id].getValuesVector()) {
                    converged = false;
                    break;
                }
            }
        }
        // Stop measuring time and calculate the elapsed time
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        printf("Time measured: %f seconds.\n", elapsed.count() * 1e-9);

        if (converged) {
            cout << "Converged at " << iterationNumber << endl;
        } else {
            cout << "Reached Max Iteration" << endl;

        }
    }


public:
    KMeans(int K) {
        this->K = K;
    }

    double getSilhouetteScoreAvg() const {
        return silhouette_score_avg;
    }

    void setSilhouetteScoreAvg(double silhouetteScoreAvg) {
        silhouette_score_avg = silhouetteScoreAvg;
    }


    void run(vector<Point> &dataPoints) {
        total_points = dataPoints.size();
        dimensions = dataPoints[0].getDimensions();
        vector<int> pointInCluster;

        kmeansPP(dataPoints);

        auto silhouette_score = silhouette(dataPoints, K);
        double silhouette_score_sum = 0;
        for (auto s : silhouette_score) {
            silhouette_score_sum += s;
        }
        double silhouette_score_avg = silhouette_score_sum / (double) silhouette_score.size();
        setSilhouetteScoreAvg(silhouette_score_avg);
        cout << "S[k=" << K << "] = " << silhouette_score_avg << endl;

    }
};

int main(int argc, char **argv) {
    int k_min = 3;
    int k_max = 13;
    vector<double> k_score(k_max - k_min + 1, 0);

    //Open file for fetching points
    cout << "Start" << endl;
    string filename = "K:\\exampleData.txt";
    ifstream inputFile(filename.c_str());
    ofstream outputfile;

    if (!inputFile.is_open()) {
        cout << "Error: Failed to open file: " << filename << endl;
        return 1;
    }

    int pointId = 1;
    vector<Point> dataPoints;
    string line;

    while (getline(inputFile, line)) {
        Point point(pointId, line);
        dataPoints.push_back(point);

        if (pointId == 1000)
            break;

        pointId++;
    }
    inputFile.close();
    cout << "Reading point finished. Total points = " << pointId - 1 << endl << endl;

    //Return if number of clusters > number of points
    if (dataPoints.size() < k_min) {
        cout << "Error: The number of clusters is greater than the number of points." << endl;
        return 1;
    }

    for (int j = k_min; j <= k_max; j++) {
        KMeans kmeans(j);
        kmeans.run(dataPoints);
        k_score[j] = kmeans.getSilhouetteScoreAvg();
    }
    double max_score = -10;
    int max_score_k = -1;
    int j = 0;

    for (double & s : k_score){
        cout << s << ", ";
        if (s > max_score)
        {
            max_score = s;
            max_score_k = j + k_min;
        }
        j++;
    }

    cout << endl ;
    cout << "The Optimal K is: " << max_score_k << " With a score: " << max_score << endl;
    KMeans kmeans(max_score_k);
    kmeans.run(dataPoints);
    return 0;
}