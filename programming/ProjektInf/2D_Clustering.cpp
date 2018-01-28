// Definition because of the conflict with std::numeric_limits
#define NOMINMAX

// Reading images
#include "stb_image.h"

// Contains directory_iterator, C++17 / C++14 with experimental
#include <filesystem>

#include <string>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <limits>

// debug
#include <windows.h>


// We can load a PNG with stbi_load(...)


/**
    Takes path to an image file, calculates centralized image moments and returns a
    10-dimensional vector that contains those image moments and represents a tunnel 
    in the image.

    @param filepath Path to an image file
    @return Vector representation of the image
*/
std::vector<double> imageMoments(std::string filepath)
{

    // Image moments needed for centroid calculation
    int M00 = 0, M10 = 0, M01 = 0;

    // Reading pixel intensities and filling image moments needed for centroid calculation
    int width, height, nrChannels;
    unsigned char *img = stbi_load(filepath.c_str(), &width, &height, &nrChannels, 1);
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            int pixel_intensity = static_cast<int>(img[x + y*width]);
            M00 += pixel_intensity;
            M10 += x * pixel_intensity;
            M01 += y * pixel_intensity;
        }
    }

    // Components of the centroid
    int xc = M10 / M00;
    int yc = M01 / M00;

    // Centralized image moments
    double mu00 = M00;
    double mu11 = 0, mu02 = 0, mu20 = 0, mu12 = 0, mu21 = 0, mu03 = 0, mu30 = 0;

    // Calculating centralized image moments
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            int pixel_intensity = static_cast<int>(img[x + y*width]);
            mu11 += (x - xc) * (y - yc) * pixel_intensity;
            mu02 += pow(y - yc, 2) * pixel_intensity;
            mu20 += pow(x - xc, 2) * pixel_intensity;
            mu12 += (x - xc) * pow(y - yc, 2) * pixel_intensity;
            mu21 += pow(x - xc, 2) * (y - yc) * pixel_intensity;
            mu03 += pow(y - yc, 3) * pixel_intensity;
            mu30 += pow(x - xc, 3) * pixel_intensity;
        }
    }

    std::vector<double> vec_representation = { mu00, 0, 0, mu11, mu02, mu20, mu12, mu21, mu03, mu30 };

    // Deallocating resources
    stbi_image_free(img);

    // Debug
    for (int i = 0; i < 10; ++i)
        // std::cout << vec_representation[i] << std::endl;
    return vec_representation;
}

/**
    A simple implementation of k-means clustering
*/
class KMeans
{
private:
    // Number of clusters
    int m_K = 2;
    std::vector< std::vector<double> > m_centroids;
    std::vector< std::vector<double> > m_data;
    std::vector< std::vector< std::vector<double> > > m_clusters;
    std::vector<int> m_current_clusters;

public:
    /**
        Takes path to the data and K, number of wanted clusters.
        Initializes k-means by selecting random initial centroids. 
                
        @param vec_representations: data to cluster
        @param K: number of clusters
    */
    KMeans(std::vector< std::vector<double> > data, int K = 2)
        : m_K{ K }, m_data{ data }, m_centroids{ K }
    {
        for (int i = 0; i < data.size(); ++i)
        {
            for (int j = 0; j < data[i].size(); ++j)
            {
                std::cout << "Vec " << i << ": " << m_data[i][j] << std::endl;
            }
        }
        // Initialize current cluster of every vector
        for (int i = 0; i < data.size(); ++i)
            m_current_clusters.push_back(-1);

        for (int i = 0; i < K; ++i)
        {
            std::vector< std::vector<double> > new_cluster;
            m_clusters.push_back(new_cluster);
            // debug
            // int rand_idx = static_cast<double>( ( (rand() + 1) / static_cast<double>((32 * 1024) ) ) + 0.5 ) * (data.size() - 1);
            std::vector<double> rand_vector = data.at(data.size() - 1);
            // Delete it, so that we don't get identical centroids
            data.erase(data.begin() + (data.size() - 1));
            m_centroids[i] = rand_vector;
        }
    }

    /**
        Reassigns each vector to its closest centroid and returns the
        number of reassignments.
        
        @return num_of_reassigned: number of reassigned vectors
    */
    int reassignment()
    {
        int num_of_reassigned = 0;

        for (int i = 0; i < m_K; ++i)
            m_clusters[i].clear();

        for (int i = 0; i < m_data.size(); ++i)
        {
            // Pair contains nearest centroid (its clusterID) and distance to it
            std::pair<int, double> nearest_centroid = std::make_pair(-2, std::numeric_limits<double>::max());

            for (int clusterID = 0; clusterID < m_clusters.size(); ++clusterID)
            {
                
                // Calculate vector to centroid distance; 
                // To compare distances we don't need to take a sqrt
                double vec_to_centr = 0;
                // std::cout << m_data[i].size() << std::endl; // debug
                for (int j = 0; j < m_data[i].size(); ++j)
                {
                    vec_to_centr += abs(m_data[i][j] - m_centroids[clusterID][j]);
                }

                if (vec_to_centr < nearest_centroid.second)
                {
                    nearest_centroid = std::make_pair(clusterID, vec_to_centr);
                }
            }

            if (nearest_centroid.first != m_current_clusters[i])
            {
                std::cout << "i: " << i << m_current_clusters.size() << std::endl;
                num_of_reassigned += 1;
                m_clusters[nearest_centroid.first].push_back(m_data[i]);
                m_current_clusters[i] = nearest_centroid.first;
            }
        }

        return num_of_reassigned;
    }

    /**
        Recomputes each centroid
    */
    void recomputation()
    {
        std::vector< double > vec_sums(m_centroids[0].size());

        for (int clusterID = 0; clusterID < m_K; ++clusterID)
        {
            for (int dim = 0; dim < m_centroids[0].size(); ++dim)
            {
                vec_sums[dim] = 0;
                for (auto vec : m_clusters[clusterID])
                {
                    vec_sums[dim] += vec[dim];
                }
                vec_sums[dim] = vec_sums[dim] / m_clusters[clusterID].size();
            }
            
            m_centroids[clusterID] = vec_sums;
        }
    }

    /**
        Starts the clustering
    */
    void run()
    {
        int num_of_reassigned = reassignment();
        int count = 0;
        while (num_of_reassigned > 0)
        {
            recomputation();
            count += 1;
            // debug
            std::cout << "Iteration: " << count << " Number of reassignments: " << num_of_reassigned << std::endl;
            num_of_reassigned = reassignment();
        }
    }
};

int main()
{
    // Testing whether it is grayscale

    //std::string filepath = "../../../../../container2.png";
    //int width, height, nrChannels;
    //unsigned char *data = stbi_load(filepath.c_str(), &width, &height, &nrChannels, 0);
    //std::cout << "First: " << static_cast<int>(data[0]) << " Second: " << static_cast<int>(data[1]) << " Third: " << static_cast<int>(data[2]) << 
    //	std::endl;
    //std::cout << "width: " << width << " height: " << height << " nrChannels: " << nrChannels << std::endl;
    //// shows that it is automatically converted to grayscale
    //for (int i = 0; i < 3 * 4; i += 4) 
    //	std::cout << "should be in grayscale: " << 0.2126 * data[i] + 0.7151 * data[i + 1] + data[i + 2] * 0.0722 << std::endl;
    //stbi_image_free(data);

    ////// Loading an image 
    //filepath = "../../../../../container2.png";
    //width, height, nrChannels;
    //unsigned char *img = stbi_load(filepath.c_str(), &width, &height, &nrChannels, 1);
    //std::cout << static_cast<int>(img[0]) << static_cast<int>(img[1]) << std::endl;

    //// Checking length of the image
    //std::string filepath = "../../../../../container2.png";
    //int width, height, nrChannels;
    //unsigned char *img = stbi_load(filepath.c_str(), &width, &height, &nrChannels, 1);
    //for (int j = 0; j < height; ++j)
    //{
    //	for (int i = 0; i < width; ++i)
    //	{
    //		std::cout << img[i + j*width] << " ";
    //	}
    //	std::cout << j << " /n";
    //}
    //std::cout << "Width: " << width << " Height: " << height << std::endl;
        

    //// Calculating vector of image moments which represents an image. This vector is then used in clustering.

    //// TODO

    //// Deallocating resources
    //stbi_image_free(img);
    //std::cin >> width;
    //return 0;

    // Check cwd
    char result[MAX_PATH];
    int bytes = GetModuleFileName(NULL, result, MAX_PATH);
    std::cout << std::string(result, bytes);

    // Seed for pseudo random number generator
    srand(static_cast<unsigned int>(time(0)));

    std::vector< std::vector<double> > vec_representations;

    // Reading images and storing their vector representations in std::vector
    std::string path = "../../../comvi/Images";
    for (auto &file : std::experimental::filesystem::directory_iterator(path))
    {
        if (file.path().extension() == ".png")
        {
            // debug
            std::cout << "Reading file: " << file.path().filename().string() << std::endl;
            vec_representations.push_back(imageMoments(file.path().string()));
        }
    }
    KMeans classifier(vec_representations, 3);
    classifier.run();

    
    // debug: keep console open
    int hold;
    std::cin >> hold;
    return 0;
}


