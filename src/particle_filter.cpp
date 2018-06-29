/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

ParticleFilter::ParticleFilter() {
    num_particles = 500;
    is_initialized = false;
}

ParticleFilter::~ParticleFilter() {}


void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    default_random_engine gen;

    //create a normal distribution for x, y and theta
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    for (int i = 0; i < num_particles; ++i) {
        //push particles into the particle vactor
        Particle temp;

        temp.id = i;
        temp.x = dist_x(gen);
        temp.y = dist_y(gen);
        temp.theta = dist_theta(gen);
        temp.weight = 1.0;

        particles.push_back(temp);
        weights.push_back(temp.weight);
    }
    if (particles.size()==num_particles) {
        is_initialized = true;
        cout << "particles are initialized!" << endl;
    }
    else {
        cout << "Fail to initialize particles" << endl;
    }
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    default_random_engine gen;
    normal_distribution<double> nx(0.0, std_pos[0]);
    normal_distribution<double> ny(0.0, std_pos[1]);
    normal_distribution<double> ntheta(0.0, std_pos[2]);
    for (int i = 0; i < num_particles; ++i) {
        double x_0 = particles[i].x;
        double y_0 = particles[i].y;
        double theta_0 = particles[i].theta;
        double noise_x = nx(gen);
        double noise_y = ny(gen);
        double noise_theta = ntheta(gen);

        //when yaw rate = 0
        if (abs(yaw_rate)<0.0001){
            particles[i].x = x_0 + velocity * delta_t * cos(theta_0) + noise_x;
            particles[i].y = y_0 + velocity * delta_t * sin(theta_0) + noise_y;
            particles[i].theta = theta_0 + noise_theta;
        }
        //when yaw rate != 0
        else{
            particles[i].x = x_0 + velocity/yaw_rate * (sin(theta_0+yaw_rate*delta_t) - sin(theta_0)) + noise_x;
            particles[i].y = y_0 + velocity/yaw_rate * (cos(theta_0) - cos(theta_0+yaw_rate*delta_t)) + noise_y;
            particles[i].theta = theta_0 + yaw_rate * delta_t + noise_theta;
        }
        //cout<<"theta = "<<particles[i].theta<<endl;
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

    for (int i = 0; i < observations.size(); ++i) {
        LandmarkObs nearest = predicted[0];
        float x_obs = observations[i].x;
        float y_obs = observations[i].y;
        for (int j = 1; j < predicted.size(); ++j) {
            if (hypot((nearest.x-x_obs),(nearest.y-y_obs)) > hypot((observations[j].x-x_obs),(observations[j].y-y_obs))){
                nearest = predicted[j];
            }
        }
        observations[i].id = nearest.id;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
    for (int i = 0; i < num_particles; ++i) {
        double w = 1.0;
        std::vector<LandmarkObs> predicts;
        float p_x = particles[i].x;
        float p_y = particles[i].y;
        float p_theta = particles[i].theta;
        particles[i].associations.clear();
        particles[i].sense_x.clear();
        particles[i].sense_y.clear();

        //check if the landmark is in the sensor range
        for (int j = 0; j < map_landmarks.landmark_list.size(); ++j) {
            LandmarkObs predict;
            float landmark_x = map_landmarks.landmark_list[j].x_f;
            float landmark_y =map_landmarks.landmark_list[j].y_f;
            if (dist(landmark_x, landmark_y, p_x, p_y) < sensor_range){
                predict.id = map_landmarks.landmark_list[j].id_i;
                predict.x = landmark_x;
                predict.y = landmark_y;
                predicts.push_back(predict);
            }
        }

        for (int k = 0; k < observations.size(); ++k) {
            //transform to map coordinate system
            LandmarkObs observe_m;
            observe_m.x = p_x + (cos(p_theta)*observations[k].x) - (sin(p_theta)*observations[k].y);
            observe_m.y = p_y + (sin(p_theta)*observations[k].x) + (cos(p_theta)*observations[k].y);

            //associate the measurement with the closest landmark
            LandmarkObs nearest;
            float x_obs = observe_m.x;
            float y_obs = observe_m.y;
            double error = 1.0e10;
            for (int m = 0; m < predicts.size(); ++m) {
                if (dist(predicts[m].x, predicts[m].y, x_obs, y_obs) < error){
                    error = dist(predicts[m].x, predicts[m].y, x_obs, y_obs);
                    nearest = predicts[m];
                }
            }
            observe_m.id = nearest.id;

            //set the parameters of particle
            particles[i].associations.push_back(observe_m.id);
            particles[i].sense_x.push_back(observe_m.x);
            particles[i].sense_y.push_back(observe_m.y);

            //calculate weight
            double weight = calweight(nearest.x, nearest.y, observe_m.x, observe_m.y, std_landmark);
            w *= weight;
        }
        particles[i].weight = w;
        weights[i] = w;
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    //initialize new particle vector
    std::vector<Particle> new_particles;
    int index;

    default_random_engine gen;
    discrete_distribution<int> weight_distribution(weights.begin(), weights.end());

    for (int i = 0; i < num_particles; ++i) {
        index = weight_distribution(gen);
        new_particles.push_back(particles[index]);
    }
    particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
