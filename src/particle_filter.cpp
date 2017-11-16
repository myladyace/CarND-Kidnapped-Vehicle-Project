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
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
    //   x, y, theta and their uncertainties from GPS) and all weights to 1.
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  num_particles = 200;

  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  for (int i=0; i<num_particles; i++){
    Particle temp;
    temp.id = i;
    temp.weight = 1.0;
    temp.x = dist_x(gen) ;
    temp.y = dist_y(gen);
    temp.theta = dist_theta(gen);
    particles.push_back(temp);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // TODO: Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/
  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);

  for (int i=0; i<num_particles; i++){
    double current_x, current_y, current_theta;
    current_x = particles[i].x;
    current_y = particles[i].y;
    current_theta = particles[i].theta;

    double update_x, update_y, update_theta;
    if(fabs(yaw_rate)>0.001){
      update_x = current_x + velocity/yaw_rate*(sin(current_theta+yaw_rate*delta_t) - sin(current_theta)) + dist_x(gen);
      update_y = current_y + velocity/yaw_rate*(cos(current_theta) - cos(current_theta+yaw_rate*delta_t)) + dist_y(gen);
    }
    else{
      update_x = current_x + velocity*cos(current_theta)*delta_t  + dist_x(gen);
      update_y = current_y + velocity*sin(current_theta)*delta_t  + dist_y(gen);
    }
    update_theta = current_theta + yaw_rate*delta_t + dist_theta(gen);

    particles[i].x = update_x;
    particles[i].y = update_y;
    particles[i].theta = update_theta;
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
    //   implement this method and use it as a helper during the updateWeights phase.
  // Predicted are landmarkObs in map and observations are those of cars
  for(int i=0; i<observations.size(); i++){
    double temp = numeric_limits<double>::max();
    int association_id;
    for(int j=0; j<predicted.size(); j++){
      double distance_ = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
      if ( distance_ < temp) {
        temp = distance_;
        association_id = predicted[j].id;
      }
    }
    observations[i].id = association_id;
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

  for(int i=0; i<num_particles; i++){
    double p_x = particles[i].x;
    double p_y = particles[i].y;
    double p_theta = particles[i].theta;

    //First convert map_landmarks to landmarkObs structure
    std::vector<LandmarkObs> predicted;
    for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
      float map_x = map_landmarks.landmark_list[j].x_f;
      float map_y = map_landmarks.landmark_list[j].y_f;
      int map_id = map_landmarks.landmark_list[j].id_i;
      // Only consider landmarks within sensor range of current particle
      if (fabs(map_x - p_x) <= sensor_range && fabs(map_y - p_y) <= sensor_range) {
        LandmarkObs temp_landmark;
        temp_landmark.id = map_id;
        temp_landmark.x = map_x;
        temp_landmark.y = map_y;
        predicted.push_back(temp_landmark);
      }
    }

    //Convert observations to Map's coordinate system from vehicle's coordinate system
    vector<LandmarkObs> observations_in_map;

    for (int j = 0; j < observations.size(); j++) {
      double oim_x = cos(p_theta)*observations[j].x - sin(p_theta)*observations[j].y + p_x;
      double oim_y = sin(p_theta)*observations[j].x + cos(p_theta)*observations[j].y + p_y;
      LandmarkObs temp_landmark;
      temp_landmark.id = observations[j].id;
      temp_landmark.x = oim_x;
      temp_landmark.y = oim_y;
      observations_in_map.push_back(temp_landmark);
    }
    dataAssociation(predicted, observations_in_map);

    // Update weight
    double update_weight = 1.0;

    for (int j = 0; j < observations.size(); j++) {
      double observation_x = observations_in_map[j].x;
      double observation_y = observations_in_map[j].y;
      int landmark_id = observations_in_map[j].id;

      // Get the actual x,y coordinates of associated with current landmark_id
      double predicted_x, predicted_y;
      for (int k = 0; k < predicted.size(); k++) {
        if (predicted[k].id == landmark_id) {
          predicted_x = predicted[k].x;
          predicted_y = predicted[k].y;
          break;
        }
      }

      // calculate weight for current observation and multiply it with total observation weight
      double std_x = std_landmark[0];
      double std_y = std_landmark[1];
      double cur_weight = (1/(2*M_PI*std_x*std_y)) * exp(-(pow(predicted_x-observation_x,2)/(2*pow(std_x, 2)) + (pow(predicted_y-observation_y,2)/(2*pow(std_y, 2)))));
      update_weight *= cur_weight;
    }
    particles[i].weight = update_weight;
  }

}

void ParticleFilter::resample() {
    // TODO: Resample particles with replacement with probability proportional to their weight.
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  std::vector<Particle> new_particles;

  weights.clear();
  // Get weights
  for(int i=0; i< num_particles; i++){
    weights.push_back(particles[i].weight);
  }

  // Get max weight
  double max_weight = *max_element(weights.begin(), weights.end());

  // Start index
  discrete_distribution<int> dist_index(0, num_particles-1);
  int index = dist_index(gen);

  // step
  uniform_real_distribution<double> dist_step(0.0, max_weight);

  double beta=0.0;

  for (int i=0; i<num_particles; i++){
    beta += dist_step(gen) * 2.0;
    while(beta>weights[index]){
      beta -= weights[index];
      index = (index+1) % num_particles;
    }
    new_particles.push_back(particles[index]);
  }

  particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    //Clear the previous associations
    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;
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
