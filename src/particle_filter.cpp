/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <limits>

#include "helper_functions.h"
#include "map.h"

using std::string;
using std::vector;
using std::normal_distribution;
using std::default_random_engine;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 1250;  // Set the number of particles
  default_random_engine gen;
  // Create a Normal Distribution for x, y, and theta
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
  for (int i = 0; i < num_particles; i++)
  {
    Particle particle;
    particle.id = i;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = 1;
    weights.push_back(particle.weight);
    particles.push_back(particle);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  default_random_engine gen;
  normal_distribution<double> dist_x;
  normal_distribution<double> dist_y;
  normal_distribution<double> dist_theta;
  for (int i = 0; i < num_particles; i++)
  {
    double x0 = particles[i].x;
    double y0 = particles[i].y;
    double theta0 = particles[i].theta;
    double xf = x0 + velocity * (sin(theta0 + yaw_rate * delta_t) - sin(theta0)) / yaw_rate;
    double yf = y0 + velocity * (cos(theta0) - cos(theta0 + yaw_rate * delta_t)) / yaw_rate;
    double thetaf = theta0 + yaw_rate * delta_t;
    dist_x = normal_distribution<double>(xf, std_pos[0]);
    dist_y = normal_distribution<double>(yf, std_pos[1]);
    dist_theta = normal_distribution<double>(thetaf, std_pos[2]);
    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(LandmarkObs& observation, 
                                     const std::vector<Map::single_landmark_s> &map_landmarks) {
  /**
   *  Find the map landmark that is the closest to the 
   *   observed measurement and update the observed measurement with the values of this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  int index = 0;
  double dist;
  dist = std::numeric_limits<double>::max(); //Max double value
  // std::cout << "observation - " << observation.x << observation.y << std::endl;  
  for (int j = 0; j < map_landmarks.size(); j++)
  {
    double tmp_dist = HELPER_FUNCTIONS_H_::dist(observation.x, observation.y, 
                                                map_landmarks[j].x_f, map_landmarks[j].y_f); 
    if (tmp_dist < dist)
    {
      dist = tmp_dist;
      index = j;
    }      
  }
  // std::cout << "dist - " << dist << std::endl;
  observation.x = map_landmarks[index].x_f;
  observation.y = map_landmarks[index].y_f;
  observation.id = map_landmarks[index].id_i;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

  double x_part, y_part, theta_part, updated_weight;
  vector<LandmarkObs> map_observations; // To hold observations xformed to Map coord.
  vector<Map::single_landmark_s> landmark_list = map_landmarks.landmark_list;
  LandmarkObs associated_landmark; //To hold the associated Landmark given an observation.
  bool isnan_observation; // Boolean param to check if Observation had nan values.
  // std::cout << "Start Wt Update" << std::endl;
  for (int i = 0; i < num_particles; i++)
  {
    map_observations.clear(); //This will be cleared for every particle
    isnan_observation = false;
    vector<int> associations; 
    vector<double> sense_x; 
    vector<double> sense_y;
    
    x_part = particles[i].x;
    y_part = particles[i].y;
    theta_part = particles[i].theta;
    updated_weight = 1.0;
    for (int j = 0; j < observations.size(); j++)
    {
      double x_obs = observations[j].x;
      double y_obs = observations[j].y;
      // xform to Map coordinates
      double x_map = x_part + (cos(theta_part) * x_obs) - (sin(theta_part) * y_obs);
      double y_map = y_part + (sin(theta_part) * x_obs) + (cos(theta_part) * y_obs);
      LandmarkObs map_observation;
      map_observation.id = observations[j].id;
      map_observation.x = x_map;
      map_observation.y = y_map;
      map_observations.push_back(map_observation);
    }
    // std::cout << "xformed to Map coord. map_observations size - " << map_observations.size() << std::endl;  
    
    // Update the observation with the nearest landmark information
    for (int k = 0; k < map_observations.size(); k++)
    {
      associated_landmark = map_observations[k];
      if (isnan(associated_landmark.x) | isnan(associated_landmark.y))
      {
        isnan_observation = true; 
      }
      else {
        dataAssociation(associated_landmark, landmark_list);
        associations.push_back(associated_landmark.id);
        sense_x.push_back(associated_landmark.x);
        sense_y.push_back(associated_landmark.y);
        // std::cout << "Orig" << map_observations[k].id << map_observations[k].x << map_observations[k].y << std::endl;  
        // std::cout << "Assoc" << associated_landmark.id << associated_landmark.x << associated_landmark.y << std::endl;  
        updated_weight *= HELPER_FUNCTIONS_H_::multiv_prob(std_landmark[0], std_landmark[1], 
                                                          map_observations[k].x, map_observations[k].y,
                                                          associated_landmark.x, associated_landmark.y);
      }      
    }
    // std::cout << "Got updated wt. - " << updated_weight << std::endl;  
    if (!isnan_observation)
    {
      particles[i].weight = updated_weight;
      weights[i] = updated_weight;
      SetAssociations(particles[i], associations, sense_x, sense_y);
    }
  }
  // std::cout << "End Wt Update" << std::endl;
}

void ParticleFilter::resample() {
  /**
   * Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  default_random_engine gen;
  std::discrete_distribution<> weighted_dist(weights.begin(), weights.end());
  vector<Particle> resampled_particles;
  uint index;
  for (int i = 0; i < num_particles; i++)
  {
    index = weighted_dist(gen);
    resampled_particles.push_back(particles[index]);
  }
  particles = resampled_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}