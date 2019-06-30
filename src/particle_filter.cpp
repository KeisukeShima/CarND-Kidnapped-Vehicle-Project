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

#include "helper_functions.h"

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	
	num_particles = 20;
	
	std::default_random_engine gen;
	
	std::normal_distribution<double> N_x(x,std[0]);
	std::normal_distribution<double> N_y(y,std[1]);
	std::normal_distribution<double> N_theta(theta,std[2]);
	
	for (int i=0; i<num_particles;i++) {	
		Particle particle;
		particle.id=i;
		particle.x=N_x(gen);
		particle.y=N_y(gen);
		particle.theta=N_theta(gen);
		particle.weight=1.0;		
		particles.push_back(particle);
		weights.push_back(particle.weight);
	}
	
	is_initialized=true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {

	std::default_random_engine gen;
	
	std::normal_distribution<double> N_x(0,std_pos[0]);
	std::normal_distribution<double> N_y(0,std_pos[1]);
	std::normal_distribution<double> N_theta(0,std_pos[2]);

	for (int i=0; i<num_particles;i++) {
		
		double new_x;
		double new_y;
		double new_theta;
		
		if (fabs(yaw_rate)<0.001) {
			new_x=particles[i].x+delta_t*velocity*cos(particles[i].theta);
			new_y=particles[i].y+delta_t*velocity*sin(particles[i].theta);
			new_theta=particles[i].theta;
		}
		else {
			new_x=particles[i].x+ velocity/yaw_rate*(sin(particles[i].theta+yaw_rate*delta_t)-sin(particles[i].theta));
			new_y=particles[i].y+ velocity/yaw_rate*(-cos(particles[i].theta+yaw_rate*delta_t)+cos(particles[i].theta));
			new_theta=particles[i].theta+delta_t*yaw_rate;			
		}

		// Add some random noise
		
		particles[i].x=new_x+N_x(gen);
		particles[i].y=new_y+N_y(gen);
		particles[i].theta=new_theta+N_theta(gen);		
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {

	LandmarkObs obs;
	double norm=1/(2*M_PI*std_landmark[0]*std_landmark[1]);		
	weights.clear(); // clear the old weights vector
				
	for (int p=0; p<num_particles;p++) { // for each particle
	
	double weight_upd=1.0; // init weight for each particle
	
		for (int i=0; i<observations.size();i++) { // transform from car co-ordinates to map co-ordinates
		
			obs=observations[i];		
			
			//STEP1: Transformation from car coordinates to map coordinates
			
			double x_obs=cos(particles[p].theta)*obs.x-sin(particles[p].theta)*obs.y+particles[p].x;
			double y_obs=cos(particles[p].theta)*obs.y+sin(particles[p].theta)*obs.x+particles[p].y;				
			
			//STEP2: Associate observation to closest landmark via euclidean distance				
			
			LandmarkObs nearest_landmark;
			double min_dist=sensor_range;
			bool no_landmark=true;
			
			for (int l=0; l<map_landmarks.landmark_list.size();l++) { // calculate rho from various landmarks
				
				double x_l_map=map_landmarks.landmark_list[l].x_f;
				double y_l_map=map_landmarks.landmark_list[l].y_f;
				
				double x2_x1=(x_l_map-x_obs); // Calculate Euclidean distance, rho
				double y2_y1=(y_l_map-y_obs);
				double rho=sqrt(pow(x2_x1,2)+pow(y2_y1,2));
				
				if (rho<min_dist) { // if rho is above sensor range, ignore and find the nearest landmark
				
					nearest_landmark.id=map_landmarks.landmark_list[l].id_i;
					nearest_landmark.x=x_l_map;
					nearest_landmark.y=y_l_map;	
					min_dist=rho;	
					no_landmark=false; // found at least one landmark within sensor range
				}				
			}
			
			// STEP3: Update Weight 
			
			if(no_landmark==false) { // ensure there is at least one landmark within sensor range
				
				double x_x_mu2=pow((x_obs-nearest_landmark.x),2);
				double y_y_mu2=pow((y_obs-nearest_landmark.y),2);
				double exp_term=(x_x_mu2/(2*std_landmark[0]*std_landmark[0])+y_y_mu2/(2*std_landmark[1]*std_landmark[1]));
				
				weight_upd= weight_upd*norm*exp(-exp_term); //norm is calculated before the loop
			}
			else {
				weight_upd=0;			
			}
		}		
		particles[p].weight=weight_upd;
		weights.push_back(weight_upd);
	}
}

void ParticleFilter::resample() {

	std::default_random_engine gen;
	std::vector <Particle> resample_particles;	
	std::discrete_distribution<int> distribution(weights.begin(),weights.end());	
	
	for (int i=0; i<num_particles;i++) {	
		resample_particles.push_back(particles[distribution(gen)]);
	}
	
	particles=resample_particles;	
}

std::string ParticleFilter::getAssociations(Particle best)
{
	std::vector<int> v = best.associations;
	std::stringstream ss;
    copy( v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
    std::string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

std::string ParticleFilter::getSenseCoord(Particle best, std::string coord) {
  std::vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  std::string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}