#include "WireSLAM.h"
#include "json.hpp"

#include <Eigen/Dense>
#include <boost/algorithm/string.hpp>

#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LinearContainerFactor.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/PriorFactor.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>

#include <gtsam/inference/Key.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/inference/VariableIndex.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <iterator>
#include <string>
#include <algorithm>
#include <chrono>
#include <sstream>

using namespace std;
using namespace std::chrono;

using namespace gtsam;
using symbol_shorthand::X;
using symbol_shorthand::L;
using symbol_shorthand::B;
using symbol_shorthand::V;
using symbol_shorthand::F;

using json = nlohmann::json;

void write_gt_pose_estimate_to_csv(std::vector<Pose3> poses, string filename) {
    std::ofstream output(filename);
    output << "#timestamp, p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], q_RS_w [], q_RS_x [],"
              " q_RS_y [], q_RS_z []\n";
    std::string sep = ",";
    for (int i = 0; i < poses.size(); i++) {
        long timestamp_ns = i;
        output << timestamp_ns << sep;
        gtsam::Pose3 pose = poses[i];
        output << pose.translation().x() << sep;
        output << pose.translation().y() << sep;
        output << pose.translation().z() << sep;
        gtsam::Quaternion quat = pose.rotation().toQuaternion();
        output << quat.w() << sep;
        output << quat.x() << sep;
        output << quat.y() << sep;
        output << quat.z() << "\n";
    }
}

struct normal_random_variable {

    normal_random_variable(Eigen::MatrixXd const& covar)
        : normal_random_variable(Eigen::VectorXd::Zero(covar.rows()), covar)
    {}

    normal_random_variable(Eigen::VectorXd const& mean, Eigen::MatrixXd const& covar)
        : mean(mean)
    {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(covar);
        transform = eigenSolver.eigenvectors() * eigenSolver.eigenvalues().cwiseSqrt().asDiagonal();
    }

    Eigen::VectorXd mean;
    Eigen::MatrixXd transform;

    Eigen::VectorXd operator()() const
    {
        static std::mt19937 gen{ std::random_device{}() };
        static std::normal_distribution<> dist;

        return mean + transform * Eigen::VectorXd{ mean.size() }.unaryExpr([&](auto x) { return dist(gen); });
    }
};

int main(int argc, char* argv[]) {
	json city_data;
	std::ifstream city_json("../wireframe/city.json");
	city_json >> city_data;

	int seq = 0;

	json init_label_data;
	std::ifstream init_label_json("../wireframe/new_label/000_" + std::to_string(seq) + "_label_1.json");
	init_label_json >> init_label_data;
	Pose3 corrected_x0 = correct_camera_extrinsic(init_label_data);
	gtsam::Pose3 camera_pose0 = corrected_x0.inverse();

	const Cal3_S2 K(3.125, 3.125, 0, 0, 0);

	auto pose_noise_model = noiseModel::Diagonal::Sigmas(
          (Vector(6) << 0.001, 0.001, 0.001, 0.001, 0.001, 0.001).finished());  // rad,rad,rad,m, m, m

	// WireSLAM<100, 100, false, true> backend_lines(camera_pose0, pose_noise_model, K, city_data);
	// WireSLAM<100, 100, true, false> backend_points(camera_pose0, pose_noise_model, K, city_data);
	// WireSLAM<100, 100, true, true> backend_lines_points(camera_pose0, pose_noise_model, K, city_data);
	WireSLAM<100, 100, true, true, true> backend_lines_points_constr(camera_pose0, pose_noise_model, K, city_data);

	std::string subdir = "../wireframe/seq" + std::to_string(seq) + "/";

	gtsam::Vector6 odom_sigmas(0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001); // rot, trans
	normal_random_variable odom_noise { odom_sigmas.asDiagonal() };

	cout << odom_noise() << endl;

	std::vector<Pose3> gt_poses;
	std::vector<Pose3> pure_odom;
	Pose3 prev_pose = camera_pose0;
	for (int frame = 0; frame < 40; frame++) {
		std::stringstream ss;
		ss << std::setw(3) << std::setfill('0') << frame;
		string filename = ss.str() + "_" + std::to_string(seq);
		cout << filename << endl;

		json label_data;
		std::ifstream label_json("../wireframe/new_label/" + filename + "_label_1.json");
		label_json >> label_data;

		Pose3 corrected = correct_camera_extrinsic(label_data);
		gtsam::Pose3 camera_pose = corrected.inverse();
		Vector6 o_noise = odom_noise();
		// cout << o_noise << endl;
		Pose3 odometry = (prev_pose.inverse() * camera_pose).retract(o_noise);
		// Pose3 odometry = prev_pose.inverse() * camera_pose;

		auto odom_noise_model = noiseModel::Diagonal::Sigmas((Vector(6) << 0.01, 0.01, 0.01, 0.01, 0.01, 0.01).finished());
		
		// backend_lines_points.add_image_measurement(label_data, odometry, odom_noise_model);
		// backend_points.add_image_measurement(label_data, odometry, odom_noise_model);
		// backend_lines.add_image_measurement(label_data, odometry, odom_noise_model);
		backend_lines_points_constr.add_image_measurement(label_data, odometry, odom_noise_model);

		prev_pose = camera_pose;
		gt_poses.push_back(camera_pose);

		if (pure_odom.size() > 0) {
			pure_odom.push_back(pure_odom[pure_odom.size() - 1] * odometry);
		} else {
			pure_odom.push_back(camera_pose);
		}

		// cout << camera_pose << endl;
		// std::vector<LineMeasurement> line_measurements = get_line_measurements(label_data);
	}

	write_gt_pose_estimate_to_csv(pure_odom, "../results/pure_odom.csv");
	write_gt_pose_estimate_to_csv(gt_poses, "../wireframe/seq0/gt_poses.csv");
	// backend_lines_points.write_pose_estimate_to_csv("../results/sc0_lines_points.csv");
	// backend_points.write_pose_estimate_to_csv("../results/sc0_points.csv");
	// backend_lines.write_pose_estimate_to_csv("../results/sc0_lines.csv");
	backend_lines_points_constr.write_structure_estimate_to_json("../results/struct_points_lines_constr_scratch.json");
	backend_lines_points_constr.write_pose_estimate_to_csv("../results/sc0_points_lines_constr_scratch.csv");
}
