#ifndef WSLAM_H
#define WSLAM_H

#include "slam_backend.cc"
#include "json.hpp"
#include "PluckerLine.h"
#include "LineSegmentMonoFactor.h"
#include "LinePointFactor.h"

#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/ImuBias.h>

#include <gtsam/geometry/CalibratedCamera.h>
#include <gtsam/geometry/PinholeCamera.h>
#include <gtsam/geometry/triangulation.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/Cal3_S2Stereo.h>

#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/ProjectionFactor.h>

#include <iostream>
#include <fstream>
#include <unordered_set>
#include <chrono>

using json = nlohmann::json;
using namespace std::chrono;

struct LineMeasurement {
    int id;
    int line_image_index;
    int endpoint1_id;
    int endpoint2_id;
    int endpoint1_image_index;
    int endpoint2_image_index;
    gtsam::Point2 endpoint1;
    gtsam::Point2 endpoint2;
};

struct JunctionMeasurement {
    int id;
    int image_index;
    gtsam::Point2 measurement;
};

Pose3 correct_camera_extrinsic(const json& label_data) {
    std::vector<std::vector<double>> pose = label_data["RT"];
    gtsam::Matrix4 T;
    T << pose[0][0], pose[0][1], pose[0][2], pose[0][3],
         pose[1][0], pose[1][1], pose[1][2], pose[1][3],
         pose[2][0], pose[2][1], pose[2][2], pose[2][3],
         pose[3][0], pose[3][1], pose[3][2], pose[3][3];
    gtsam::Pose3 T_cw(T);
    gtsam::Rot3 correction(1, 0, 0, 0, -1, 0, 0, 0, -1);
    gtsam::Pose3 gt_camera_ext(correction.inverse() * T_cw.rotation(), 
                               correction.inverse() * T_cw.translation());
    return gt_camera_ext;
}

// struct Frame {
//     std::map<int, Point2> junction_measurements;
//     std::map<key, value> map;
// }


template<int window_size, int num_gn_steps, bool use_points, bool use_lines, bool use_constraints>
class WireSLAM {
    public:
        SLAMBackend optimizer;
        std::vector<long> keyframeTimestamps;
        std::unordered_set<size_t> initializedLines;
        std::unordered_set<size_t> initializedJunctions;
        std::unordered_set<size_t> activeLines;
        std::unordered_set<size_t> activeJunctions;

        std::vector<std::map<int, LineMeasurement> > lineMeasurements;
        std::vector<std::map<int, JunctionMeasurement> > junctionMeasurements;

        std::map<int, std::set<int> > junctionsLyingOnLine;
        std::map<int, std::set<int> > junctionsLineIsConstrainedTo;

        json city_data;
        std::vector<json> frameLabels;
        std::map<int, std::set<int> > framesLineIsVisibleFrom;
        std::map<int, std::set<int> > linesVisibleFromFrame;
        std::map<int, std::set<int> > framesJunctionIsVisibleFrom;
        std::map<int, std::set<int> > junctionsVisibleFromFrame;
        Cal3_S2 calibration;

        WireSLAM(const Pose3 prior_pose, const SharedNoiseModel& prior_pose_cov,
                  const Cal3_S2& calibration, const json& city_data): 
                city_data(city_data),
                calibration(calibration)
        {
            optimizer.add_variable(X(0), prior_pose, 6);
            optimizer.add_factor(boost::make_shared<PriorFactor<Pose3> >(X(0), prior_pose, prior_pose_cov));
        }

        void add_image_measurement(const json& label_data, const Pose3& odometry, const SharedNoiseModel& odom_noise) {
            int curr_index = frameLabels.size();
            frameLabels.push_back(label_data);

            if (curr_index > 0) {
                Pose3 new_pose = optimizer.estimate.at<Pose3>(X(curr_index - 1)) * odometry;
                optimizer.add_variable(X(curr_index), new_pose, 6);
                optimizer.add_factor(boost::make_shared<BetweenFactor<Pose3>>(X(curr_index - 1), X(curr_index), 
                                                                              odometry, odom_noise));
            }

            std::map<int, LineMeasurement> lines = get_line_measurements(label_data, curr_index);
            lineMeasurements.push_back(lines);
            std::map<int, JunctionMeasurement> junctions = get_junction_measurements(label_data, curr_index);
            junctionMeasurements.push_back(junctions);


            if (use_points) {
                const auto image_noise_model = noiseModel::Isotropic::Sigma(2, 0.01);
                // Add point-measurement factors for measured junctions.
                std::vector<int> alreadyActive;
                std::vector<int> initializedNow;
                for (auto junc_meas : junctions) {
                    JunctionMeasurement junc = junc_meas.second;
                    if (activeJunctions.find(junc.id) != activeJunctions.end()) {
                        optimizer.add_factor(
                            boost::make_shared<GenericProjectionFactor<Pose3, Point3, Cal3_S2> >(
                                junc.measurement, image_noise_model, X(curr_index), J(junc.id), 
                                boost::make_shared<Cal3_S2>(calibration)));
                    } else {
                        std::vector<int> active_frames_visible_in;
                        for (int frame : framesJunctionIsVisibleFrom[junc.id]) {
                            if (frame >= curr_index - window_size) {
                                active_frames_visible_in.push_back(frame);
                            }
                        }
                        if (active_frames_visible_in.size() > 1) {
                            if (!optimizer.estimate.exists(J(junc.id))) {
                                Point3 junc_estimate = initialize_junction(junc.id, curr_index);
                                optimizer.add_variable(J(junc.id), junc_estimate, 3);
                            }
                            for (int f : active_frames_visible_in) {
                                Point2 measurement = junctionMeasurements[f][junc.id].measurement;
                                optimizer.add_factor(
                                    boost::make_shared<GenericProjectionFactor<Pose3, Point3, Cal3_S2> >(
                                        measurement, image_noise_model, X(f), J(junc.id), 
                                        boost::make_shared<Cal3_S2>(calibration)));
                                activeJunctions.insert(junc.id);
                            }
                        }
                    }
                }
            }

            if (use_lines) {
                const auto image_line_noise_model = noiseModel::Isotropic::Sigma(2, 0.01);
                // Add line-measurement factors for measured lines.
                for (auto line_meas : lines) {
                    LineMeasurement line = line_meas.second;
                    if (activeLines.find(line.id) != activeLines.end()) {
                        optimizer.add_factor(
                            boost::make_shared<LineSegmentFactor>(
                                line.endpoint1, line.endpoint2, image_line_noise_model, 
                                X(curr_index), L(line.id), 
                                boost::make_shared<Cal3_S2>(calibration)));
                    }
                    else {
                        std::vector<int> active_frames_visible_in;
                        for (int frame : framesLineIsVisibleFrom[line.id]) {
                            if (frame >= curr_index - window_size) {
                                active_frames_visible_in.push_back(frame);
                            }
                        }
                        if (active_frames_visible_in.size() > 1) {
                            if (!optimizer.estimate.exists(L(line.id))) {
                                PluckerLine line_estimate = initialize_line(line, curr_index);
                                optimizer.add_variable(L(line.id), line_estimate, 4);
                            }
                            for (int f : active_frames_visible_in) {
                                Point2 measurement1 = lineMeasurements[f][line.id].endpoint1;
                                Point2 measurement2 = lineMeasurements[f][line.id].endpoint2;

                                // cout << "Line projection for line " << line.id << " onto frame " << f << endl;
                                // PluckerLine this_line = optimizer.estimate.at<PluckerLine>(L(line.id));
                                // Matrix3 K_line;
                                // K_line << calibration.fy(), 0, 0, 0, calibration.fx(), 0, 0, 0, calibration.fx() * calibration.fy();
                                // Vector3 l = K_line * this_line.normal();
                                // cout << "Endpoint 1: " << (l(0) * measurement1(0) + l(1) * measurement1(1) + l(2)) / sqrt(l(0)*l(0) + l(1)*l(1)) << endl;
                                // cout << "Endpoint 2: " << (l(0) * measurement2(0) + l(1) * measurement2(1) + l(2)) / sqrt(l(0)*l(0) + l(1)*l(1)) << endl;

                                optimizer.add_factor(
                                    boost::make_shared<LineSegmentFactor>(
                                        measurement1, measurement2, image_line_noise_model, 
                                        X(f), L(line.id), 
                                        boost::make_shared<Cal3_S2>(calibration)));

                                activeLines.insert(line.id);
                            }
                        }
                    }
                }
            }

            if (use_constraints) {
                const auto point_line_noise_model = noiseModel::Isotropic::Sigma(3, 0.01);
                for (auto line : activeLines) {
                    if (junctionsLyingOnLine.find(line) != junctionsLyingOnLine.end()) {
                        for (auto junc : junctionsLyingOnLine[line]) {
                            if (junctionsLineIsConstrainedTo.find(line) == junctionsLineIsConstrainedTo.end()
                                || junctionsLineIsConstrainedTo[line].find(junc) == junctionsLineIsConstrainedTo[line].end()) {

                                optimizer.add_factor(
                                    boost::make_shared<LinePointFactor>(
                                        J(junc), L(line), point_line_noise_model));
                                junctionsLineIsConstrainedTo[line].insert(junc);
                            }
                        }
                    }
                }
            }

            if (curr_index > 0) {
                auto start = high_resolution_clock::now();
                // Perform a measurement update.
                optimizer.gn_step(num_gn_steps);
                auto stop = high_resolution_clock::now(); 
                auto duration = duration_cast<milliseconds>(stop - start);

                cout << "===================================" << endl;
                cout << "Gauss-Newton Time (ms): " << duration.count() << endl;
                cout << "===================================" << endl;
                print_graph();
                cout << "===================================" << endl;
            }

            // Marginalize out oldest robot state and old features, if needed.
            // if (curr_index >= window_size) {
            //     cout << "========= Marginalizing ===========" << endl;
            //     int pose_to_remove = curr_index - window_size;
            //     auto start = high_resolution_clock::now();

            //     std::vector<int> poses_to_remove {pose_to_remove};
            //     std::vector<int> feats_to_remove = features_to_marginalize();
            //     gtsam::KeyVector keys_to_remove;
            //     for (int p : poses_to_remove) {
            //         keys_to_remove.push_back(X(p));
            //         keys_to_remove.push_back(V(p));
            //         keys_to_remove.push_back(B(p));
            //     }
            //     for (int f : feats_to_remove) {
            //         keys_to_remove.push_back(F(f));
            //         activeLandmarks.erase(f);
            //     }
            //     // cout << "Reached here" << endl;
            //     optimizer.marginalize_factors(keys_to_remove, false);
            //     cout << "Removed " << keys_to_remove.size() << " features" << endl;
            //     auto stop = high_resolution_clock::now(); 
            //     auto duration = duration_cast<milliseconds>(stop - start);
            //     cout << "Marginalization Time (ms): " << duration.count() << endl;
            //     cout << "Number of active features: " << activeLandmarks.size() << endl;
            //     cout << "===================================" << endl;
            // }
        }

        void print_graph() {
            optimizer.print_graph("\nCurrent Factor Graph:\n");
        }

        void write_pose_estimate_to_csv(std::string filename) {
            std::ofstream output(filename);
            output << "#timestamp, p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], q_RS_w [], q_RS_x [],"
                      " q_RS_y [], q_RS_z []\n";
            std::string sep = ",";
            for (int i = 0; i < frameLabels.size(); i++) {
                long timestamp_ns = i;
                output << timestamp_ns << sep;
                gtsam::Pose3 pose = optimizer.estimate.at<gtsam::Pose3>(X(i));
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

        void write_structure_estimate_to_json(string filename) {
            std::ofstream output(filename);
            json out;
            if (use_points) {
                std::map<int, std::vector<double> > junc_estimates;
                for (auto junc : framesJunctionIsVisibleFrom) {
                    int j_id = junc.first;
                    if (optimizer.estimate.exists(J(j_id))) {
                        Point3 junc_est = optimizer.estimate.at<Point3>(J(j_id));
                        junc_estimates[j_id] = {junc_est(0), junc_est(1), junc_est(2)};
                    }
                }
                out["junctions"] = junc_estimates;
            }

            if (use_lines) {
                std::map<int, std::vector<double> > line_estimates;
                std::map<int, std::set<int> > line_junctions;
                for (auto line : framesLineIsVisibleFrom) {
                    int l_id = line.first;
                    if (optimizer.estimate.exists(L(l_id))) {
                        PluckerLine line_est = optimizer.estimate.at<PluckerLine>(L(l_id));
                        Vector3 m = line_est.normal();
                        Vector3 v = line_est.direction();
                        line_estimates[l_id] = {m(0), m(1), m(2), v(0), v(1), v(2)};

                        if (junctionsLyingOnLine.find(l_id) != junctionsLyingOnLine.end()) {
                            line_junctions[l_id] = junctionsLyingOnLine[l_id];
                        }
                    }
                }
                out["lines"] = line_estimates;
                out["junctions_on_line"] = line_junctions;
            }
            output << out;
        }

    private:

        std::map<int, LineMeasurement> get_line_measurements(const json& label_data, const int frame_idx) {
            std::map<int, LineMeasurement> result;
            for (int i = 0; i < label_data["line"].size(); i++) {
                auto endpoints = label_data["line"][i];
                int j1 = endpoints[0];
                int j2 = endpoints[1];
                int i1 = label_data["junindex"][j1];
                int i2 = label_data["junindex"][j2];
                if (i1 != -1 && i2 != -1) {
                    gtsam::Point2 e1((double)label_data["junction"][j1][0], -(double)label_data["junction"][j1][1]);
                    gtsam::Point2 e2((double)label_data["junction"][j2][0], -(double)label_data["junction"][j2][1]);
                    int id = label_data["lineidx"][i];
                    result[id] = (LineMeasurement {id, i, i1, i2, j1, j2, e1, e2});
                    
                    junctionsLyingOnLine[id].insert(i1);
                    junctionsLyingOnLine[id].insert(i2);

                    framesLineIsVisibleFrom[id].insert(frame_idx);
                    linesVisibleFromFrame[frame_idx].insert(id);
                }
            }
            return result;
        }

        std::map<int, JunctionMeasurement> get_junction_measurements(const json& label_data, const int frame_idx) {
            std::map<int, JunctionMeasurement> result;
            for (int j = 0; j < label_data["junction"].size(); j++) {
                int j_id = label_data["junindex"][j];
                if (j_id != -1) {
                    gtsam::Point2 e((double)label_data["junction"][j][0], -(double)label_data["junction"][j][1]);
                    framesJunctionIsVisibleFrom[j_id].insert(frame_idx);
                    junctionsVisibleFromFrame[frame_idx].insert(j_id);
                    result[j_id] = JunctionMeasurement{j_id, j, e};
                }
            }
            return result;
        }

        Point3 initialize_junction(int junc_id, int curr_index) {
            json& label_data = frameLabels[curr_index];
            Pose3 gt_camera_ext = correct_camera_extrinsic(label_data);

            gtsam::Pose3 est_camera_pose = optimizer.estimate.at<Pose3>(X(curr_index));

            Point3 world(city_data["vertex"][junc_id][0],
                         city_data["vertex"][junc_id][1], 
                         city_data["vertex"][junc_id][2]);
            if (junc_id == 408) {
                cout << "gt position:" << endl;
                cout << world << endl;
            }
            Point3 cam = gt_camera_ext * world;
            Point3 init = est_camera_pose * cam;

            // cout << "================================" << endl;
            // cout << "Point projection for junction " << junc_id << " onto frame " << curr_index << endl;
            // // PluckerLine this_line = optimizer.estimate.at<PluckerLine>(L(line.id));
            // Matrix3 K_point;
            // K_point << calibration.fx(), 0, 0,  0, calibration.fy(), 0,  0, 0, 1;
            // Point2 measurement = junctionMeasurements[curr_index][junc_id].measurement;
            // cout << "Reproj: " << ((K_point * cam / cam(2)).head<2>() - measurement).norm() << endl;

            return init;
        }

        PluckerLine initialize_line(const LineMeasurement& line, int curr_index) {
            json& label_data = frameLabels[curr_index];
            Pose3 gt_camera_ext = correct_camera_extrinsic(label_data);

            gtsam::Pose3 est_camera_pose = optimizer.estimate.at<Pose3>(X(curr_index));

            Point3 world1(city_data["vertex"][line.endpoint1_id][0], 
                          city_data["vertex"][line.endpoint1_id][1], 
                          city_data["vertex"][line.endpoint1_id][2]);
            Point3 world2(city_data["vertex"][line.endpoint2_id][0], 
                          city_data["vertex"][line.endpoint2_id][1], 
                          city_data["vertex"][line.endpoint2_id][2]);
            Point3 cam1 = gt_camera_ext * world1;
            Point3 cam2 = gt_camera_ext * world2;

            Point3 init1 = est_camera_pose * cam1;
            Point3 init2 = est_camera_pose * cam2;

            // Point3 init1 = cam1;
            // Point3 init2 = cam2;

            Vector3 v = (init1 - init2) / (init1 - init2).norm();
            Vector3 m = init2.cross(v);

            // cout << "================================" << endl;
            // cout << "Line projection for line " << line.id << " onto frame " << curr_index << endl;
            // // PluckerLine this_line = optimizer.estimate.at<PluckerLine>(L(line.id));
            // Matrix3 K_line, K_point;
            // K_line << calibration.fy(), 0, 0, 0, calibration.fx(), 0, 0, 0, calibration.fx() * calibration.fy();
            // K_point << calibration.fx(), 0, 0,  0, calibration.fy(), 0,  0, 0, 1;
            // Vector3 l = K_line * m;
            // Point2 measurement1 = line.endpoint1;
            // Point2 measurement2 = line.endpoint2;
            // cout << "Endpoint 1: " << (l(0) * measurement1(0) + l(1) * measurement1(1) + l(2)) / sqrt(l(0)*l(0) + l(1)*l(1)) << endl;
            // cout << "Endpoint 2: " << (l(0) * measurement2(0) + l(1) * measurement2(1) + l(2)) / sqrt(l(0)*l(0) + l(1)*l(1)) << endl;
            // cout << "Reproj Endpoint 1: " << ((K_point * cam1 / cam1(2)).head<2>() - measurement1).norm() << endl;
            // cout << "Reproj Endpoint 2: " << ((K_point * cam2 / cam2(2)).head<2>() - measurement2).norm() << endl;
            // cout << "Norm v: " << v.norm() << endl;

            return PluckerLine(m, v);
        }
};

#endif
