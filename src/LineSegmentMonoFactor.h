#pragma once

#include <math.h>

#include <gtsam/nonlinear/NonlinearFactor.h>

#include <gtsam/geometry/Point2.h>
#include <gtsam/geometry/Cal3_S2.h>

#include "PluckerLine.h"


namespace gtsam {

class LineSegmentFactor : public NoiseModelFactor2<Pose3, PluckerLine> {
    private:
        gtsam::Point2 endpoint1_;
        gtsam::Point2 endpoint2_;
        gtsam::Cal3_S2::shared_ptr K_;
        gtsam::Pose3 body_P_sensor_;
        gtsam::Matrix3 K_line_;

    public:
        LineSegmentFactor(const gtsam::Point2& measured1, const gtsam::Point2& measured2, 
                          const SharedNoiseModel& model,
                          Key poseKey, Key landmarkKey, 
                          const gtsam::Cal3_S2::shared_ptr& K,
                          boost::optional<gtsam::Pose3> body_P_sensor = boost::none) :
                          NoiseModelFactor2<Pose3, PluckerLine>(model, poseKey, landmarkKey), 
                          endpoint1_(measured1), endpoint2_(measured2), 
                          K_(K) //,
                          // body_P_sensor_(body_P_sensor) 
                          {
                            K_line_ << K_->fy(),            0,               0,
                                       0,                 K_->fx(),          0,
                                       -K_->fy() * K_->px(), -K_->fx() * K_->py(), K_->fx() * K_->fy();
                          }

        gtsam::Vector evaluateError(const gtsam::Pose3& pose, const PluckerLine& line,
                             boost::optional<Matrix&> HP = boost::none, boost::optional<Matrix&> HL = boost::none) const override {
            
            gtsam::Matrix64 mlDL;
            gtsam::Matrix6  mlDP;
            gtsam::Vector6 transformed = line.transformTo(pose, HL ? &mlDL : 0, HP ? &mlDP : 0);

            Vector3 m(transformed(0), transformed(1), transformed(2));
            Vector3 l = K_line_ * m;
            double l1 = l.x();
            double l2 = l.y();
            double l3 = l.z();
            double a1 = endpoint1_(0);
            double a2 = endpoint1_(1);
            double b1 = endpoint2_(0);
            double b2 = endpoint2_(1);

            double ln = 1.0 / sqrt(l1*l1 + l2*l2);
            Vector2 error((a1*l1 + a2*l2 + l3) * ln,
                          (b1*l1 + b2*l2 + l3) * ln);

            if (HL || HP) {
                Matrix23 eDl;
                eDl(0, 0) = -(l1*l3-a1*(l2*l2)+a2*l1*l2)/(l1*l1+l2*l2);
                eDl(0, 1) = -(l2*l3-a2*(l1*l1)+a1*l1*l2)/(l1*l1+l2*l2);
                eDl(0, 2) = 1.0;
                eDl(1, 0) = -(l1*l3-b1*(l2*l2)+b2*l1*l2)/(l1*l1+l2*l2);
                eDl(1, 1) = -(l2*l3-b2*(l1*l1)+b1*l1*l2)/(l1*l1+l2*l2);
                eDl(1, 2) = 1.0;
                eDl *= ln;
                Matrix23 eDm = eDl * K_line_;

                if (HL) {
                    Matrix34 mDL = mlDL.topRows<3>();
                    *HL = eDm * mDL;
                }

                if (HP) {
                    Matrix36 mDP = mlDP.topRows<3>();
                    *HP = eDm * mDP;
                }
            }
            return error;
        }
};
} // namespace gtsam

