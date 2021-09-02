#pragma once

#include <math.h>

#include <gtsam/nonlinear/NonlinearFactor.h>

#include <gtsam/geometry/Point2.h>
#include <gtsam/geometry/Cal3_S2.h>

#include "PluckerLine.h"


namespace gtsam {

class LinePointFactor : public NoiseModelFactor2<Point3, PluckerLine> {
    public:
        LinePointFactor(Key pointKey, Key lineKey, const SharedNoiseModel& model) :
                          NoiseModelFactor2<Point3, PluckerLine>(model, pointKey, lineKey) {}

        gtsam::Vector evaluateError(const gtsam::Point3& point, const PluckerLine& line,
                             boost::optional<Matrix&> HP = boost::none, boost::optional<Matrix&> HL = boost::none) const override {

            gtsam::Matrix34 mDL, vDL;
            gtsam::Vector3 m = line.normal(HL ? &mDL : OptionalJacobian<3, 4>());
            gtsam::Vector3 v = line.direction(HL ? &vDL : OptionalJacobian<3, 4>());

            gtsam::Vector3 error(v.cross(point) + m);

            if (HL || HP) {
              if (HL) {
                Matrix3 eDv = -PluckerLine::rotWedge(point);
                Matrix3 eDm = I_3x3;
                *HL = eDv * vDL + eDm * mDL;
              }

              if (HP) {
                *HP = PluckerLine::rotWedge(Point3(v.x(), v.y(), v.z()));
              }
            }
            return error;
        }
};
} // namespace gtsam

