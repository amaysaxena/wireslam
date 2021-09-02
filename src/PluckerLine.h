#pragma once

#include <gtsam/config.h>

#include <gtsam/geometry/Unit3.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot2.h>
#include <gtsam/base/Lie.h>
#include <gtsam/inference/Symbol.h>

namespace gtsam {

class PluckerLine: public LieGroup<PluckerLine, 4> {
// public:

//   /** Pose Concept requirements */
//   typedef Rot3 Rotation;
//   typedef Point3 Translation;

private:

  Rot3 U_; ///
  Rot2 W_; ///

public:
    PluckerLine() : U_(traits<Rot3>::Identity()), W_(traits<Rot2>::Identity()) {}

    /** Copy constructor */
    PluckerLine(const PluckerLine& line) :
        U_(line.U_), W_(line.W_) {
    }

    /** Construct from U, W */
    PluckerLine(const Rot3& U, const Rot2& W) :
        U_(U), W_(W) {
    }

    /** Construct from [m, v]*/
    PluckerLine(const Vector3& m, const Vector3& v) {
        double w1 = m.norm();
        double w2 = v.norm();
        W_ = Rot2::atan2(w2, w1);

        Vector3 r1 = m / w1;       // first column of U
        Vector3 r2 = v / w2;       // second column of U
        Vector3 r3 = r1.cross(r2); // third column of U

        U_ = Rot3(r1.x(), r2.x(), r3.x(),
                  r1.y(), r2.y(), r3.y(),
                  r1.z(), r2.z(), r3.z());
    }

    gtsam::Rot3 U() const {
        return U_;
    }

    gtsam::Rot2 W() const {
        return W_;
    }

    void print(const std::string& s = "") const {
        cout << (s.empty() ? "" : s);
        cout << "U:\n";
        U_.print();
        cout << "\nW:\n";
        W_.print();
    }

    bool equals(const PluckerLine& other, double tol = 1e-9) const {
        return U_.equals(other.U_, tol) && W_.equals(other.W_, tol);
    }

    gtsam::Vector3 unitNormal(OptionalJacobian<3, 4> H = boost::none) const {
        Matrix3 nDu;
        gtsam::Point3 n = U_.rotate(gtsam::Point3(1.0, 0.0, 0.0), &nDu);
        if (H) {
            H->leftCols<3>() = nDu;
            H->rightCols<1>() = gtsam::Vector3(0.0, 0.0, 0.0);
        }
        return n;
    }

    gtsam::Vector3 unitDirection(OptionalJacobian<3, 4> H = boost::none) const {
        Matrix3 vDu;
        gtsam::Point3 v = U_.rotate(gtsam::Point3(0.0, 1.0, 0.0), &vDu);
        if (H) {
            H->leftCols<3>() = vDu;
            H->rightCols<1>() = gtsam::Vector3(0.0, 0.0, 0.0);
        }
        return v;
    }

    // Returns the first column of W with optional jacobian.
    gtsam::Vector2 vectorNorms(OptionalJacobian<2, 4> H = boost::none) const {
        Matrix21 nDw;
        gtsam::Point2 n = W_.rotate(gtsam::Point2(1.0, 0.0), &nDw);
        if (H) {
            H->leftCols<3>() << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
            H->rightCols<1>() = nDw;
        }
        return n;
    }

    // Returns normal vector (with correct magnitude) with optional jacobian
    // with respect to the plucker-line.
    gtsam::Vector3 normal(OptionalJacobian<3, 4> H = boost::none) const {
        Matrix34 n_hatDL;
        gtsam::Vector3 n_hat = unitNormal(&n_hatDL);

        Matrix24 wDL;
        Vector2 norms = vectorNorms(&wDL);
        double w1 = norms(0); double w2 = norms(1);
        double s = w1 / w2;
        gtsam::Vector3 result(s * n_hat);

        if (H) {
            gtsam::Matrix3 nDn_hat((s * I_3x3));
            gtsam::Matrix32 nDw;
            nDw << n_hat(0) / w2, -(n_hat(0) * w1 / (w2 * w2)),
                   n_hat(1) / w2, -(n_hat(1) * w1 / (w2 * w2)),
                   n_hat(2) / w2, -(n_hat(2) * w1 / (w2 * w2));
            *H = nDn_hat * n_hatDL + nDw * wDL;
        }
        return result;
    }

    // We enforce |v| = 1.
    gtsam::Vector3 direction(OptionalJacobian<3, 4> H = boost::none) const {
        return unitDirection(H);
    }

    // Returns a transformed line in plucker coordinates as [m, v], a 6 vector.
    gtsam::Vector6 transformTo(const gtsam::Pose3 pose, 
        OptionalJacobian<6, 4> HL = boost::none, OptionalJacobian<6, 6> HP = boost::none) const {
        // The [m, l] line transforms according to the jacobian.

        gtsam::Matrix36 RDP;
        gtsam::Matrix36  pDP;
        gtsam::Matrix3 Rt = pose.rotation(HP ? &RDP : OptionalJacobian<3, 6>()).inverse().matrix();
        gtsam::Matrix3 p_hat = rotWedge(pose.translation(HP ? &pDP : OptionalJacobian<3, 6>()));

        gtsam::Matrix34 mDL, vDL;
        gtsam::Vector3 m = normal(HL ? &mDL : OptionalJacobian<3, 4>());
        gtsam::Vector3 v = direction(HL ? &vDL : OptionalJacobian<3, 4>());

        gtsam::Vector3 new_m = Rt * (m - p_hat * v);
        gtsam::Vector3 new_v = Rt * v;

        if (HP) {
            gtsam::Matrix3 new_mDR = rotWedge(gtsam::Point3(new_m.x(), new_m.y(), new_m.z()));
            gtsam::Matrix3 new_mDp = Rt * rotWedge(gtsam::Point3(v.x(), v.y(), v.z()));
            gtsam::Matrix36 new_mDP = new_mDR * RDP + new_mDp * pDP;

            gtsam::Matrix3 new_vDR = rotWedge(Rt * v);
            gtsam::Matrix36 new_vDP = new_vDR * RDP;

            *HP << new_mDP, new_vDP;
        }

        if (HL) {
            gtsam::Matrix3 new_mDm = Rt;
            gtsam::Matrix3 new_mDv = -Rt * p_hat;
            gtsam::Matrix3 new_vDm = Z_3x3;
            gtsam::Matrix3 new_vDv = Rt;

            gtsam::Matrix34 new_mDL = new_mDm * mDL + new_mDv * vDL;
            gtsam::Matrix34 new_vDL = new_vDm * mDL + new_vDv * vDL;

            *HL << new_mDL, new_vDL;
            // *HL << Rt,   -Rt * p_hat, 
            //        Z_3x3, Rt;
        }
        return gtsam::Vector6(new_m.x(), new_m.y(), new_m.z(), new_v.x(), new_v.y(), new_v.z());
    }

    static Matrix3 rotWedge(const Point3& w) {
            double wx = w.x(); double wy = w.y(); double wz = w.z();
            return (Matrix(3, 3) << 0., -wz, wy, wz, 0., -wx, -wy, wx, 0.).finished();
        }

    /// @}
    /// @name Group
    /// @{

    /// identity for group operation
    static PluckerLine identity() {
        return PluckerLine();
    }

    /// inverse transformation with derivatives
    PluckerLine inverse() const {
        return PluckerLine(U_.inverse(), W_.inverse());
    }

    /// compose syntactic sugar
    PluckerLine operator*(const PluckerLine& L) const {
        return PluckerLine(U_ * L.U_, W_ * L.W_);
    }


    /// @}
    /// @name Lie Group
    /// @{

    /// Exponential map at identity.
    static PluckerLine Expmap(const Vector4& delta, OptionalJacobian<4, 4> H = boost::none) {
        Vector3 w = delta.head<3>();
        Vector1 t = delta.tail<1>();
        Matrix3 DU;
        Matrix1 DW;
        Rot3 U = Rot3::Expmap(w, H ? &DU : 0);
        Rot2 W = Rot2::Expmap(t, H ? &DW : 0);

        if (H) {
            H->topLeftCorner<3, 3>() = DU;
            H->bottomRightCorner<1, 1>() = DW;
        }
        return PluckerLine(U, W);
    }

    /// Log map at identity
    static Vector4 Logmap(const PluckerLine& line, OptionalJacobian<4, 4> H = boost::none) {
        Matrix3 DU;
        Matrix1 DW;
        Vector3 du = Rot3::Logmap(line.U_, H ? &DU : 0);
        Vector1 dw = Rot2::Logmap(line.W_, H ? &DW : 0);
        Vector4 result(du(0), du(1), du(2), dw(0));
        if (H) {
            H->topLeftCorner<3, 3>() = DU;
            H->bottomRightCorner<1, 1>() = DW;
        }
        return result;
    }

    // Adjoint map. Block diagonal since this is just a direct product SO(3) x SO(2).
    gtsam::Matrix4 AdjointMap() const {
        gtsam::Matrix4 Ad;
        Ad.topLeftCorner<3, 3>() = U_.AdjointMap();
        Ad.bottomRightCorner<1, 1>() = W_.AdjointMap();
        return Ad;
    }

    // Chart at origin.
    struct ChartAtOrigin {
        static PluckerLine Retract(const Vector4& delta, ChartJacobian H = boost::none) {
            Matrix3 DU;
            Matrix1 DW;
            Vector3 w = delta.head<3>();
            Vector1 t = delta.tail<1>();
            Rot3 U = Rot3::Retract(w, H ? &DU : 0);
            Rot2 W = Rot2::Retract(t, H ? &DW : 0);
            if (H) {
                H->topLeftCorner<3, 3>() = DU;
                H->bottomRightCorner<1, 1>() = DW;
            }
            return PluckerLine(U, W);
        }

        static Vector4 Local(const PluckerLine& line, ChartJacobian H = boost::none) {
            Matrix3 DU;
            Matrix1 DW;
            Vector3 du = Rot3::LocalCoordinates(line.U_, H ? &DU : 0);
            Vector1 dw = Rot2::LocalCoordinates(line.W_, H ? &DW : 0);
            Vector4 result(du(0), du(1), du(2), dw(0));
            if (H) {
                H->topLeftCorner<3, 3>() = DU;
                H->bottomRightCorner<1, 1>() = DW;
            }
            return result;
        }
    };
    using LieGroup<PluckerLine, 4>::inverse;
};

// template <>
// struct traits<PluckerLine> : public internal::LieGroupTraits<PluckerLine> {};

template<>
struct traits<PluckerLine> : public internal::LieGroup<PluckerLine> {};

// template <>
// struct traits<const PluckerLine> : public internal::LieGroup<PluckerLine> {};

// template<> 
// struct traits<Class> : public internal::LieGroupTraits<Class> {}; 

}
