#ifndef SLAM_BACKEND_H
#define SLAM_BACKEND_H
// #include "slam_backend.h"
#include <Eigen/Dense>

#include <gtsam/base/SymmetricBlockMatrix.h>

#include <gtsam/inference/Key.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/inference/VariableIndex.h>

#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LinearContainerFactor.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/PriorFactor.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtParams.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>

#include <gtsam/linear/GaussianFactorGraph.h>
#include <gtsam/linear/HessianFactor.h>
#include <gtsam/linear/VectorValues.h>

#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Point2.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/sam/BearingRangeFactor.h>

#include <map>
#include <unordered_map>
#include <vector>
#include <math.h>

#include <boost/algorithm/string.hpp>

#include <iostream>
#include <fstream>
#include <vector>
#include <iterator>
#include <string>
#include <algorithm>

using namespace gtsam;
using symbol_shorthand::X;
using symbol_shorthand::L;
using symbol_shorthand::B;
using symbol_shorthand::V;
using symbol_shorthand::F;
using symbol_shorthand::J;
using namespace std;

template <typename K>
std::ostream& operator<< (std::ostream &out, set<K>& data) {
    if (data.size() > 0) {
        out << "{";
        int i = 0;
        for (K k : data) {
            if (i < data.size() - 1) {
                out << k << ", ";
            } else {
                out << k;
            }
            i++;
        }
        out << "}";
    } else {
        out << "{}";
    }
    return out;
}

template <typename K>
std::ostream& operator<< (std::ostream &out, vector<K>& data) {
    if (data.size() > 0) {
        out << "[" << data.at(0);
        for (int i = 1; i < data.size(); i++) {
            out << ", " << data.at(i);
        }
        out << "]";
    } else {
        out << "[]";
    }
    return out;
}

template <typename K>
std::ostream& operator<< (std::ostream &out, deque<K>& data) {
    if (data.size() > 0) {
        out << "[" << data.at(0);
        for (int i = 1; i < data.size(); i++) {
            out << ", " << data.at(i);
        }
        out << "]";
    } else {
        out << "[]";
    }
    return out;
}

template <typename K, typename V>
std::ostream& operator<< (std::ostream &out, map<K, V>& data) {
    if (data.size() > 0) {
        for (auto pair : data) {
            cout << pair.first << ": " << pair.second << endl; 
        }   
    } else {
        cout << "empty";
    }
    return out;
}

class SLAMBackend {
    public:
        SLAMBackend() : size(0) {}

        void add_factor(const NonlinearFactorGraph::sharedFactor& factor) {
            size_t index = factors.size();
            if (availableIndices.empty()) {
                factors.push_back(factor);
            } else {
                index = availableIndices.front();
                availableIndices.pop_front();
                factors.replace(index, factor);
            }
            for (Key k : factor->keys()) {
                keyIndex[k].insert(index);
            }
            size++;
        }

        template<typename T>
        void add_prior(Key key, const T& prior, const SharedNoiseModel& model = nullptr) {
            factors.addPrior(key, prior, model);
        }

        void remove_factor(size_t index) {
            // cout << "==============\n";
            // cout << "KeyIndex:" << endl << keyIndex << endl;
            // cout << "availableIndices: " << availableIndices << endl;
            // print_graph("Graph:\n");
            if (0 <= index && index < factors.size() && factors.at(index)) {
                for (Key k : factors.at(index)->keys()) {
                    auto pair = keyIndex.find(k);
                    if (pair != keyIndex.end()) {
                        (pair->second).erase(index);
                        if ((pair->second).empty()) {
                            keyIndex.erase(k);
                        }
                    }
                }
                factors.remove(index);
                availableIndices.push_back(index);
                size--;
            }
            // cout << "==============\n";
            // cout << "KeyIndex:" << endl << keyIndex << endl;
            // cout << "availableIndices: " << availableIndices << endl;
            // print_graph("Graph:\n");
        }

        template <typename V>
        void add_variable(Key key, const V& value, size_t dim) {
            estimate.insert(key, value);
            manifoldDim[key] = dim;
        }

        // GaussianFactorGraph::shared_ptr 
        void gn_step(size_t num_steps=100, bool LM=false) {
            /*
                Improves estimate by taking Gauss-Newton steps. Takes num_steps steps, or till
                convergence, whichever happens first.
            */
            // GaussianFactorGraph::shared_ptr linear = factors.linearize(estimate);
            // const VectorValues delta = linear->optimize();
            // estimate = estimate.retract(delta);

            // for (int i = 1; i < num_steps; i++) {
            //  linear = factors.linearize(estimate);
            //  const VectorValues delta = linear->optimize();
            //  estimate = estimate.retract(delta);
            // }

            // return linear;
            
            if (!LM) {
                gtsam::GaussNewtonParams parameters;
                parameters.maxIterations = num_steps;
                GaussNewtonOptimizer optimizer(factors, estimate, parameters);
                estimate = optimizer.optimize();
                cout << "All estimated: " << estimate.size() << endl;
            } else {
                gtsam::LevenbergMarquardtParams parameters;
                parameters.maxIterations = num_steps;
                LevenbergMarquardtOptimizer optimizer(factors, estimate, parameters);
                estimate = optimizer.optimize();
                cout << "All estimated: " << estimate.size() << endl;
            }
        }

        void marginalize_factors(const KeyVector& marginalizeKeys, bool update_estimate=true) {
            // Find factors involved in marginalized variables.
            set<size_t> factorIndicesToRemove;
            for (Key marg_key : marginalizeKeys) {
                for (size_t delete_index : keyIndex[marg_key]) {
                    factorIndicesToRemove.insert(delete_index);
                }
            }
            // Add removed factors to a new factor graph.
            NonlinearFactorGraph removedFactors;
            for (size_t i : factorIndicesToRemove) {
                removedFactors.push_back(factors.at(i));
            }
            // Compute new prior over non-marginalized states.
            NonlinearFactorGraph marginalPrior = compute_marginal_factors(removedFactors, marginalizeKeys);
            // Remove marginalized factors.
            for (size_t j : factorIndicesToRemove) {
                remove_factor(j);
            }
            // Re-introduce marginal priors.
            for (size_t f = 0; f < marginalPrior.size(); f++) {
                if (marginalPrior.at(f)) {
                    add_factor(marginalPrior.at(f));
                }
            }
            // Update estimate if required.
            if (update_estimate) {
                gn_step();
            }
        }

        void marginalize_manual_dense(const KeyVector& marginalizeKeys, bool update_estimate=true) {
            // Get variable ordering.
            Ordering ordering;
            marginalization_ordering(marginalizeKeys, ordering);
            ordering.print("ordering:\n");

            // Compute Jacobian.
            GaussianFactorGraph::shared_ptr linear = factors.linearize(estimate);
            // vector<boost::tuple<size_t, size_t, double> > sparse_jacobian = linear->sparseJacobian();
            auto jb_pair = linear->jacobian(ordering);
            Matrix J = (1.0 / sqrt(2.0)) * jb_pair.first;
            Matrix b = -(1.0 / sqrt(2.0)) * jb_pair.second;
            cout << b.squaredNorm() << " " << factors.error(estimate) << endl;

            // find total dimension of marginalized variables.
            size_t marg_dim = total_dim(marginalizeKeys);
            Matrix Jm = J.leftCols(marg_dim);
            Matrix Jk = J.rightCols(J.cols() - marg_dim);

            // Compute new information matrix, mean update, and information vector for the non-marginalized
            // variables.
            Matrix S;
            Vector mu_delta, info_vector;
            marginalization_helper(Jm, Jk, b, S, mu_delta, info_vector);

            // Create new prior factor.
            // First, compute constant term of Hessian factor.
            float constantTerm = mu_delta.transpose() * S * mu_delta;
            // Create augmented Hessian.
            S.conservativeResize(S.rows() + 1, S.rows() + 1);
            S.topRightCorner(info_vector.rows(), 1) = info_vector;
            S.bottomLeftCorner(1, info_vector.rows()) = info_vector.transpose();
            S(S.rows() - 1, S.rows() - 1) = constantTerm;
            S *= 2.0;

            // Define block structure of augmented Hessian. Use same loop to also collect mu_delta into
            // a VectorValues.
            vector<size_t> block_dims;
            KeyVector nonMarginalizedKeys;
            VectorValues delta;
            size_t mu_delta_index = 0;

            for (int i = marginalizeKeys.size(); i < ordering.size(); i++) {
                Key k = ordering[i];
                size_t dim = manifoldDim[k];
                Vector value = mu_delta.segment(mu_delta_index, dim);
                delta.insert(k, value);
                block_dims.push_back(dim);
                nonMarginalizedKeys.push_back(k);
                mu_delta_index += dim;
            }
            block_dims.push_back(1);
            SymmetricBlockMatrix augmentedHessian(block_dims, S);

            // Define new prior factor.
            HessianFactor new_prior_hessian(nonMarginalizedKeys, augmentedHessian);
            LinearContainerFactor new_prior(new_prior_hessian, estimate);

            // Find factor indices that need to be removed.
            set<size_t> factorsToRemove;
            for (Key marg_key : marginalizeKeys) {
                for (size_t delete_index : keyIndex[marg_key]) {
                    factorsToRemove.insert(delete_index);
                }
            }

            // Delete factors.
            for (size_t indexToRemove : factorsToRemove) {
                remove_factor(indexToRemove);
            }

            // Add new prior.
            add_factor(boost::make_shared<LinearContainerFactor>(new_prior_hessian));

            // Update estimate if requested.
            if (update_estimate) {
                estimate = estimate.retract(delta);
            }
        }

        void print_graph(string header) {
            // factors.print(header);
            
            // cout << "Number of keys involved: " << factors.keys().size() << endl;
            // cout << "Involved keys:";
            // gtsam::PrintKeySet(factors.keys());
            // cout << "]" << endl;
        }

        Values get_estimate() {
            return estimate;
        }

        // The current estimate of all variables seen so far. Also serves as the current linearization point.
        Values estimate;

    private:
        // Set of all factors that form the current optimization problem.
        NonlinearFactorGraph factors;
        // Maps variable key to a set of indices into factors for the factors that involve that varibale.
        map<Key, set<size_t> > keyIndex;
        // Maintains a queue of empty spots in the factor graph. When a new graph is added, it is first
        // checked if an empty spot is available to add it to. This way, we do not need to rearrange factor
        // indices each time a factor is removed.
        deque<size_t> availableIndices;
        // Maps variable key to the dimension of its tangent space.
        map<Key, size_t> manifoldDim;
        // Number of terms currently in the optimization problem.
        size_t size;

        void marginalization_ordering(const KeyVector& marginalizeKeys, Ordering& ordering) {
            for (Key k : marginalizeKeys) {
                ordering.push_back(k);
            }
            for (auto kv = keyIndex.begin(); kv != keyIndex.end(); ++kv) {
                Key k = kv->first;
                if (find(marginalizeKeys.begin(), marginalizeKeys.end(), k) == marginalizeKeys.end()){
                    ordering.push_back(kv->first);
                }
            }
        }

        void marginalization_helper(const Matrix& Jm, const Matrix& Jk, const Vector& C, 
                                                    Matrix& S, Vector& mu_delta, Vector& info_vector) {
            /*
                Returns the information matrix \Sigma_K^-1 and the update vector \Delta \mu_K that result from
                marginalization.
            */
            Matrix JmtJm = Jm.transpose() * Jm;
            Matrix I1 = Matrix::Identity(JmtJm.rows(), JmtJm.cols());
            Matrix JmtJm_inv = JmtJm.llt().solve(I1);

            Matrix I2 = Matrix::Identity(Jm.rows(), Jm.rows());
            Matrix A = I2 - (Jm * JmtJm_inv * Jm.transpose());

            S = Jk.transpose() * A * Jk;
            info_vector = -Jk.transpose() * A * C;
            mu_delta = S.llt().solve(info_vector);
        }

        NonlinearFactorGraph compute_marginal_factors(const NonlinearFactorGraph& graph, const KeyVector& marginalizeKeys) {
            if (marginalizeKeys.size() == 0) {
                return graph;
            }
            // Linearize.
            const auto linearized = graph.linearize(estimate);
            // Compute linear marginals.
            // .first is the eliminated Bayes tree, while .second is the remaining factor graph
            const GaussianFactorGraph marginalLinear = *(linearized->eliminatePartialMultifrontal(marginalizeKeys).second);
            // Turn into a prior over the manifold-valued states.
            return LinearContainerFactor::ConvertLinearGraph(marginalLinear, estimate);
        }

        size_t total_dim(const KeyVector& keys) {
            size_t sum = 0;
            for (Key key: keys) {
                sum += manifoldDim[key];
            }
            return sum;
        }
};

#endif
