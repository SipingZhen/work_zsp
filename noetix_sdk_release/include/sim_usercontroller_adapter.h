#pragma once

// Small stand-ins for the Eigen vectors used by the observation functions.
// Only the sim build includes this file; the robot build keeps its normal types.
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace legged {
enum class WorkMode : uint8_t { STAND, LIE, USERWALK, DEFAULT };
enum class PolicyKind : uint8_t { DEFAULT, WALK, RUN };
using scalar_t = double;
using tensor_element_t = float;

template <typename T>
class SimVector {
 public:
  explicit SimVector(std::size_t size = 0) : values_(size) {}
  std::size_t size() const { return values_.size(); }
  T& operator()(std::size_t i) { return values_.at(i); }
  const T& operator()(std::size_t i) const { return values_.at(i); }
  T& operator[](std::size_t i) { return values_[i]; }
  const T& operator[](std::size_t i) const { return values_[i]; }
  void setZero() { std::fill(values_.begin(), values_.end(), T{}); }

  template <typename U>
  SimVector<U> cast() const {
    if (inserted_ && inserted_ != size())
      throw std::runtime_error("computeObservation() filled fewer values than observations_size");
    SimVector<U> result(size());
    for (std::size_t i = 0; i < size(); ++i) result(i) = static_cast<U>(values_[i]);
    return result;
  }

  SimVector operator*(double scale) const {
    SimVector result(size());
    for (std::size_t i = 0; i < size(); ++i) result(i) = static_cast<T>(values_[i] * scale);
    return result;
  }

  class Slice {
   public:
    Slice(SimVector& parent, std::size_t start, std::size_t count)
        : parent_(parent), start_(start), count_(count) {}
    Slice& operator=(const Slice& source) {
      SimVector copy(source.count_);
      for (std::size_t i = 0; i < source.count_; ++i)
        copy(i) = source.parent_(source.start_ + i);
      return *this = copy;
    }
    template <typename U>
    Slice& operator=(const SimVector<U>& source) {
      if (source.size() != count_) throw std::runtime_error("observation slice size mismatch");
      for (std::size_t i = 0; i < count_; ++i)
        parent_(start_ + i) = static_cast<T>(source(i));
      return *this;
    }
   private:
    SimVector& parent_;
    std::size_t start_, count_;
  };

  Slice segment(std::size_t start, std::size_t count) { return Slice(*this, start, count); }
  Slice head(std::size_t count) { return Slice(*this, 0, count); }
  Slice tail(std::size_t count) { return Slice(*this, size() - count, count); }

  class Inserter {
   public:
    explicit Inserter(SimVector& target) : target_(target) {}
    Inserter& operator,(double value) {
      target_(next_++) = static_cast<T>(value);
      target_.inserted_ = next_;
      return *this;
    }
    template <typename U>
    Inserter& operator,(const SimVector<U>& source) {
      for (std::size_t i = 0; i < source.size(); ++i)
        target_(next_++) = static_cast<T>(source(i));
      target_.inserted_ = next_;
      return *this;
    }
   private:
    SimVector& target_;
    std::size_t next_ = 0;
  };

  friend Inserter operator<<(SimVector& target, const SimVector& first) {
    target.inserted_ = 0;
    Inserter insert(target);
    insert, first;
    return insert;
  }

 private:
  std::vector<T> values_;
  std::size_t inserted_ = 0;
};

using vector_t = SimVector<double>;

struct RobotCfg {
  struct ObsScales {
    double linVel = 1, angVel = 1, dofPos = 1, dofVel = 1;
  } obsScales;
};

struct Proprioception {
  explicit Proprioception(int actions) : jointPos(actions), jointVel(actions) {}
  vector_t jointPos, jointVel, baseAngVel{3}, baseEulerXyz{3};
  vector_t projectedGravity{3};
};

class UserController {
 public:
  UserController(int actions_size, int observation_size, int stack_size)
      : actionsSize_(actions_size), observationSize_(observation_size), stackSize_(stack_size),
        propri_(actions_size),
        lastActions_(actions_size), defaultJointAngles_(actions_size), command_(3),
        proprioHistoryBuffer_(observation_size * stack_size),
        policyObservations_(observation_size * stack_size),
        policyToMotor_(actions_size), motorToPolicy_(actions_size),
        walkObservations_(195), runObservations_(315) {
    for (int i = 0; i < actions_size; ++i) policyToMotor_[i] = i;
  }
  void computeObservation();
  void computeWalkObservation();
  void computeRunObservation();
  void readRemoteCommand(double* command) const;
  void updateRemoteMode();
  int remoteStep(const double* axes, const int* buttons, double* command) {
    for (int i = 0; i < 2; ++i) remote_data.axes[i] = axes[i];
    for (int i = 0; i < 14; ++i) remote_data.button[i] = buttons[i];
    readRemoteCommand(command);
    updateRemoteMode();
    return static_cast<int>(mode_);
  }
  void setDefaults(const double* values) {
    for (int i = 0; i < actionsSize_; ++i) defaultJointAngles_(i) = values[i];
  }
  void setScales(double scalex_value, double scaley_value, double lin_vel,
                 double ang_vel, double dof_pos, double dof_vel) {
    scalex = scalex_value;
    scaley = scaley_value;
    robotconfig.obsScales = {lin_vel, ang_vel, dof_pos, dof_vel};
  }
  void setCommandConfig(const double* values) {
    defaultMinX_ = values[0];
    defaultMaxX_ = values[1];
    defaultXFilterStep_ = values[2];
    defaultYawFilterStep_ = values[3];
    walkMinX_ = values[4];
    walkMaxX_ = values[5];
    runMinX_ = values[6];
    runMaxX_ = values[7];
    runMinAbsX_ = values[8];
    runYawDeadzone_ = values[9];
    runXFilterStep_ = values[10];
  }
  void configureWalk(const int* motor_to_policy, const double* defaults, const double* scales) {
    for (int i = 0; i < 18; ++i) {
      motorToPolicy_[i] = motor_to_policy[i];
      walkConfig_.defaults[i] = defaults[i];
    }
    walkConfig_.scalex = scales[0];
    walkConfig_.scaley = scales[1];
    walkConfig_.linVel = scales[2];
    walkConfig_.angVel = scales[3];
    walkConfig_.dofPos = scales[4];
    walkConfig_.dofVel = scales[5];
    walkConfig_.quat = scales[6];
    walkConfig_.clipObs = scales[7];
  }
  void configureRun(const int* motor_to_policy, const double* defaults, const double* scales) {
    for (int i = 0; i < 18; ++i) {
      motorToPolicy_[i] = motor_to_policy[i];
      runConfig_.defaults[i] = defaults[i];
    }
    runConfig_.scalex = scales[0];
    runConfig_.scaley = scales[1];
    runConfig_.linVel = scales[2];
    runConfig_.angVel = scales[3];
    runConfig_.dofPos = scales[4];
    runConfig_.dofVel = scales[5];
    runConfig_.quat = scales[6];
    runConfig_.clipObs = scales[7];
  }
  void observe(const double* command, const double* angular_velocity,
               const double* roll_pitch, const double* joint_pos,
               const double* joint_vel, const float* last_actions) {
    for (int i = 0; i < 3; ++i) {
      command_(i) = command[i];
      propri_.baseAngVel(i) = angular_velocity[i];
    }
    propri_.baseEulerXyz(0) = roll_pitch[0];
    propri_.baseEulerXyz(1) = roll_pitch[1];
    for (int i = 0; i < actionsSize_; ++i) {
      policyToMotor_[i] = i;
      propri_.jointPos(i) = joint_pos[i];
      propri_.jointVel(i) = joint_vel[i];
      lastActions_(i) = last_actions[i];
    }
    computeObservation();
  }
  void observeWalk(const double* command, const double* angular_velocity,
                   const double* gravity, const double* joint_pos,
                   const double* joint_vel, const float* last_actions) {
    for (int i = 0; i < 3; ++i) {
      command_(i) = command[i];
      propri_.baseAngVel(i) = angular_velocity[i];
      propri_.projectedGravity(i) = gravity[i];
    }
    for (int motor = 0; motor < 18; ++motor) {
      const int policy = motorToPolicy_[motor];
      propri_.jointPos(motor) = joint_pos[policy];
      propri_.jointVel(motor) = joint_vel[policy];
      lastActions_(policy) = last_actions[policy];
    }
    computeWalkObservation();
  }
  void observeRun(const double* command, const double* angular_velocity,
                  const double* gravity, const double* joint_pos,
                  const double* joint_vel, const float* last_actions) {
    for (int i = 0; i < 3; ++i) {
      command_(i) = command[i];
      propri_.baseAngVel(i) = angular_velocity[i];
      propri_.projectedGravity(i) = gravity[i];
    }
    for (int motor = 0; motor < 18; ++motor) {
      const int policy = motorToPolicy_[motor];
      policyToMotor_[policy] = motor;
      propri_.jointPos(motor) = joint_pos[policy];
      propri_.jointVel(motor) = joint_vel[policy];
      lastActions_(policy) = last_actions[policy];
    }
    computeRunObservation();
  }
  const float* observations() const { return policyObservations_.data(); }
  const float* walkObservations() const { return walkObservations_.data(); }
  const float* runObservations() const { return runObservations_.data(); }
  void resetObservations() { first_ = true; }
  void resetWalkObservations() { walkHistoryFirst_ = true; }
  void resetRunObservations() { runHistoryFirst_ = true; }
  void enableWalkPolicy() { walkPolicyReady_ = true; }
  void enableRunPolicy() { runPolicyReady_ = true; }
  int activePolicy() const { return static_cast<int>(activePolicy_); }

 private:
  int actionsSize_, observationSize_, stackSize_;
  double scalex = 1, scaley = 1;
  bool first_ = true;
  bool* isfirstRecObs_ = &first_;
  RobotCfg robotconfig;
  Proprioception propri_;
  vector_t lastActions_, defaultJointAngles_, command_;
  scalar_t defaultMinX_{};
  scalar_t defaultMaxX_{};
  scalar_t walkMinX_{};
  scalar_t walkMaxX_{};
  scalar_t runMinX_{};
  scalar_t runMaxX_{};
  scalar_t runMinAbsX_{};
  scalar_t runYawDeadzone_{};
  scalar_t defaultXFilterStep_{};
  scalar_t defaultYawFilterStep_{};
  scalar_t runXFilterStep_{};
  scalar_t defaultLimitedX_ = 0.0;
  scalar_t defaultLimitedYaw_ = 0.0;
  scalar_t runLimitedX_ = 0.0;
  SimVector<float> proprioHistoryBuffer_;
  std::vector<float> policyObservations_;
  std::vector<int> policyToMotor_, motorToPolicy_;
  struct AuxPolicyConfig {
    std::array<double, 18> defaults{};
    double scalex = 0, scaley = 0, linVel = 0, angVel = 0;
    double dofPos = 0, dofVel = 0, quat = 0, clipObs = 0;
  } walkConfig_, runConfig_;
  std::vector<float> walkObservations_;
  std::vector<float> runObservations_;
  struct { double axes[2]{}; int button[14]{}; } remote_data;
  WorkMode mode_ = WorkMode::DEFAULT;
  PolicyKind activePolicy_ = PolicyKind::DEFAULT;
  bool walkPolicyReady_ = false;
  bool runPolicyReady_ = false;
  bool walkHistoryFirst_ = true;
  bool runHistoryFirst_ = true;
  bool startcontrol = false;
  bool isChangeMode_ = false;
  double standPercent = 0;
};
}

extern "C" {
void sim_command_observation_configure(void* handle, const double* values) {
  static_cast<legged::UserController*>(handle)->setCommandConfig(values);
}

void* sim_observation_create(int actions_size, int observation_size, int stack_size,
                             const double* defaults, double scalex, double scaley,
                             double lin_vel, double ang_vel, double dof_pos, double dof_vel) {
  auto* controller = new legged::UserController(actions_size, observation_size, stack_size);
  controller->setDefaults(defaults);
  controller->setScales(scalex, scaley, lin_vel, ang_vel, dof_pos, dof_vel);
  return controller;
}

void sim_observation_destroy(void* handle) {
  delete static_cast<legged::UserController*>(handle);
}

void sim_observation_reset(void* handle) {
  static_cast<legged::UserController*>(handle)->resetObservations();
}

void sim_walk_observation_configure(void* handle, const int* motor_to_policy,
                                    const double* defaults, const double* scales) {
  static_cast<legged::UserController*>(handle)->configureWalk(motor_to_policy, defaults, scales);
}

void sim_walk_observation_reset(void* handle) {
  static_cast<legged::UserController*>(handle)->resetWalkObservations();
}

void sim_run_observation_configure(void* handle, const int* motor_to_policy,
                                   const double* defaults, const double* scales) {
  static_cast<legged::UserController*>(handle)->configureRun(motor_to_policy, defaults, scales);
}

void sim_run_observation_reset(void* handle) {
  static_cast<legged::UserController*>(handle)->resetRunObservations();
}

int sim_remote_step(void* handle, const double* axes, const int* buttons, double* command) {
  return static_cast<legged::UserController*>(handle)->remoteStep(axes, buttons, command);
}

void sim_enable_walk_policy(void* handle) {
  static_cast<legged::UserController*>(handle)->enableWalkPolicy();
}

void sim_enable_run_policy(void* handle) {
  static_cast<legged::UserController*>(handle)->enableRunPolicy();
}

int sim_active_policy(void* handle) {
  return static_cast<legged::UserController*>(handle)->activePolicy();
}

int sim_observation_step(void* handle, const double* command,
                          const double* angular_velocity, const double* roll_pitch,
                          const double* joint_pos, const double* joint_vel,
                          const float* last_actions) {
  try {
    static_cast<legged::UserController*>(handle)->observe(
        command, angular_velocity, roll_pitch, joint_pos, joint_vel, last_actions);
    return 0;
  } catch (const std::exception&) {
    return -1;
  }
}

int sim_walk_observation_step(void* handle, const double* command,
                              const double* angular_velocity, const double* gravity,
                              const double* joint_pos, const double* joint_vel,
                              const float* last_actions) {
  try {
    static_cast<legged::UserController*>(handle)->observeWalk(
        command, angular_velocity, gravity, joint_pos, joint_vel, last_actions);
    return 0;
  } catch (const std::exception&) {
    return -1;
  }
}

int sim_run_observation_step(void* handle, const double* command,
                             const double* angular_velocity, const double* gravity,
                             const double* joint_pos, const double* joint_vel,
                             const float* last_actions) {
  try {
    static_cast<legged::UserController*>(handle)->observeRun(
        command, angular_velocity, gravity, joint_pos, joint_vel, last_actions);
    return 0;
  } catch (const std::exception&) {
    return -1;
  }
}

const float* sim_observation_data(void* handle) {
  return static_cast<legged::UserController*>(handle)->observations();
}

const float* sim_walk_observation_data(void* handle) {
  return static_cast<legged::UserController*>(handle)->walkObservations();
}

const float* sim_run_observation_data(void* handle) {
  return static_cast<legged::UserController*>(handle)->runObservations();
}
}
