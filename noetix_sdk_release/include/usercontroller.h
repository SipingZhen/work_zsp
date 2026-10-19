#ifndef UserController_H
#define UserController_H

#include "controllerbase.h"
#include "common.h"
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <chrono>
#include <eigen3/Eigen/Dense>
#include <atomic>
#include <array>
#ifdef RK3588
#include "rknn_api.h"
#endif

namespace legged
{

enum class WorkMode : uint8_t
{
  STAND,
  LIE,
  USERWALK,
  DEFAULT
};

enum class PolicyKind : uint8_t { DEFAULT, WALK };

  struct RobotCfg
  {
    struct ControlCfg
    {
      std::map<std::string, float> stiffness;
      std::map<std::string, float> damping;
      std::map<std::string, float> actionScale;
      int decimation;
      float user_torque_limit;
      float user_power_limit;
      float cycle_time;
    };

    struct InitState
    {
      // default joint angles
      scalar_t arm_l1_joint;
      scalar_t arm_l2_joint;
      scalar_t arm_l3_joint;
      scalar_t arm_l4_joint;
      scalar_t leg_l1_joint;
      scalar_t leg_l2_joint;
      scalar_t leg_l3_joint;
      scalar_t leg_l4_joint;
      scalar_t leg_l5_joint;
      scalar_t arm_r1_joint;
      scalar_t arm_r2_joint;
      scalar_t arm_r3_joint;
      scalar_t arm_r4_joint;
      scalar_t leg_r1_joint;
      scalar_t leg_r2_joint;
      scalar_t leg_r3_joint;
      scalar_t leg_r4_joint;
      scalar_t leg_r5_joint;
    };

    struct ObsScales
    {
      scalar_t linVel;
      scalar_t angVel;
      scalar_t dofPos;
      scalar_t dofVel;
      scalar_t quat;
      scalar_t heightMeasurements;
    };

    bool encoder_nomalize;

    scalar_t clipActions;
    scalar_t clipObs;

    InitState initState;
    ObsScales obsScales;
    ControlCfg controlCfg;

  
    int loophz;
    double cycletimeerrorThreshold;
    int ThreadPriority;

  };

class UserController 
{
   
public:
   
   UserController() : memoryInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)){}

    ~UserController()  = default;
    static UserController* Instance()
    {
       static UserController  usercontrol;
       return &usercontrol;
    }
    static UserController* instance; // 静态指针，用于调用非静态方法

    static void StaticCallback() {
        if (instance) {
            instance->process(); // 通过实例指针调用成员函数
        }
    }
    bool init(ControlMode mode );
    bool loadModel(std::string modelpath);// override ;
    bool loadWalkModel(const std::string& modelpath);
    bool loadrknnmodel(std::string modle_path);
    bool handleWalkMode() ;
    void setparameter(Command &cmd,bool* isfirst);
    void handleStandMode();
    void handleDefautMode();
    void handleLieMode();
    
    void process();
    #ifdef RK3588
    unsigned char *load_model(const char* filename,int* model_size);
    void dump_tensor_attr(rknn_tensor_attr* attr);
    unsigned int as_uint(const float x);
    float as_float(const unsigned int x);
    float  half_to_float(const unsigned short x);
    unsigned short float_to_half(const float x);
    #endif
  protected:
   
   
    void computeActions() ;
    void computeWalkActions();
    void computeWalkObservation();
    void computeActionsrknn();
    void computeObservation() ;
   

    void onnxdatainit();
    bool getmodelparam();
    bool updateStateEstimation(); 
    

  private:
    void readRemoteCommand(double* command) const;
    void updateRemoteMode();
    std::vector<std::string> jointNames{"arm_l1_joint", "arm_l2_joint", "arm_l3_joint", "arm_l4_joint",
                                        "leg_l1_joint", "leg_l2_joint", "leg_l3_joint", "leg_l4_joint", "leg_l5_joint",
                                        "arm_r1_joint", "arm_r2_joint", "arm_r3_joint", "arm_r4_joint",
                                        "leg_r1_joint", "leg_r2_joint", "leg_r3_joint", "leg_r4_joint", "leg_r5_joint"
                                                                               
                                      };
    std::vector<int> policyToMotor_;
    std::vector<int> motorToPolicy_;

 
    Ort::MemoryInfo memoryInfo;
    std::shared_ptr<Ort::Env> onnxEnvPrt_;
    Controllerbase controllerbase;
  
    std::string policyFilePath_;
    std::string modelname;
    int64_t count;
    RobotCfg robotconfig;
    std::unique_ptr<Ort::Session> policySessionPtr;
    std::unique_ptr<Ort::Session> walkSessionPtr_;
    std::string walkInputName_;
    std::string walkOutputName_;
    std::vector<tensor_element_t> walkObservations_;
    bool walkHistoryFirst_ = true;
    struct WalkConfig
    {
      std::array<scalar_t, 18> defaults{};
      std::array<scalar_t, 18> stiffness{};
      std::array<scalar_t, 18> damping{};
      scalar_t actionScale = 0;
      scalar_t linVel = 0;
      scalar_t angVel = 0;
      scalar_t dofPos = 0;
      scalar_t dofVel = 0;
      scalar_t quat = 0;
      scalar_t scalex = 0;
      scalar_t scaley = 0;
      scalar_t clipObs = 0;
      scalar_t clipActions = 0;
      int decimation = 0;
    } walkConfig_;
    std::unique_ptr<Ort::Session> estSessionPtr;
    std::vector<const char *> policyInputNames_;
    std::vector<const char *> policyOutputNames_;
    std::vector<const char *> estInputNames_;
    std::vector<const char *> estOutputNames_;
    std::vector<Ort::AllocatedStringPtr> policyInputNodeNameAllocatedStrings;
    std::vector<Ort::AllocatedStringPtr> policyOutputNodeNameAllocatedStrings;
    std::vector<Ort::AllocatedStringPtr> estInputNodeNameAllocatedStrings;
    std::vector<Ort::AllocatedStringPtr> estOutputNodeNameAllocatedStrings;
    std::vector<std::vector<int64_t>> policyInputShapes_;
    std::vector<std::vector<int64_t>> policyOutputShapes_;
    std::vector<std::vector<int64_t>> estInputShapes_;
    std::vector<std::vector<int64_t>> estOutputShapes_;

    vector3_t baseLinVel_;
    vector3_t basePosition_;
    vector_t lastActions_;
    vector_t defaultJointAngles_;
    int actuatedDofNum_;
    bool* isfirstRecObs_;
    int actionsSize_;
    int motionSize;
    int observationSize_;
    int stackSize_;
    float scalez;
    float scalex;
    float scaley;
    std::vector<tensor_element_t> actions_;
    std::vector<tensor_element_t> policyObservations_;
    std::vector<tensor_element_t> estObservations_;
    
    Eigen::Matrix<tensor_element_t, Eigen::Dynamic, 1> proprioHistoryBuffer_;
    bool isfirstCompAct_{true};
    vector_t command_;
    Proprioception propri_;
    double phase_;
    int64_t loopcount_;
    NingImuData imu_data;
    joydata remote_data;
    double standPercent;
    scalar_t standDuration;
    vector_t lieJointAngles;
    JointState lieJointState{ 0.0,0.0,0.0,0.0,
                              0.0,0.0,0.0,0.0,0.0,
                              0.0,0.0,0.0,0.0,
                              0.0,0.0,0.0,0.0,0.0};
    JointState standJointStatie_{0.0,0.2,0.0,0.0,
                                0.0,0.0,-0.25,0.5,-0.25,
                                0.0,-0.2,0.0,0.0,
                                0.0,0.0,-0.25,0.5,-0.25 };                            
    WorkMode  mode_;        
    PolicyKind activePolicy_ = PolicyKind::DEFAULT;
    bool walkPolicyReady_ = false;
    bool isChangeMode_=false;                                 
    bool startcontrol = false; 
    int initfinish = 0;
 



    vector_t lieJointAngles_;
    vector_t standJointAngles_;
    #ifdef RK3588
    rknn_context ctx =0;
    std::vector<rknn_tensor_attr> input_attrs;
    std::vector<rknn_tensor_attr> output_attrs;
    rknn_input_output_num io_num;
    #endif






   
};
} 
#endif
