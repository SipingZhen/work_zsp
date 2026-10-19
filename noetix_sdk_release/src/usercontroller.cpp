#ifdef NOETIX_SIM_OBS
#include "sim_usercontroller_adapter.h"
#else
#include "usercontroller.h"
#include "controllerbase.h"
#include "RotationTools.h"
#include <algorithm>
#include <array>
#include <cmath>
#include "yaml-cpp/yaml.h"
#include <iostream>
#include <chrono>
#include <thread>
#include <unistd.h>
#include <stdio.h>
#include <stdexcept>

namespace legged
{
  //Controllerbase controllerbase;
  UserController* UserController::instance = nullptr;
  #ifdef RK3588
   unsigned char *UserController::load_model(const char* filename,int* model_size)
  {
     FILE* fp=fopen(filename,"rb");
    if(fp == nullptr){
      printf("fopen %s fail \n",filename);
      return NULL;
    }
    fseek(fp,0,SEEK_END);
    int model_len=ftell(fp);
    unsigned char* model = (unsigned char*)malloc(model_len);
    fseek(fp,0,SEEK_SET);
    if(model_len != fread(model,1,model_len,fp)){
      printf("fread %s fail \n",filename);
      free(model);
      return NULL;
    }
    *model_size = model_len;
    if(fp)
    {
      fclose(fp);
    }
    return model;
  }
  void UserController::dump_tensor_attr(rknn_tensor_attr* attr)
  {
     std::string shape_str = attr->n_dims < 1 ? "" : std::to_string(attr->dims[0]);
     for(int i =1;i<attr->n_dims;++i)
     {
       shape_str +=", "+std::to_string(attr->dims[i]);
     }
     printf(" index = %d ,name= %s,n_dims=%d,dims=[%s],n_elems=%d,size=%d,w_stride=%d,size_with_stride=%d,fmt=%s,"
           "type=%s, qnt_type=%s,zp=%d,scale=%f\n",
           attr->index,attr->name,attr->n_dims,shape_str.c_str(),attr->n_elems,attr->size,attr->w_stride,attr->size_with_stride,get_format_string(attr->fmt),get_type_string(attr->type),
           get_qnt_type_string(attr->qnt_type),attr->zp,attr->scale);
  }
  unsigned int UserController::as_uint(const float x)
  {
    return *(unsigned int*)&x;
  }
  float UserController::as_float(const unsigned int x)
  {
    return *(float*)&x;
  }
  float  UserController::half_to_float(const unsigned short x)
  {
    const unsigned int e=(x&0x7c00)>>10;
    const unsigned int m=(x&0x03ff)<<13;
    const unsigned int v=as_uint((float)m)>>23;
    return as_float((x&0x8000)<<16 | (e!=0)*((e+112)<<23|m) | ((e==0)&(m!=0))*((v-37)<<23 |((m<<(150-v))&0x007fe000)));
  }
  unsigned short UserController::float_to_half(const float x)
  {
    const unsigned int b = as_uint(x)+0x00001000;
    const unsigned int e = (b&0x7f800000)>>23;
    const unsigned int m = b&0x007fffff;
    return (b&0x80000000)>>16 | (e>112)*((((e-112)<<10)&0x7c00)|m>>13) | ((e<113)&(e>101))*((((0x007ff000+m)>>(125-e))+1)>>1)|(e>143)*0x7fff;
  }
  #endif
  bool UserController::init(ControlMode mode)
  {
    char buf[256];
    getcwd(buf,sizeof(buf)); 
    std::string path=std::string(buf);
    printf("cur path is %s\n",path.c_str());
    YAML::Node  acconfig = YAML::LoadFile(path+"/config/ning_user.yaml");
    mode_ = WorkMode::DEFAULT;
    standDuration = 1000;
    standPercent = 0;
    lieJointAngles_.resize(18);
    standJointAngles_.resize(18);
    auto &LieState = lieJointState;
    auto &StandState = standJointStatie_;
    lieJointAngles_ <<  LieState.arm_l1_joint, LieState.arm_l2_joint, LieState.arm_l3_joint,LieState.arm_l4_joint, 
                  LieState.leg_l1_joint, LieState.leg_l2_joint, LieState.leg_l3_joint,LieState.leg_l4_joint, LieState.leg_l5_joint,
                  LieState.arm_r1_joint, LieState.arm_r2_joint, LieState.arm_r3_joint,LieState.arm_r4_joint, 
                  LieState.leg_r1_joint, LieState.leg_r2_joint, LieState.leg_r3_joint,LieState.leg_r4_joint, LieState.leg_r5_joint;
    standJointAngles_ <<  StandState.arm_l1_joint, StandState.arm_l2_joint, StandState.arm_l3_joint,StandState.arm_l4_joint, 
                            StandState.leg_l1_joint, StandState.leg_l2_joint, StandState.leg_l3_joint,StandState.leg_l4_joint, StandState.leg_l5_joint,
                            StandState.arm_r1_joint, StandState.arm_r2_joint, StandState.arm_r3_joint,StandState.arm_r4_joint, 
                            StandState.leg_r1_joint, StandState.leg_r2_joint, StandState.leg_r3_joint,StandState.leg_r4_joint, StandState.leg_r5_joint;
    instance = this;
    controllerbase.setcallback(StaticCallback);
    initfinish = 1;  
    return controllerbase.init(mode);
  }
  void UserController::setparameter(Command &cmd,bool* isfirst)
  {
      isfirstRecObs_ = isfirst;
      isfirstCompAct_ = *isfirstRecObs_;
      command_[0] = cmd.x;
      command_[1] = cmd.y;
      command_[2] = cmd.yaw;
  }
  
  bool UserController::updateStateEstimation()
  {
    vector_t jointPosnoarm(18),jointVelnoarm(18),jointTornoarm(18);

    quaternion_t quat;
    vector3_t angularVel, linearAccel;
    static int num=0;
    std::array<MotorState,18> joint_state = controllerbase.get_joint_state();
    NingImuData imudata;
    std::chrono::microseconds now = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch());
   
    // int j=0;
    for (size_t i = 0; i < actuatedDofNum_; ++i)
    {

      jointPosnoarm(i) = joint_state[i].pos ;
      jointVelnoarm(i) = joint_state[i].vel ;
      jointTornoarm(i) = joint_state[i].tau ;
      if(abs(jointPosnoarm(i)) == 12.5)
      {
        //LogError(" {} disconnect",jointNames[i]);
        return false;
      }
    }
    imudata = controllerbase.get_imu_data();
    for (size_t i = 0; i < 4; ++i)
    {
      quat.coeffs()(i) = imudata.ori[i];
    }
    for (size_t i = 0; i < 3; ++i)
    {
      angularVel(i) = imudata.angular_vel[i];
      linearAccel(i) = imudata.linear_acc[i];
      //printf("imudata.angular_vel %f  imudata.linear_acc  %f  i  %d\n",imudata.angular_vel[i],imudata.linear_acc[i],i);
    }


    propri_.jointPos = jointPosnoarm;
    propri_.jointVel = jointVelnoarm;
    propri_.baseAngVel = angularVel;

    vector3_t gravityVector(0, 0, -1);
    vector3_t zyx = quatToZyx(quat);
    matrix_t inverseRot = getRotationMatrixFromZyxEulerAngles(zyx).inverse();
    propri_.projectedGravity = inverseRot * gravityVector;
    propri_.baseEulerXyz = quatToXyz(quat);

    double seconds = now.count();
    phase_ = seconds/1000000.0;

    
    return true;
  }
  void UserController::handleDefautMode()
  {
    MotorCmd  motorcmd;

    for (int j = 0; j < 18; j++)
    {
      motorcmd.kd=0.1;
      motorcmd.pos=0;
      motorcmd.kp=0;
      motorcmd.motor_id = j;
      motorcmd.vel =0;
      motorcmd.tau = 0;

      controllerbase.set_joint(motorcmd);
    }

  }
  void UserController::handleStandMode()
  {
    MotorCmd motorcmd;

    if (standPercent <= 1)
    {
      for (int j = 0; j < 18; j++)
      {
          scalar_t pos_des = lieJointAngles_[j] * (1 - standPercent) + standJointAngles_[j] * standPercent;
          if(j <4 ){
        
          motorcmd.pos=pos_des;
          motorcmd.kp=20;
          motorcmd.kd=0.1;
          motorcmd.motor_id = j;
          motorcmd.vel =0;
          motorcmd.tau = 0;
          
          controllerbase.set_joint(motorcmd);
        }
        else if((j > 4) && (j<9)){
          
          motorcmd.pos=pos_des;
          motorcmd.kp=50;
          motorcmd.kd=1;
          motorcmd.motor_id =j;
          motorcmd.vel =0;
          motorcmd.tau = 0;
          
          controllerbase.set_joint(motorcmd);
        }
        else if((j > 9) && (j<13)){

          motorcmd.pos=pos_des;
          motorcmd.kp=20;
          motorcmd.kd=0.1;
          motorcmd.motor_id = j;
          motorcmd.vel =0;
          motorcmd.tau = 0;
          
          controllerbase.set_joint(motorcmd);
          
        }
        else{
          
          motorcmd.pos=pos_des;
          motorcmd.kp=50;
          motorcmd.kd=1;
          motorcmd.motor_id = j;
          motorcmd.vel =0;
          motorcmd.tau = 0;
          
          controllerbase.set_joint(motorcmd);
        }
      }
      standPercent += 1 / standDuration;
      standPercent = std::min(standPercent, scalar_t(1));
    }
    
  }
  void UserController::handleLieMode()
  {
      MotorCmd motorcmd;
      /*accontrol update  pos stiffness damping to control joint to action
      *
      *
      */
      std::array<MotorState,18> joint_state = controllerbase.get_joint_state();
      
      for(int i=0;i<18;i++)
      {
      
        if(abs(joint_state[i].pos) == 12.5)
        {
          printf("joint %d disconnect\n",i);
          return;
        }
      }
      if (standPercent <= 1)
      {

        for (int j = 0; j < 18; j++)
        {
          scalar_t pos_des = joint_state[j].pos * (1 - standPercent) + lieJointAngles_[j] * standPercent;
          if(j <4 ){
            motorcmd.pos=pos_des;
            motorcmd.kp=20;
            motorcmd.kd=0.1;
            motorcmd.motor_id = j;
            motorcmd.vel =0;
            motorcmd.tau = 0;
            controllerbase.set_joint(motorcmd);
            
          }
          else if((j > 4) && (j<9)){
            motorcmd.pos=pos_des;
            motorcmd.kp=50;
            motorcmd.kd=1;
            motorcmd.motor_id = j;
            motorcmd.vel =0;
            motorcmd.tau = 0;
            controllerbase.set_joint(motorcmd);
          }
          else if((j > 9) && (j<13)){

            motorcmd.pos=pos_des;
            motorcmd.kp=20;
            motorcmd.kd=0.1;
            motorcmd.motor_id = j;
            motorcmd.vel =0;
            motorcmd.tau = 0;
            controllerbase.set_joint(motorcmd);
          }
          else {

            motorcmd.pos=pos_des;
            motorcmd.kp=50;
            motorcmd.kd=1;
            motorcmd.motor_id = j;
            motorcmd.vel =0;
            motorcmd.tau = 0;
            controllerbase.set_joint(motorcmd);
          }
        }
        standPercent += 1 / standDuration;
        standPercent = std::min(standPercent, double(1));
      }
      
    
  }
 
  void UserController::process()
  {
    
    if(initfinish == 0)
      return;    
    Command  cmd;
    
    remote_data =controllerbase.get_jsdata();
#if 1  // Set to 0 to disable button-number logging.
    static bool previousButtons[14] = {};
    for (int i = 0; i < 14; ++i)
    {
      const bool pressed = remote_data.button[i] != 0;
      if (pressed && !previousButtons[i])
        printf("Remote button pressed: %d\n", i);
      previousButtons[i] = pressed;
    }
#endif
    double remoteCommand[3];
    readRemoteCommand(remoteCommand);
    cmd.x = remoteCommand[0];
    cmd.y = remoteCommand[1];
    cmd.yaw = remoteCommand[2];
    
    updateRemoteMode();

    switch (mode_)
    {
      case WorkMode::STAND:
        handleStandMode();
        break;
      case WorkMode::LIE:
        handleLieMode();
        break;  
      case WorkMode::USERWALK:
        setparameter(cmd,&isChangeMode_);
        handleWalkMode();
        break;
      case WorkMode::DEFAULT:
        handleDefautMode();
        break;  
      default:
        printf("Unexpected mode encountered: %d\n",static_cast<int>(mode_));
        break;
    }

  }
  bool UserController::handleWalkMode()
  {
    if(updateStateEstimation() == false)
       return false;
    // compute observation & actions
    const int decimation = activePolicy_ == PolicyKind::WALK
        ? walkConfig_.decimation : robotconfig.controlCfg.decimation;
    if (count % decimation == 0)
    {
      count = 0;    
      if (activePolicy_ == PolicyKind::WALK)
      {
        computeWalkObservation();
        computeWalkActions();
      }
      else
      {
        computeObservation();
        #ifdef RK3588
        computeActionsrknn();
        #else
        computeActions();
        #endif
      }
      // limit action range
      const scalar_t clip = activePolicy_ == PolicyKind::WALK
          ? walkConfig_.clipActions : robotconfig.clipActions;
      scalar_t actionMin = -clip;
      scalar_t actionMax = clip;
      std::transform(actions_.begin(), actions_.end(), actions_.begin(),
                    [actionMin, actionMax](scalar_t x)
                    { return std::max(actionMin, std::min(actionMax, x)); });
    }
    // set action
    MotorCmd motorcmd;
    for (int i = 0; i < actionsSize_; i++)
    {
      std::string partName = jointNames[i];
      const bool walk = activePolicy_ == PolicyKind::WALK;
      scalar_t pos_des = walk
          ? actions_[motorToPolicy_[i]] * walkConfig_.actionScale + walkConfig_.defaults[i]
          : actions_[motorToPolicy_[i]] * robotconfig.controlCfg.actionScale.at(partName) + defaultJointAngles_(i);
      double stiffness = walk ? walkConfig_.stiffness[i] : robotconfig.controlCfg.stiffness[partName];
      double damping = walk ? walkConfig_.damping[i] : robotconfig.controlCfg.damping[partName];
      // std::cout << "joint id: " << i << " joint_name:" << partName << " kp:" << stiffness << " kd:" << damping << std::endl;
      motorcmd.pos= pos_des;
      motorcmd.kp= stiffness;
      motorcmd.kd= damping;
      motorcmd.motor_id = i;
      motorcmd.vel =0;
      motorcmd.tau = 0;
      controllerbase.set_joint(motorcmd);
      lastActions_(motorToPolicy_[i], 0) = actions_[motorToPolicy_[i]];
    }
    count++;
    return true;
  }
   
  void UserController::onnxdatainit()
  {
    Ort::AllocatorWithDefaultOptions allocator;
   // ROS_INFO_STREAM("count: " << poliycfg->policySessionPtr->GetOutputCount());
    for (int i = 0; i < policySessionPtr->GetInputCount(); i++)
    {
      auto policyInputnamePtr = policySessionPtr->GetInputNameAllocated(i, allocator);
      policyInputNodeNameAllocatedStrings.push_back(std::move(policyInputnamePtr));
      policyInputNames_.push_back(policyInputNodeNameAllocatedStrings.back().get());
      // inputNames_.push_back(sessionPtr_->GetInputNameAllocated(i, allocator).get());
      policyInputShapes_.push_back(policySessionPtr->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
      std::vector<int64_t> policyShape = policySessionPtr->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
      std::cerr << "Policy Shape: [";
      for (size_t j = 0; j < policyShape.size(); ++j)
      {
          std::cout << policyShape[j];
          if (j != policyShape.size() - 1)
          {
              std::cerr << ", ";
          }
      }
      std::cout << "]" << std::endl;
    }
    
    for (int i = 0; i < policySessionPtr->GetOutputCount(); i++)
    {
      auto policyOutputnamePtr = policySessionPtr->GetOutputNameAllocated(i, allocator);
      policyOutputNodeNameAllocatedStrings.push_back(std::move(policyOutputnamePtr));
      policyOutputNames_.push_back(policyOutputNodeNameAllocatedStrings.back().get());
      // outputNames_.push_back(sessionPtr_->GetOutputNameAllocated(i, allocator).get());
      std::cout << policySessionPtr->GetOutputNameAllocated(i, allocator).get() << std::endl;
      policyOutputShapes_.push_back(policySessionPtr->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
      std::vector<int64_t> policyShape = policySessionPtr->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
      std::cerr << "Policy Shape: [";
      for (size_t j = 0; j < policyShape.size(); ++j)
      {
          std::cout << policyShape[j];
          if (j != policyShape.size() - 1)
          {
              std::cerr << ", ";
          }
      }
      std::cout << "]" << std::endl;
    }
    

  }
  bool UserController::loadrknnmodel(std::string modle_path)
  { 
    #ifdef RK3588
    std::string policyFilePath;  
    int model_data_size =0;
    int ret = 0;
    policyFilePath = modle_path+"/policy.rknn";  //TODO: change your policy at here
    unsigned char* model_data=load_model(policyFilePath.c_str(),&model_data_size);
    ret = rknn_init(&ctx,model_data,model_data_size,0,NULL);
    if(ret <0)
    {
      printf("rknn init error %d \n",ret);
      return 0;
    }
    rknn_core_mask core_mask = RKNN_NPU_CORE_0_1_2;
    ret = rknn_set_core_mask(ctx,core_mask);
    if(ret <0)
    {
      printf("rknn rknn_set_core_maskerror %d \n",ret);
      return 0;
    } 

    rknn_sdk_version version;
    ret = rknn_query(ctx,RKNN_QUERY_SDK_VERSION,&version,sizeof(rknn_sdk_version));
    if(ret <0)
    {
      printf("rknn RKNN_QUERY_SDK_VERSION error %d \n",ret);
      return 0;
    }
    printf("sdk version : %s driver version : %s\n",version.api_version,version.drv_version);
    
    ret = rknn_query(ctx,RKNN_QUERY_IN_OUT_NUM,&io_num,sizeof(io_num));
    if(ret <0)
    {
      printf("rknn RKNN_QUERY_IN_OUT_NUM error %d \n",ret);
      return 0;
    }
    printf("model input num: %d,output num: %d \n",io_num.n_input,io_num.n_output);
    if (io_num.n_input != 1 || io_num.n_output != 1)
    {
      printf("N2 policy requires one input and one output\n");
      return false;
    }
    input_attrs.resize(io_num.n_input);
    for(int i=0;i<io_num.n_input;i++){
      input_attrs[i].index =i;
      ret = rknn_query(ctx,RKNN_QUERY_INPUT_ATTR,&(input_attrs[i]),sizeof(rknn_tensor_attr));
      if(ret <0)
      {
        printf("rknn RKNN_QUERY_INPUT_ATTR error %d \n",ret);
        return 0;
      }
      printf(" input info: ");
      dump_tensor_attr(&(input_attrs[i]));
    }

    output_attrs.resize(io_num.n_output);
    for(int i=0;i<io_num.n_output;i++){
      output_attrs[i].index =i;
      ret = rknn_query(ctx,RKNN_QUERY_OUTPUT_ATTR,&(output_attrs[i]),sizeof(rknn_tensor_attr));
      if(ret <0)
      {
        printf("rknn RKNN_QUERY_OUTPUT_ATTR error %d \n",ret);
        return -1;
      }
      printf(" output info: ");
      dump_tensor_attr(&(output_attrs[i]));
    }
    
    modelname="user";
    command_.resize(3);
    isfirstCompAct_ = true;
    isfirstRecObs_= NULL;
    count = 0;
    
    if (!getmodelparam()) return false;
    if (input_attrs[0].n_elems != observationSize_ * stackSize_ || output_attrs[0].n_elems != actionsSize_)
    {
      printf("RKNN policy shape does not match N2 (expected %d inputs, %d outputs)\n",
             observationSize_ * stackSize_, actionsSize_);
      return false;
    }
    
    #endif
    return true;
    
  }
  bool UserController::loadModel(std::string modelpath)
  {
    std::string policyFilePath;
    std::string estFilePath;
      // create session
    Ort::SessionOptions sessionOptions;
    bool ret;
    onnxEnvPrt_.reset(new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "LeggedOnnxController"));
    sessionOptions.SetInterOpNumThreads(1);
    if(onnxEnvPrt_ == NULL)
    {
        printf("onnxEnvPrt_  is null\n");
        return false;
    }
        
    policyFilePath= modelpath+"/policy.onnx";
    printf("Load Onnx model from path : %s\n",policyFilePath.c_str());

    
    policySessionPtr = std::make_unique<Ort::Session>(*onnxEnvPrt_, policyFilePath.c_str(), sessionOptions);
    if(policySessionPtr == NULL)
    {
        printf("load run model failed\n");
        return false;
    }



    // get input and output info
    policyInputNames_.clear();
    policyOutputNames_.clear();
    policyInputShapes_.clear();
    policyOutputShapes_.clear();
    estSessionPtr = NULL;
    modelname="user";
    command_.resize(3);
    isfirstCompAct_ = true;
    isfirstRecObs_= NULL;
    count = 0;
    onnxdatainit();
    if (!getmodelparam()) return false;
    if (policyInputShapes_.size() != 1 || policyOutputShapes_.size() != 1 ||
        (policyInputShapes_[0] != std::vector<int64_t>{1, observationSize_ * stackSize_} &&
         policyInputShapes_[0] != std::vector<int64_t>{1, stackSize_, observationSize_}) ||
        policyOutputShapes_[0].size() != 2 ||
        (policyOutputShapes_[0][1] > 0 && policyOutputShapes_[0][1] != actionsSize_))
    {
      printf("ONNX policy shape does not match N2 (expected 310 inputs, 18 outputs)\n");
      return false;
    }

    printf("Load Onnx run model successfully !!!\n");
    return true;
  }

  bool UserController::loadWalkModel(const std::string& modelpath)
  {
    try
    {
      if (!onnxEnvPrt_)
        onnxEnvPrt_ = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "LeggedOnnxController");
      Ort::SessionOptions options;
      options.SetInterOpNumThreads(1);
      const std::string path = modelpath + "/policy_walk.onnx";
      auto session = std::make_unique<Ort::Session>(*onnxEnvPrt_, path.c_str(), options);
      if (session->GetInputCount() != 1 || session->GetOutputCount() != 1 ||
          session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape() != std::vector<int64_t>{1, 195} ||
          session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape() != std::vector<int64_t>{1, 10})
      {
        printf("Walk policy must have shape [1,195] -> [1,10]\n");
        return false;
      }
      Ort::AllocatorWithDefaultOptions allocator;
      const YAML::Node walk = YAML::LoadFile(modelpath + "/../config/ning_ac.yaml")["walk"];
      if (walk["size"]["actions_size"].as<int>() != 10 ||
          walk["size"]["observations_size"].as<int>() != 39 ||
          walk["size"]["stack_size"].as<int>() != 5)
        throw std::runtime_error("Walk config must specify 5 x 39 observations and 10 actions");
      for (int i = 0; i < 18; ++i)
      {
        const std::string& name = jointNames[i];
        walkConfig_.defaults[i] = walk["init_state"]["default_joint_angle"][name].as<scalar_t>();
        walkConfig_.stiffness[i] = walk["control"]["stiffness"][name].as<scalar_t>();
        walkConfig_.damping[i] = walk["control"]["damping"][name].as<scalar_t>();
      }
      walkConfig_.actionScale = walk["control"]["action_scale"].as<scalar_t>();
      walkConfig_.decimation = walk["control"]["decimation"].as<int>();
      walkConfig_.clipObs = walk["normalization"]["clip_scales"]["clip_observations"].as<scalar_t>();
      walkConfig_.clipActions = walk["normalization"]["clip_scales"]["clip_actions"].as<scalar_t>();
      auto scales = walk["normalization"]["obs_scales"];
      walkConfig_.linVel = scales["lin_vel"].as<scalar_t>();
      walkConfig_.angVel = scales["ang_vel"].as<scalar_t>();
      walkConfig_.dofPos = scales["dof_pos"].as<scalar_t>();
      walkConfig_.dofVel = scales["dof_vel"].as<scalar_t>();
      walkConfig_.quat = scales["quat"].as<scalar_t>();
      walkConfig_.scalex = walk["axis_mappings"]["scalex"].as<scalar_t>();
      walkConfig_.scaley = walk["axis_mappings"]["scaley"].as<scalar_t>();
      if (walkConfig_.decimation <= 0) throw std::runtime_error("Walk decimation must be positive");
      walkInputName_ = session->GetInputNameAllocated(0, allocator).get();
      walkOutputName_ = session->GetOutputNameAllocated(0, allocator).get();
      walkObservations_.resize(195);
      walkSessionPtr_ = std::move(session);
      walkHistoryFirst_ = true;
      walkPolicyReady_ = true;
      printf("Loaded walk ONNX policy: %s\n", path.c_str());
      return true;
    }
    catch (const Ort::Exception& error)
    {
      printf("Failed to load walk ONNX policy: %s\n", error.what());
      return false;
    }
    catch (const std::exception& error)
    {
      printf("Failed to load walk configuration: %s\n", error.what());
      return false;
    }
  }

  void UserController::computeWalkActions()
  {
    const std::array<int64_t, 2> shape{1, 195};
    auto input = Ort::Value::CreateTensor<tensor_element_t>(
        memoryInfo, walkObservations_.data(), walkObservations_.size(), shape.data(), shape.size());
    const char* inputName = walkInputName_.c_str();
    const char* outputName = walkOutputName_.c_str();
    Ort::RunOptions options;
    auto outputs = walkSessionPtr_->Run(options, &inputName, &input, 1, &outputName, 1);
    const auto* walkActions = outputs[0].GetTensorData<tensor_element_t>();
    std::fill(actions_.begin(), actions_.end(), 0.0f);
    for (int i = 0; i < 10; ++i)
    {
      const int motor = i < 5 ? i + 4 : i + 8;
      actions_[motorToPolicy_[motor]] = walkActions[i];
    }
  }
  

  bool UserController::getmodelparam()
  {
    char buf[256];
    getcwd(buf,sizeof(buf)); 
    std::string conpath=std::string(buf);
    std::string path=modelname;
    RobotCfg::InitState &initState = robotconfig.initState;
    RobotCfg::ControlCfg &controlCfg = robotconfig.controlCfg;
    RobotCfg::ObsScales &obsScales = robotconfig.obsScales;
    
    
    YAML::Node  acconfig = YAML::LoadFile(conpath+"/config/ning_user.yaml");

    int error = 0;
    
    initState.arm_l1_joint = acconfig[modelname]["init_state"]["default_joint_angle"]["arm_l1_joint"].as<double>();
    initState.arm_l2_joint = acconfig[modelname]["init_state"]["default_joint_angle"]["arm_l2_joint"].as<double>();
    initState.arm_l3_joint = acconfig[modelname]["init_state"]["default_joint_angle"]["arm_l3_joint"].as<double>();
    initState.arm_l4_joint = acconfig[modelname]["init_state"]["default_joint_angle"]["arm_l4_joint"].as<double>();
    initState.arm_r1_joint = acconfig[modelname]["init_state"]["default_joint_angle"]["arm_r1_joint"].as<double>();
    initState.arm_r2_joint = acconfig[modelname]["init_state"]["default_joint_angle"]["arm_r2_joint"].as<double>();
    initState.arm_r3_joint = acconfig[modelname]["init_state"]["default_joint_angle"]["arm_r3_joint"].as<double>();
    initState.arm_r4_joint = acconfig[modelname]["init_state"]["default_joint_angle"]["arm_r4_joint"].as<double>();
   
    initState.leg_l1_joint = acconfig[modelname]["init_state"]["default_joint_angle"]["leg_l1_joint"].as<double>();
    initState.leg_l2_joint = acconfig[modelname]["init_state"]["default_joint_angle"]["leg_l2_joint"].as<double>();
    initState.leg_l3_joint = acconfig[modelname]["init_state"]["default_joint_angle"]["leg_l3_joint"].as<double>();
    initState.leg_l4_joint = acconfig[modelname]["init_state"]["default_joint_angle"]["leg_l4_joint"].as<double>();
    initState.leg_l5_joint = acconfig[modelname]["init_state"]["default_joint_angle"]["leg_l5_joint"].as<double>();

    initState.leg_r1_joint = acconfig[modelname]["init_state"]["default_joint_angle"]["leg_r1_joint"].as<double>();
    initState.leg_r2_joint = acconfig[modelname]["init_state"]["default_joint_angle"]["leg_r2_joint"].as<double>();
    initState.leg_r3_joint = acconfig[modelname]["init_state"]["default_joint_angle"]["leg_r3_joint"].as<double>();
    initState.leg_r4_joint = acconfig[modelname]["init_state"]["default_joint_angle"]["leg_r4_joint"].as<double>();
    initState.leg_r5_joint = acconfig[modelname]["init_state"]["default_joint_angle"]["leg_r5_joint"].as<double>();

   
    for(const auto& pair :acconfig[modelname]["control"]["stiffness"])
    {
      controlCfg.stiffness[pair.first.as<std::string>()] = pair.second.as<float>();
      
    }   
    
    for(const auto& pair :acconfig[modelname]["control"]["damping"])
    {
      controlCfg.damping[pair.first.as<std::string>()] =  pair.second.as<float>();  
        
    }     

    controlCfg.actionScale.clear();
    for (const auto& pair : acconfig[modelname]["control"]["action_scale"])
    {
      controlCfg.actionScale[pair.first.as<std::string>()] = pair.second.as<float>();
    }
    controlCfg.decimation = acconfig[modelname]["control"]["decimation"].as<int>();
    controlCfg.cycle_time = acconfig[modelname]["control"]["cycle_time"].as<float>();
 
    robotconfig.clipObs = acconfig[modelname]["normalization"]["clip_scales"]["clip_observations"].as<double>();
    robotconfig.clipActions = acconfig[modelname]["normalization"]["clip_scales"]["clip_actions"].as<double>();

    obsScales.linVel = acconfig[modelname]["normalization"]["obs_scales"]["lin_vel"].as<double>();
    obsScales.angVel = acconfig[modelname]["normalization"]["obs_scales"]["ang_vel"].as<double>();
    obsScales.dofPos = acconfig[modelname]["normalization"]["obs_scales"]["dof_pos"].as<double>();
    obsScales.dofVel = acconfig[modelname]["normalization"]["obs_scales"]["dof_vel"].as<double>();
    obsScales.heightMeasurements = acconfig[modelname]["normalization"]["obs_scales"]["height_measurements"].as<double>();
    obsScales.quat = acconfig[modelname]["normalization"]["obs_scales"]["quat"].as<double>();

    actionsSize_ = acconfig[modelname]["size"]["actions_size"].as<int>();
    observationSize_ = acconfig[modelname]["size"]["observations_size"].as<int>();
    stackSize_ = acconfig[modelname]["size"]["stack_size"].as<int>();
    if (actionsSize_ != 18 || observationSize_ != 62 || stackSize_ != 5) return false;

    policyToMotor_.clear();
    motorToPolicy_.assign(jointNames.size(), -1);
    for (const auto& name : acconfig[modelname]["policy_joint_order"])
    {
      const std::string joint = name.as<std::string>();
      auto it = std::find(jointNames.begin(), jointNames.end(), joint);
      if (it == jointNames.end()) throw std::runtime_error("Unknown policy joint: " + joint);
      const int motor = std::distance(jointNames.begin(), it);
      if (motorToPolicy_[motor] != -1) throw std::runtime_error("Duplicate policy joint: " + joint);
      motorToPolicy_[motor] = policyToMotor_.size();
      policyToMotor_.push_back(motor);
    }
    if (policyToMotor_.size() != jointNames.size() || controlCfg.actionScale.size() != jointNames.size())
      throw std::runtime_error("Incomplete N2 policy joint order or action scales");
    for (const auto& joint : jointNames)
      if (!controlCfg.actionScale.count(joint)) throw std::runtime_error("Missing action scale for " + joint);

    scalez = acconfig[modelname]["axis_mappings"]["scalez"].as<float>();
    scaley = acconfig[modelname]["axis_mappings"]["scaley"].as<float>();
    scalex = acconfig[modelname]["axis_mappings"]["scalex"].as<float>();
 
    actions_.resize(actionsSize_);
 
    std::vector<scalar_t> defaultJointAngles{
        robotconfig.initState.arm_l1_joint, robotconfig.initState.arm_l2_joint, robotconfig.initState.arm_l3_joint, robotconfig.initState.arm_l4_joint, 
        robotconfig.initState.leg_l1_joint, robotconfig.initState.leg_l2_joint, robotconfig.initState.leg_l3_joint,
        robotconfig.initState.leg_l4_joint, robotconfig.initState.leg_l5_joint,
        robotconfig.initState.arm_r1_joint, robotconfig.initState.arm_r2_joint, robotconfig.initState.arm_r3_joint, robotconfig.initState.arm_r4_joint, 
        robotconfig.initState.leg_r1_joint,robotconfig.initState.leg_r2_joint, robotconfig.initState.leg_r3_joint,
        robotconfig.initState.leg_r4_joint,robotconfig.initState.leg_r5_joint};
 
    actuatedDofNum_ =  18;
     
     
    policyObservations_.resize(observationSize_ * stackSize_);
    
    std::fill(policyObservations_.begin(), policyObservations_.end(), 0.0f);
    lastActions_.resize(actionsSize_);
    lastActions_.setZero();
    const int inputSize = stackSize_ * observationSize_;
    proprioHistoryBuffer_.resize(inputSize);
    defaultJointAngles_.resize(actuatedDofNum_);
    for (int i = 0; i < actuatedDofNum_; i++)
    {
      defaultJointAngles_(i) = defaultJointAngles[i];
      //printf("defaultJointAngles[%d]  %f\n",i,modelcfg->defaultJointAngles_(i));
    }
    return true;
  }

  void UserController::computeActionsrknn()
  { 
    
    #ifdef RK3588
    rknn_input inputs[1];
    int ret;
    float* outdata=NULL;
    float* inputdata=NULL;
        
    inputdata =(float*)policyObservations_.data();
    //std::cout<<"policyObservations_.size() "<<policyObservations_.size()<< std::endl;
    memset(inputs,0,sizeof(inputs));
    inputs[0].index= input_attrs[0].index;
    inputs[0].type= RKNN_TENSOR_FLOAT32;
    inputs[0].size= policyObservations_.size() * sizeof(tensor_element_t);
    inputs[0].fmt= input_attrs[0].fmt;
    inputs[0].pass_through= 0;
    inputs[0].buf = inputdata;

    ret=rknn_inputs_set(ctx,1,inputs);
    if (ret != 0)
    {
      printf("rknn_inputs_set failed: %d\n", ret);
      return;
    }

    ret=rknn_run(ctx,NULL);

    if(ret != 0)
    {
      printf("rknn run failed: %d\n", ret);
      return;
    }
    rknn_output outputs[io_num.n_output];
    memset(outputs,0,sizeof(outputs));
    for(int i=0;i<io_num.n_output;i++)
    {
      outputs[i].index = i;
      outputs[i].is_prealloc =0;
      outputs[i].want_float =1;
    }
    ret=rknn_outputs_get(ctx,io_num.n_output,outputs,NULL);
    if (ret != 0)
    {
      printf("rknn_outputs_get failed: %d\n", ret);
      return;
    }
    //printf("output buf size is %d,n_elems %d\n",outputs[0].size,output_attrs[0].n_elems);
    outdata = (float*)outputs[0].buf;
     if (isfirstCompAct_){
      printf("rknn@@@@@@@@@@@@@@@@\n");
      for (int i = 0; i < policyObservations_.size(); ++i) {
        std::cout << policyObservations_[i] << " ";
        if ((i + 1) % observationSize_ == 0) {
            std::cout << std::endl;
        }
      }
      isfirstCompAct_ = false;
    }
   
    for(int i=0;i<actionsSize_;i++)
    {
       actions_[i] = outdata[i];
       //actions_[i] = half_to_float(outdata[i]);
        //std::cout<<"rknn actions "<<actions_[i]<< std::endl;
    }
    //printf("###############################################\n");*/
    ret = rknn_outputs_release(ctx,io_num.n_output,outputs);
    #endif
  }
  void UserController::computeActions()
  {
    std::cout << "******************onxx****************" << std::endl;
    std::vector<Ort::Value> policyInputValues;
    policyInputValues.push_back(Ort::Value::CreateTensor<tensor_element_t>(memoryInfo, policyObservations_.data(),policyObservations_.size(),
                                                                          policyInputShapes_[0].data(), policyInputShapes_[0].size()));
     // run inference
    Ort::RunOptions runOptions;
    std::vector<Ort::Value> outputValues = policySessionPtr->Run(runOptions, policyInputNames_.data(), policyInputValues.data(), 1, policyOutputNames_.data(), 1);
    if (isfirstCompAct_)
    {
      for (int i = 0; i < policyObservations_.size(); ++i) {
        std::cout << policyObservations_[i] << " ";
        if ((i + 1) % observationSize_ == 0) {
            std::cout << std::endl;
        }
      }
      isfirstCompAct_ = false;
    }

    for (int i = 0; i < actionsSize_; i++)
    {
      actions_[i] = *(outputValues[0].GetTensorMutableData<tensor_element_t>() + i);
    }
    
    
  }

#endif // NOETIX_SIM_OBS
#ifdef NOETIX_SIM_OBS
namespace legged {
#endif
  void UserController::computeWalkObservation()
  {
    constexpr int frameSize = 39;
    constexpr int legActions = 10;
    constexpr int historyFrames = 5;

    vector_t command(3);
    command(0) = command_[0] * walkConfig_.scalex * walkConfig_.linVel;
    command(1) = command_[1] * walkConfig_.scaley * walkConfig_.linVel;
    command(2) = command_[2] * walkConfig_.angVel;

    vector_t jointPosObs(legActions);
    vector_t jointVelObs(legActions);
    vector_t actions(legActions);
    for (int i = 0; i < legActions; ++i)
    {
      const int motor = i < 5 ? i + 4 : i + 8;
      const int policy = motorToPolicy_[motor];
      jointPosObs(i) = (propri_.jointPos(motor) - walkConfig_.defaults[motor]) * walkConfig_.dofPos;
      jointVelObs(i) = propri_.jointVel(motor) * walkConfig_.dofVel;
      actions(i) = lastActions_(policy);
    }

    vector_t proprioObs(frameSize);
    proprioObs << command,                                      // 3
                    propri_.baseAngVel * walkConfig_.angVel,     // 3
                    propri_.projectedGravity * walkConfig_.quat, // 3
                    jointPosObs,                                 // 10
                    jointVelObs,                                 // 10
                    actions;                                     // 10

    tensor_element_t frame[frameSize];
    for (int i = 0; i < frameSize; ++i)
    {
      const scalar_t value = static_cast<tensor_element_t>(proprioObs(i));
      frame[i] = std::max(-walkConfig_.clipObs, std::min(walkConfig_.clipObs, value));
    }
    if (walkHistoryFirst_)
    {
      for (int i = 0; i < historyFrames; ++i)
        std::copy_n(frame, frameSize, walkObservations_.begin() + i * frameSize);
      walkHistoryFirst_ = false;
    }
    else
    {
      for (int i = frameSize; i < frameSize * historyFrames; ++i)
        walkObservations_[i - frameSize] = walkObservations_[i];
      std::copy_n(frame, frameSize, walkObservations_.end() - frameSize);
    }
  }

  void UserController::readRemoteCommand(double* command) const
  {
    command[0] = remote_data.axes[1];
    command[1] = 0;
    command[2] = remote_data.axes[0];
  }

  void UserController::updateRemoteMode()
  {
    static int keyflag[14];
    if((remote_data.button[9] == 1)&&(keyflag[9] == 0))
    {

      if(!startcontrol)
      {
        startcontrol = true;
        standPercent = 0;
        mode_ = WorkMode::LIE;
        keyflag[9] =1;
        printf("TO LIE MODE\n");

      }else
      {
          startcontrol = false;
          mode_ = WorkMode::DEFAULT;
          keyflag[9] =1;
          printf("stop control\n");
      }

    }else if(remote_data.button[9] == 0)
      keyflag[9] =0;
    if((remote_data.button[10] == 1)&&(remote_data.button[2] == 1)&&(keyflag[10] == 0))
    {
      keyflag[10] =1;
      if (startcontrol == true)
      {
        if (mode_ == WorkMode::LIE)
        {
          standPercent = 0;
          mode_ = WorkMode::STAND;
          printf("LIE2STAND\n");
        }
        else if (mode_ == WorkMode::STAND)
        {
          standPercent = 0;
          mode_ = WorkMode::LIE;
          printf("STAND2LIE\n");
        }

      }
    }else if(remote_data.button[10] == 0)
      keyflag[10] =0;
    if((remote_data.button[5] == 1)&&(remote_data.button[2] == 1)&&(keyflag[5] == 0))
    {

      if(mode_ == WorkMode::STAND || mode_ == WorkMode::USERWALK)
      {
        standPercent = 0;
        isChangeMode_ = true;
        lastActions_.setZero();
#ifndef NOETIX_SIM_OBS
        std::fill(actions_.begin(), actions_.end(), 0.0f);
        count = 0;
#endif
        activePolicy_ = PolicyKind::DEFAULT;
        walkHistoryFirst_ = true;
        mode_ = WorkMode::USERWALK;
        keyflag[5] =1;
        printf("TO DEFAULT POLICY\n");
      }

    }else if(remote_data.button[5] == 0)
      keyflag[5] =0;
    if((remote_data.button[6] == 1)&&(remote_data.button[2] == 1)&&(keyflag[6] == 0))
    {
      keyflag[6] = 1;
      if(walkPolicyReady_ && (mode_ == WorkMode::STAND || mode_ == WorkMode::USERWALK))
      {
        standPercent = 0;
        isChangeMode_ = true;
        lastActions_.setZero();
#ifndef NOETIX_SIM_OBS
        std::fill(actions_.begin(), actions_.end(), 0.0f);
        count = 0;
#endif
        activePolicy_ = PolicyKind::WALK;
        walkHistoryFirst_ = true;
        mode_ = WorkMode::USERWALK;
        printf("TO WALK POLICY\n");
      }
    }else if(remote_data.button[6] == 0)
      keyflag[6] = 0;
    if((remote_data.button[11] == 1)&&(keyflag[11] == 0))
    {
      keyflag[11] =1;
      if(mode_ == WorkMode::USERWALK)
      {
        isChangeMode_ = true;
        mode_ = WorkMode::STAND;
        printf("WALK2STAND\n");

      }else if (mode_ == WorkMode::DEFAULT)
      {
        standPercent = 0;
        printf("deftolie\n");
        isChangeMode_ = true;
        mode_ = WorkMode::LIE;

      }

    }else if(remote_data.button[11] == 0)
      keyflag[11] =0;

  }

  void UserController::computeObservation()
  {
    
    std::atomic<scalar_t> comm_x;
    std::atomic<scalar_t> comm_y;
    
    
    RobotCfg::ObsScales &obsScales = robotconfig.obsScales;
    
    // command
    vector_t proprioObs(observationSize_);
    const bool firstObservation = *isfirstRecObs_;
    if (firstObservation) lastActions_.setZero();
    
    vector_t command(3);
    comm_x = command_[0] * scalex;
    comm_y = command_[1] * scaley;

    // Apply a dead zone to the target, then limit each policy-frame change.
    static scalar_t limitedX = 0.0;
    if (firstObservation) limitedX = 0.0;
    scalar_t targetX = comm_x * obsScales.linVel;
    if (std::abs(targetX) < 0.3) targetX = 0.0;
    const scalar_t deltaX = std::max(-0.05, std::min(0.05, targetX - limitedX));
    limitedX += deltaX;

    command[0] = limitedX;
    command[1] = comm_y * obsScales.linVel;
    command[2] = command_[2] * obsScales.angVel;
    if (std::abs(command[0]) > 0.3) command[2] *= 0.6;


    
    
    vector_t actions(lastActions_);

    vector_t jointPosObs(actionsSize_);
    vector_t jointVelObs(actionsSize_);
    for (int i = 0; i < actionsSize_; ++i)
    {
      const int motor = policyToMotor_[i];
      jointPosObs(i) = (propri_.jointPos(motor) - defaultJointAngles_(motor)) * obsScales.dofPos;
      jointVelObs(i) = propri_.jointVel(motor) * obsScales.dofVel;
    }

    proprioObs << command, // 3
   	propri_.baseAngVel * obsScales.angVel,  // 3
        propri_.baseEulerXyz(0),  // roll
        propri_.baseEulerXyz(1),  // pitch
        
        jointPosObs,  // 18
        jointVelObs,  // 18
        actions;  // 18

    if (firstObservation)
    {
      for (int i = 0; i < stackSize_; i++)
        proprioHistoryBuffer_.segment(i * observationSize_, observationSize_) = proprioObs.cast<tensor_element_t>();
      *isfirstRecObs_ = false;
    }
    else
    {
      proprioHistoryBuffer_.head(proprioHistoryBuffer_.size() - observationSize_) =
          proprioHistoryBuffer_.tail(proprioHistoryBuffer_.size() - observationSize_);
      proprioHistoryBuffer_.tail(observationSize_) = proprioObs.cast<tensor_element_t>();
    }
  

    //clang-format on

    for (size_t i = 0; i < (observationSize_ * stackSize_); i++){
      policyObservations_[i] = static_cast<tensor_element_t>(proprioHistoryBuffer_[i]);
      // if(i < observationSize_)
      // std::cout << i << "obs:::" << estObservations_[i] << std::endl;
    }
   
    
  }

} // namespace legged


