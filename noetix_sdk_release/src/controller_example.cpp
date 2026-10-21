#include "controllerbase.h"
#include "usercontroller.h"
#include "yaml-cpp/yaml.h"
#include <chrono>
#include <thread>
#include <unistd.h>
#include <stdio.h>
#define usermode
using namespace legged;
int main()
{
  char buf[256];
  bool ret = true;
  getcwd(buf,sizeof(buf)); 
  std::string path=std::string(buf);
  printf("cur path is %s\n",path.c_str());
  #ifdef usermode
  UserController usercontroller;
  ret = usercontroller.init(ControlMode::USERMODE);
  if(ret == false)
  {
    printf("usercontroller.init   failed\n");
    return 0;
  }
  const YAML::Node policyConfig = YAML::LoadFile(path+"/config/policy.yaml");
  #ifdef RK3588
    const std::string configuredPolicyPath = policyConfig["default"]["rknn"].as<std::string>();
  #else
    const std::string configuredPolicyPath = policyConfig["default"]["onnx"].as<std::string>();
  #endif
  if (configuredPolicyPath.empty()) return 1;
  const std::string defaultPolicyPath = configuredPolicyPath.front() == '/'
      ? configuredPolicyPath : path+"/"+configuredPolicyPath;
  #ifdef RK3588
    ret = usercontroller.loadrknnmodel(defaultPolicyPath);
  #else
    ret = usercontroller.loadModel(defaultPolicyPath);
  #endif
  if (!ret) return 1;
  if (!usercontroller.loadWalkModel(path+"/policy")) return 1;
  if (!usercontroller.loadRunModel(path+"/policy")) return 1;
  usercontroller.process();
  #else
  Controllerbase controller_base;
  ret = controller_base.init(ControlMode::DEFAULT);
  
  if(ret == false)
  {
    
    return 0;
  }
  #ifdef RK3588
  ret = controller_base.loadModel("walk",path+"/policy","rknn");
  ret = controller_base.loadModel("run",path+"/policy","rknn");
  ret = controller_base.loadModel("ygdance",path+"/policy","rknn");

  #else
  ret = controller_base.loadModel("walk",path+"/policy","onnx");
  ret = controller_base.loadModel("run",path+"/policy","onnx");
  ret = controller_base.loadModel("ygdance",path+"/policy","onnx");
  
  
  #endif
  printf("loadModel ret %d\n",ret);
  controller_base.start();
  #endif
  
  
  while(1)
  {
    usleep(10);
  }
  return 0;
}
