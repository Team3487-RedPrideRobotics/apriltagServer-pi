// Copyright (c) FIRST and other WPILib contributors.
// Open Source Software; you can modify and/or share it under the terms of
// the WPILib BSD license file in the root directory of this project.

#include <cstdio>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <fmt/format.h>
#include <networktables/NetworkTableInstance.h>
#include <vision/VisionPipeline.h>
#include <vision/VisionRunner.h>
#include <wpi/StringExtras.h>
#include <wpi/json.h>
#include <wpi/raw_istream.h>
#include <wpi/raw_ostream.h>

#include "cameraserver/CameraServer.h"
#include "apriltag/common/image_u8.h"
#include "apriltag/common/zarray.h"
#include "apriltag/apriltag.h"
#include "apriltag/tag36h11.h"
#include "apriltag/apriltag_pose.h"

#include "opencv4/opencv2/core/mat.hpp"
#include "opencv4/opencv2/calib3d/calib3d.hpp"



/*
   JSON format:
   {
       "team": <team number>,
       "ntmode": <"client" or "server", "client" if unspecified>
       "cameras": [
           {
               "name": <camera name>
               "path": <path, e.g. "/dev/video0">
               "pixel format": <"MJPEG", "YUYV", etc>   // optional
               "width": <video mode width>              // optional
               "height": <video mode height>            // optional
               "fps": <video mode fps>                  // optional
               "brightness": <percentage brightness>    // optional
               "white balance": <"auto", "hold", value> // optional
               "exposure": <"auto", "hold", value>      // optional
               "properties": [                          // optional
                   {
                       "name": <property name>
                       "value": <property value>
                   }
               ],
               "stream": {                              // optional
                   "properties": [
                       {
                           "name": <stream property name>
                           "value": <stream property value>
                       }
                   ]
               }
           }
       ]
       "switched cameras": [
           {
               "name": <virtual camera name>
               "key": <network table key used for selection>
               // if NT value is a string, it's treated as a name
               // if NT value is a double, it's treated as an integer index
           }
       ]
   }
 */

static const char* configFile = "/boot/frc.json";

namespace {

unsigned int team;
bool server = false;

struct CameraConfig {
  std::string name;
  std::string path;
  wpi::json config;
  wpi::json streamConfig;
};

struct SwitchedCameraConfig {
  std::string name;
  std::string key;
};

std::vector<CameraConfig> cameraConfigs;
std::vector<SwitchedCameraConfig> switchedCameraConfigs;
std::vector<cs::VideoSource> cameras;

void ParseError(std::string_view msg) {
  fmt::print(stderr, "config error in '{}': {}\n", configFile, msg);
}

bool ReadCameraConfig(const wpi::json& config) {
  CameraConfig c;

  // name
  try {
    c.name = config.at("name").get<std::string>();
  } catch (const wpi::json::exception& e) {
    ParseError(fmt::format("could not read camera name: {}", e.what()));
    return false;
  }

  // path
  try {
    c.path = config.at("path").get<std::string>();
  } catch (const wpi::json::exception& e) {
    ParseError(fmt::format("camera '{}': could not read path: {}", c.name, e.what()));
    return false;
  }

  // stream properties
  if (config.count("stream") != 0) c.streamConfig = config.at("stream");

  c.config = config;

  cameraConfigs.emplace_back(std::move(c));
  return true;
}

bool ReadSwitchedCameraConfig(const wpi::json& config) {
  SwitchedCameraConfig c;

  // name
  try {
    c.name = config.at("name").get<std::string>();
  } catch (const wpi::json::exception& e) {
    ParseError(fmt::format("could not read switched camera name: {}",
                           e.what()));
    return false;
  }

  // key
  try {
    c.key = config.at("key").get<std::string>();
  } catch (const wpi::json::exception& e) {
    ParseError(fmt::format("switched camera '{}': could not read key: {}",
                           c.name, e.what()));
    return false;
  }

  switchedCameraConfigs.emplace_back(std::move(c));
  return true;
}

bool ReadConfig() {
  // open config file
  std::error_code ec;
  wpi::raw_fd_istream is(configFile, ec);
  if (ec) {
    wpi::errs() << "could not open '" << configFile << "': " << ec.message()
                << '\n';
    return false;
  }

  // parse file
  wpi::json j;
  try {
    j = wpi::json::parse(is);
  } catch (const wpi::json::parse_error& e) {
    ParseError(fmt::format("byte {}: {}", e.byte, e.what()));
    return false;
  }

  // top level must be an object
  if (!j.is_object()) {
    ParseError("must be JSON object");
    return false;
  }

  // team number
  try {
    team = j.at("team").get<unsigned int>();
  } catch (const wpi::json::exception& e) {
    ParseError(fmt::format("could not read team number: {}", e.what()));
    return false;
  }

  // ntmode (optional)
  if (j.count("ntmode") != 0) {
    try {
      auto str = j.at("ntmode").get<std::string>();
      if (wpi::equals_lower(str, "client")) {
        server = false;
      } else if (wpi::equals_lower(str, "server")) {
        server = true;
      } else {
        ParseError(fmt::format("could not understand ntmode value '{}'", str));
      }
    } catch (const wpi::json::exception& e) {
      ParseError(fmt::format("could not read ntmode: {}", e.what()));
    }
  }

  // cameras
  try {
    for (auto&& camera : j.at("cameras")) {
      if (!ReadCameraConfig(camera)) return false;
    }
  } catch (const wpi::json::exception& e) {
    ParseError(fmt::format("could not read cameras: {}", e.what()));
    return false;
  }

  // switched cameras (optional)
  if (j.count("switched cameras") != 0) {
    try {
      for (auto&& camera : j.at("switched cameras")) {
        if (!ReadSwitchedCameraConfig(camera)) return false;
      }
    } catch (const wpi::json::exception& e) {
      ParseError(fmt::format("could not read switched cameras: {}", e.what()));
      return false;
    }
  }

  return true;
}

cs::UsbCamera StartCamera(const CameraConfig& config) {
  fmt::print("Starting camera '{}' on {}\n", config.name, config.path);
  cs::UsbCamera camera{config.name, config.path};
  auto server = frc::CameraServer::StartAutomaticCapture(camera);

  camera.SetConfigJson(config.config);
  camera.SetConnectionStrategy(cs::VideoSource::kConnectionKeepOpen);

  if (config.streamConfig.is_object())
    server.SetConfigJson(config.streamConfig);

  return camera;
}

cs::MjpegServer StartSwitchedCamera(const SwitchedCameraConfig& config) {
  fmt::print("Starting switched camera '{}' on {}\n", config.name, config.key);
  auto server = frc::CameraServer::AddSwitchedCamera(config.name);

  nt::NetworkTableInstance::GetDefault()
      .GetEntry(config.key)
      .AddListener(
          [server](const auto& event) mutable {
            if (event.value->IsDouble()) {
              int i = event.value->GetDouble();
              if (i >= 0 && i < cameras.size()) server.SetSource(cameras[i]);
            } else if (event.value->IsString()) {
              auto str = event.value->GetString();
              for (int i = 0; i < cameraConfigs.size(); ++i) {
                if (str == cameraConfigs[i].name) {
                  server.SetSource(cameras[i]);
                  break;
                }
              }
            }
          },
          NT_NOTIFY_IMMEDIATE | NT_NOTIFY_NEW | NT_NOTIFY_UPDATE);

  return server;
}

// example pipeline
class MyPipeline : public frc::VisionPipeline {
 public:
      apriltag_detector_t *td = apriltag_detector_create();
      apriltag_family_t *tf = tag36h11_create();
      nt::NetworkTableEntry detectionDistance;
      double cameraArray[3][3] = {{1430.24403,0.0,645.226606},{0.0,1415.29681,487.429715},{0.0,0.0,1.0}};
      double newCameraArray[3][3] = {1413.22522,0.0,645.461355,0.0,1407.51892,488.005117,0.0,0.0,1.0};
      double dist[5] = {-0.0322087,-0.01251858,0.00189451,0.00087394,0.37527268};
      cv::Mat cameraMatrix = cv::Mat(3, 3, CV_32FC1, &cameraArray);
      cv::Mat newCameraMatrix = cv::Mat(3, 3, CV_32FC1, &newCameraArray);
      cv::Mat distMatrix = cv::Mat(1,5,CV_32FC1, &dist);
      int x = 4;
      int y = 2;
      int w = 1272;
      int h = 954;
      apriltag_detection_info_t info;
      double tagsize = 0.156;
      double target_distance_from_tag = 23/100; // radius of basketball hoop (cm * m/cm)
  
  MyPipeline(void){
    apriltag_detector_add_family(td, tf);
    auto inst = nt::NetworkTableInstance::GetDefault();
    auto table = inst.GetTable("datatable");
    detectionDistance = table->GetEntry("Detection Distance");
    info.tagsize = tagsize;
    info.fx = cameraArray[0][0];
    info.fy = cameraArray[1][1];
    info.cx = cameraArray[0][2];
    info.cy = cameraArray[1][2];
  }

  void Process(cv::Mat& mat) override {
    //TODO: refractor (probably don't need to create and destroy the detector every frame)
    image_u8_t img_header = { .width = mat.cols,
    .height = mat.rows,
    .stride = mat.cols,
    .buf = mat.data
    };
    zarray_t *detections = apriltag_detector_detect(td, &img_header);
    for (int i = 0; i < zarray_size(detections); i++) {
    apriltag_detection_t *det;
    zarray_get(detections, i, &det);

    info.det = det;

    apriltag_pose_t pose;
    estimate_tag_pose(&info, &pose);


    matd_t* detection_coordinates = matd_multiply(matd_multiply(matd_transpose(pose.R), matd_create_scalar(-1)),pose.t);

    
     detectionDistance.SetDouble(sqrt((pow(matd_get(detection_coordinates,0,2)+target_distance_from_tag,2))+pow(matd_get(detection_coordinates,0,0),2))); 
    // Do stuff with detections here.
    // I'm stuff    
}
  }
};
}  // namespace

int main(int argc, char* argv[]) {
  if (argc >= 2) configFile = argv[1];

  // read configuration
  if (!ReadConfig()) return EXIT_FAILURE;

  // start NetworkTables
  auto ntinst = nt::NetworkTableInstance::GetDefault();
  if (server) {
    fmt::print("Setting up NetworkTables server\n");
    ntinst.StartServer();
  } else {
    fmt::print("Setting up NetworkTables client for team {}\n", team);
    ntinst.StartClientTeam(team);
    ntinst.StartDSClient();
  }

  // start cameras
  for (const auto& config : cameraConfigs)
    cameras.emplace_back(StartCamera(config));

  // start switched cameras
  for (const auto& config : switchedCameraConfigs) StartSwitchedCamera(config);

  // start image processing on camera 0 if present
  if (cameras.size() >= 1) {
    std::thread([&] {
      frc::VisionRunner<MyPipeline> runner(cameras[0], new MyPipeline(),
                                           [&](MyPipeline &pipeline) {
        // do something with pipeline results
      });
      /* something like this for GRIP:
      frc::VisionRunner<MyPipeline> runner(cameras[0], new grip::GripPipeline(),
                                           [&](grip::GripPipeline& pipeline) {
        ...
      });
       */
      runner.RunForever();
    }).detach();
  }

  // loop forever
  for (;;) std::this_thread::sleep_for(std::chrono::seconds(10));
}
