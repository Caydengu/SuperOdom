// Raw, non-taring OptiTrack recorder for independent G1 reference evidence.
//
// This source links against the official NatNet SDK 4.4 Linux library. It
// records Motive coordinates exactly as delivered. Coordinate conversion and
// rigid-body-to-pelvis calibration are intentionally separate offline steps.

#include <NatNetCAPI.h>
#include <NatNetClient.h>
#include <NatNetTypes.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

#include <csignal>
#include <ctime>

namespace {

constexpr char kSchema[] = "g1_optitrack_raw_v1";
constexpr char kPinnedSdk[] = "NatNet_SDK_4.4";
constexpr char kDefaultMulticast[] = NATNET_DEFAULT_MULTICAST_ADDRESS;
constexpr std::size_t kMaximumQueuedRecords = 4096;

std::atomic<bool> g_stop_requested{false};

void HandleSignal(int) {
  g_stop_requested.store(true);
}

int64_t ClockNanoseconds(clockid_t clock_id) {
  timespec stamp{};
  if (clock_gettime(clock_id, &stamp) != 0) {
    throw std::runtime_error("clock_gettime failed");
  }
  return static_cast<int64_t>(stamp.tv_sec) * 1'000'000'000LL +
         static_cast<int64_t>(stamp.tv_nsec);
}

std::string JsonEscape(const std::string& input) {
  std::ostringstream output;
  for (const unsigned char character : input) {
    switch (character) {
      case '"':
        output << "\\\"";
        break;
      case '\\':
        output << "\\\\";
        break;
      case '\b':
        output << "\\b";
        break;
      case '\f':
        output << "\\f";
        break;
      case '\n':
        output << "\\n";
        break;
      case '\r':
        output << "\\r";
        break;
      case '\t':
        output << "\\t";
        break;
      default:
        if (character < 0x20) {
          output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                 << static_cast<int>(character) << std::dec;
        } else {
          output << static_cast<char>(character);
        }
    }
  }
  return output.str();
}

class AsyncJsonlWriter {
 public:
  explicit AsyncJsonlWriter(const std::filesystem::path& path) {
    if (std::filesystem::exists(path)) {
      throw std::runtime_error("refusing to overwrite output");
    }
    if (!path.parent_path().empty()) {
      std::filesystem::create_directories(path.parent_path());
    }
    stream_.open(path, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!stream_) {
      throw std::runtime_error("failed to open output");
    }
  }

  ~AsyncJsonlWriter() {
    Stop();
  }

  void WriteMetadata(const std::string& line) {
    stream_ << line << '\n';
    stream_.flush();
    if (!stream_) {
      throw std::runtime_error("failed to write metadata");
    }
  }

  void Start() {
    worker_ = std::thread([this]() { Run(); });
  }

  void Enqueue(std::string line) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (queue_.size() >= kMaximumQueuedRecords) {
        dropped_records_.fetch_add(1);
        return;
      }
      queue_.push_back(std::move(line));
    }
    condition_.notify_one();
  }

  void Stop() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (stopping_) {
        return;
      }
      stopping_ = true;
    }
    condition_.notify_all();
    if (worker_.joinable()) {
      worker_.join();
    }
    if (stream_.is_open()) {
      stream_.flush();
      stream_.close();
    }
  }

  uint64_t dropped_records() const {
    return dropped_records_.load();
  }

 private:
  void Run() {
    while (true) {
      std::string line;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(
            lock, [this]() { return stopping_ || !queue_.empty(); });
        if (queue_.empty()) {
          if (stopping_) {
            break;
          }
          continue;
        }
        line = std::move(queue_.front());
        queue_.pop_front();
      }
      stream_ << line << '\n';
    }
  }

  std::ofstream stream_;
  std::mutex mutex_;
  std::condition_variable condition_;
  std::deque<std::string> queue_;
  std::thread worker_;
  bool stopping_{false};
  std::atomic<uint64_t> dropped_records_{0};
};

struct Options {
  std::string server_address;
  std::string local_address;
  std::filesystem::path output_path;
  int rigid_body_id{-1};
  std::string rigid_body_name;
  std::string calibration_id;
  std::string sdk_archive_sha256;
  std::string connection{"multicast"};
  std::string multicast_address{kDefaultMulticast};
  double duration_seconds{10.0};
};

void PrintUsage(std::ostream& stream, const char* executable) {
  stream
      << "Usage: " << executable
      << " --server IP --local IP --output FILE --rigid-body-id ID"
      << " --rigid-body-name NAME --calibration-id ID"
      << " --sdk-archive-sha256 HEX [--duration-sec N]"
      << " [--connection multicast|unicast]"
      << " [--multicast-address IP]\n";
}

Options ParseArguments(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--help" || argument == "-h") {
      PrintUsage(std::cout, argv[0]);
      std::exit(0);
    }
    if (index + 1 >= argc) {
      throw std::runtime_error("missing value for " + argument);
    }
    const std::string value = argv[++index];
    if (argument == "--server") {
      options.server_address = value;
    } else if (argument == "--local") {
      options.local_address = value;
    } else if (argument == "--output") {
      options.output_path = value;
    } else if (argument == "--rigid-body-id") {
      options.rigid_body_id = std::stoi(value);
    } else if (argument == "--rigid-body-name") {
      options.rigid_body_name = value;
    } else if (argument == "--calibration-id") {
      options.calibration_id = value;
    } else if (argument == "--sdk-archive-sha256") {
      options.sdk_archive_sha256 = value;
    } else if (argument == "--duration-sec") {
      options.duration_seconds = std::stod(value);
    } else if (argument == "--connection") {
      options.connection = value;
    } else if (argument == "--multicast-address") {
      options.multicast_address = value;
    } else {
      throw std::runtime_error("unknown argument: " + argument);
    }
  }
  if (options.server_address.empty() || options.local_address.empty() ||
      options.output_path.empty() || options.rigid_body_id < 0 ||
      options.rigid_body_name.empty() || options.calibration_id.empty() ||
      options.sdk_archive_sha256.size() != 64) {
    throw std::runtime_error("missing or invalid required argument");
  }
  if (options.duration_seconds <= 0.0 ||
      !std::isfinite(options.duration_seconds)) {
    throw std::runtime_error("duration must be finite and positive");
  }
  if (options.connection != "multicast" &&
      options.connection != "unicast") {
    throw std::runtime_error("connection must be multicast or unicast");
  }
  return options;
}

struct RecorderContext {
  NatNetClient* client{nullptr};
  AsyncJsonlWriter* writer{nullptr};
  int rigid_body_id{-1};
  uint64_t high_resolution_clock_frequency_hz{0};
  std::atomic<bool> enabled{false};
  std::atomic<uint64_t> frame_count{0};
  std::atomic<uint64_t> body_present_count{0};
  std::atomic<uint64_t> tracking_valid_count{0};
  std::atomic<uint64_t> callback_error_count{0};
};

void FrameHandlerImpl(
    sFrameOfMocapData* frame, void* user_context) {
  auto* context = static_cast<RecorderContext*>(user_context);
  if (context == nullptr || frame == nullptr ||
      !context->enabled.load() || context->writer == nullptr ||
      context->client == nullptr) {
    return;
  }

  const int64_t receipt_realtime_ns = ClockNanoseconds(CLOCK_REALTIME);
  const int64_t receipt_monotonic_ns = ClockNanoseconds(CLOCK_MONOTONIC_RAW);
  const sRigidBodyData* selected = nullptr;
  for (int index = 0; index < frame->nRigidBodies; ++index) {
    if (frame->RigidBodies[index].ID == context->rigid_body_id) {
      selected = &frame->RigidBodies[index];
      break;
    }
  }

  std::optional<double> seconds_since_host_mid_exposure;
  std::optional<int64_t> capture_realtime_estimate_ns;
  if (frame->CameraMidExposureTimestamp != 0) {
    const double seconds_since =
        context->client->SecondsSinceHostTimestamp(
            frame->CameraMidExposureTimestamp);
    if (std::isfinite(seconds_since) && seconds_since >= 0.0 &&
        seconds_since < 10.0) {
      seconds_since_host_mid_exposure = seconds_since;
      capture_realtime_estimate_ns =
          receipt_realtime_ns -
          static_cast<int64_t>(std::llround(seconds_since * 1.0e9));
    }
  }

  const bool body_present = selected != nullptr;
  const bool tracking_valid =
      body_present && ((selected->params & 0x01) != 0);
  const bool geometry_finite =
      body_present && std::isfinite(selected->MeanError) &&
      std::isfinite(selected->x) && std::isfinite(selected->y) &&
      std::isfinite(selected->z) && std::isfinite(selected->qx) &&
      std::isfinite(selected->qy) && std::isfinite(selected->qz) &&
      std::isfinite(selected->qw);
  context->frame_count.fetch_add(1);
  if (body_present) {
    context->body_present_count.fetch_add(1);
  }
  if (tracking_valid) {
    context->tracking_valid_count.fetch_add(1);
  }

  std::ostringstream line;
  line << std::setprecision(17);
  line << "{\"schema\":\"" << kSchema
       << "\",\"kind\":\"raw_rigid_body_pose\""
       << ",\"frame_number\":" << frame->iFrame
       << ",\"motive_software_time_s\":";
  if (std::isfinite(frame->fTimestamp)) {
    line << frame->fTimestamp;
  } else {
    line << "null";
  }
  line << ",\"camera_mid_exposure_ticks\":"
       << frame->CameraMidExposureTimestamp
       << ",\"camera_data_received_ticks\":"
       << frame->CameraDataReceivedTimestamp
       << ",\"transmit_ticks\":" << frame->TransmitTimestamp
       << ",\"precision_timestamp_seconds\":"
       << frame->PrecisionTimestampSecs
       << ",\"precision_timestamp_fractional_seconds\":"
       << frame->PrecisionTimestampFractionalSecs
       << ",\"receipt_realtime_ns\":" << receipt_realtime_ns
       << ",\"receipt_monotonic_ns\":" << receipt_monotonic_ns
       << ",\"high_resolution_clock_frequency_hz\":"
       << context->high_resolution_clock_frequency_hz
       << ",\"seconds_since_host_mid_exposure\":";
  if (seconds_since_host_mid_exposure.has_value()) {
    line << seconds_since_host_mid_exposure.value();
  } else {
    line << "null";
  }
  line << ",\"capture_realtime_estimate_ns\":";
  if (capture_realtime_estimate_ns.has_value()) {
    line << capture_realtime_estimate_ns.value();
  } else {
    line << "null";
  }
  line << ",\"rigid_body_id\":" << context->rigid_body_id
       << ",\"rigid_body_present\":" << (body_present ? "true" : "false")
       << ",\"tracking_valid\":" << (tracking_valid ? "true" : "false")
       << ",\"geometry_finite\":" << (geometry_finite ? "true" : "false")
       << ",\"mean_marker_error_m\":";
  if (geometry_finite) {
    line << selected->MeanError;
  } else {
    line << "null";
  }
  line << ",\"position_motive_xyz_m\":";
  if (geometry_finite) {
    line << '[' << selected->x << ',' << selected->y << ',' << selected->z
         << ']';
  } else {
    line << "null";
  }
  line << ",\"quaternion_motive_xyzw\":";
  if (geometry_finite) {
    line << '[' << selected->qx << ',' << selected->qy << ',' << selected->qz
         << ',' << selected->qw << ']';
  } else {
    line << "null";
  }
  line << '}';
  context->writer->Enqueue(line.str());
}

void NATNET_CALLCONV FrameHandler(
    sFrameOfMocapData* frame, void* user_context) {
  auto* context = static_cast<RecorderContext*>(user_context);
  try {
    FrameHandlerImpl(frame, user_context);
  } catch (...) {
    if (context != nullptr) {
      context->callback_error_count.fetch_add(1);
      context->enabled.store(false);
    }
  }
}

std::string MetadataLine(
    const Options& options,
    const sServerDescription& server,
    const unsigned char sdk_version[4]) {
  std::ostringstream line;
  line << "{\"schema\":\"" << kSchema << "\",\"kind\":\"metadata\""
       << ",\"pose_semantics\":\"rigid_body_pose_raw\""
       << ",\"coordinate_frame\":\"motive_world_native\""
       << ",\"quaternion_order\":\"xyzw\""
       << ",\"position_units\":\"meters\""
       << ",\"source\":\"optitrack_natnet\""
       << ",\"natnet_sdk_family\":\"" << kPinnedSdk << "\""
       << ",\"natnet_sdk_version\":\""
       << static_cast<int>(sdk_version[0]) << '.'
       << static_cast<int>(sdk_version[1]) << '.'
       << static_cast<int>(sdk_version[2]) << '.'
       << static_cast<int>(sdk_version[3]) << "\""
       << ",\"natnet_sdk_archive_sha256\":\""
       << JsonEscape(options.sdk_archive_sha256) << "\""
       << ",\"server_address\":\""
       << JsonEscape(options.server_address) << "\""
       << ",\"local_address\":\"" << JsonEscape(options.local_address) << "\""
       << ",\"connection\":\"" << JsonEscape(options.connection) << "\""
       << ",\"multicast_address\":\""
       << JsonEscape(options.multicast_address) << "\""
       << ",\"server_application\":\"" << JsonEscape(server.szHostApp) << "\""
       << ",\"server_computer\":\""
       << JsonEscape(server.szHostComputerName) << "\""
       << ",\"server_natnet_version\":\""
       << static_cast<int>(server.NatNetVersion[0]) << '.'
       << static_cast<int>(server.NatNetVersion[1]) << '.'
       << static_cast<int>(server.NatNetVersion[2]) << '.'
       << static_cast<int>(server.NatNetVersion[3]) << "\""
       << ",\"high_resolution_clock_frequency_hz\":"
       << server.HighResClockFrequency
       << ",\"rigid_body_id\":" << options.rigid_body_id
       << ",\"rigid_body_name\":\""
       << JsonEscape(options.rigid_body_name) << "\""
       << ",\"calibration_id\":\""
       << JsonEscape(options.calibration_id) << "\""
       << ",\"capture_start_realtime_ns\":"
       << ClockNanoseconds(CLOCK_REALTIME)
       << ",\"capture_start_monotonic_ns\":"
       << ClockNanoseconds(CLOCK_MONOTONIC_RAW) << '}';
  return line.str();
}

std::string SummaryLine(
    const RecorderContext& context, const AsyncJsonlWriter& writer) {
  std::ostringstream line;
  line << "{\"schema\":\"" << kSchema << "\",\"kind\":\"summary\""
       << ",\"frame_count\":" << context.frame_count.load()
       << ",\"rigid_body_present_count\":"
       << context.body_present_count.load()
       << ",\"tracking_valid_count\":"
       << context.tracking_valid_count.load()
       << ",\"callback_error_count\":"
       << context.callback_error_count.load()
       << ",\"writer_drop_count\":" << writer.dropped_records()
       << ",\"capture_end_realtime_ns\":"
       << ClockNanoseconds(CLOCK_REALTIME)
       << ",\"capture_end_monotonic_ns\":"
       << ClockNanoseconds(CLOCK_MONOTONIC_RAW) << '}';
  return line.str();
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = ParseArguments(argc, argv);
    if (std::filesystem::exists(options.output_path)) {
      throw std::runtime_error("refusing to overwrite output");
    }

    std::signal(SIGINT, HandleSignal);
    std::signal(SIGTERM, HandleSignal);

    NatNetClient client;
    RecorderContext context;
    context.client = &client;
    context.rigid_body_id = options.rigid_body_id;
    if (client.SetFrameReceivedCallback(FrameHandler, &context) != ErrorCode_OK) {
      throw std::runtime_error("failed to install frame callback");
    }

    sNatNetClientConnectParams parameters;
    parameters.connectionType =
        options.connection == "multicast" ? ConnectionType_Multicast
                                            : ConnectionType_Unicast;
    parameters.serverAddress = options.server_address.c_str();
    parameters.localAddress = options.local_address.c_str();
    parameters.multicastAddress =
        options.connection == "multicast"
            ? options.multicast_address.c_str()
            : nullptr;
    const ErrorCode connect_result = client.Connect(parameters);
    if (connect_result != ErrorCode_OK) {
      std::ostringstream message;
      message << "NatNet connection failed with error "
              << static_cast<int>(connect_result);
      throw std::runtime_error(message.str());
    }

    sServerDescription server{};
    if (client.GetServerDescription(&server) != ErrorCode_OK ||
        !server.HostPresent) {
      client.Disconnect();
      throw std::runtime_error("NatNet server description is unavailable");
    }
    context.high_resolution_clock_frequency_hz =
        server.HighResClockFrequency;

    unsigned char sdk_version[4]{};
    NatNet_GetVersion(sdk_version);
    AsyncJsonlWriter writer(options.output_path);
    context.writer = &writer;
    writer.WriteMetadata(MetadataLine(options, server, sdk_version));
    writer.Start();
    context.enabled.store(true);

    const auto deadline =
        std::chrono::steady_clock::now() +
        std::chrono::duration<double>(options.duration_seconds);
    while (!g_stop_requested.load() &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    context.enabled.store(false);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    client.Disconnect();
    writer.Enqueue(SummaryLine(context, writer));
    writer.Stop();

    const bool complete =
        context.frame_count.load() > 0 &&
        context.body_present_count.load() > 0 &&
        context.callback_error_count.load() == 0 &&
        writer.dropped_records() == 0;
    std::cout << "frames=" << context.frame_count.load()
              << " body_present=" << context.body_present_count.load()
              << " tracking_valid=" << context.tracking_valid_count.load()
              << " callback_errors=" << context.callback_error_count.load()
              << " writer_drops=" << writer.dropped_records() << '\n';
    return complete ? 0 : 4;
  } catch (const std::exception& exception) {
    std::cerr << "optitrack_reference_recorder: " << exception.what() << '\n';
    PrintUsage(std::cerr, argv[0]);
    return 2;
  }
}
