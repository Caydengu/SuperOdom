# Evidence that IMU preintegration accepted and applied one mapped correction.
# header.stamp is the correction pose reference (scan start).
std_msgs/Header header
string semantics_version
uint64 sequence
builtin_interfaces/Time newest_observation_stamp
builtin_interfaces/Time mapping_output_stamp
builtin_interfaces/Time application_stamp
uint64 reset_id
bool valid
