# Versioned scan timing carried atomically with one mapped LiDAR pose.
# odometry.header.stamp is the scan-start pose/deskew reference time.
string semantics_version
uint64 sequence
builtin_interfaces/Time newest_observation_stamp
builtin_interfaces/Time output_stamp
nav_msgs/Odometry odometry
