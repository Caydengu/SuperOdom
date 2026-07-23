# Latched, typed frame contract for /state_estimation.
# Transform names use A_T_B: coordinates in B are mapped into A.
std_msgs/Header header

string semantics_version
string map_frame
string sensor_frame
string gravity_frame

# Rotation-only startup alignment. The gravity frame is z-up and preserves
# map +X projected onto the horizontal plane; translation must be zero.
geometry_msgs/Transform map_t_gravity

# Static Mid-360 calibration used by the estimator (IMU_T_LiDAR).
geometry_msgs/Transform imu_t_lidar

# Increments whenever the propagated estimator is reinitialized.
uint64 reset_id
bool valid
