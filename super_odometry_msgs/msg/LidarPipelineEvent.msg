uint8 RAW_RECEIVED=1
uint8 RAW_SKIPPED=2
uint8 FEATURE_PUBLISHED=3
uint8 MAPPING_RECEIVED=4
uint8 MAPPING_QUEUE_DROPPED=5
uint8 CORRECTION_PUBLISHED=6
uint8 MAPPING_REJECTED=7

std_msgs/Header header
string semantics_version
uint8 event_type
uint64 input_count
uint64 output_count
uint64 dropped_count
uint32 queue_depth
builtin_interfaces/Time scan_reference_stamp
builtin_interfaces/Time newest_observation_stamp
