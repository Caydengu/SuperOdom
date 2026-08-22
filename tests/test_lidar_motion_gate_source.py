from pathlib import Path


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "super_odometry"
    / "src"
    / "LidarProcess"
    / "LidarSlam.cpp"
)


def test_motion_gate_advances_pose_and_time_only_inside_acceptance_branch() -> None:
    source = SOURCE.read_text()
    start = source.index("void LidarSLAM::performPostOptimizationProcessing")
    end = source.index("bool LidarSLAM::checkMotionThresholds", start)
    function = source[start:end]
    acceptance = function.index("if (checkMotionThresholds")
    opening = function.index("{", acceptance)
    depth = 0
    closing = None
    for index in range(opening, len(function)):
        if function[index] == "{":
            depth += 1
        elif function[index] == "}":
            depth -= 1
            if depth == 0:
                closing = index
                break
    assert closing is not None
    accepted_body = function[opening:closing]

    assert "last_T_w_lidar = T_w_lidar;" in accepted_body
    assert "lasttimeLaserOdometry = timeLaserOdometry;" in accepted_body
    assert function.count("last_T_w_lidar = T_w_lidar;") == 1
    assert function.count("lasttimeLaserOdometry = timeLaserOdometry;") == 1


def test_motion_gate_has_no_unconditional_true_overwrite() -> None:
    source = SOURCE.read_text()
    start = source.index("bool LidarSLAM::checkMotionThresholds")
    end = source.index("void LidarSLAM::updateOptimizationStats", start)
    function = source[start:end]
    assert "acceptResult = true" not in function
    assert "delta_t <= 0.0 || !std::isfinite(delta_t)" in function
    assert "T_w_lidar = last_T_w_lidar;" in function
