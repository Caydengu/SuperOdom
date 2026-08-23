from g1_root_state_bridge.fault_matrix_cli import run_fault_matrix


def test_fault_matrix_has_zero_false_healthy_outputs() -> None:
    report = run_fault_matrix()
    assert report["status"] == "pass"
    assert report["fault_count"] >= 6
    assert report["false_healthy_count"] == 0
