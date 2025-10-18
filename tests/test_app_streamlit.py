import ast
from pathlib import Path


APP_PATH = Path("app_streamlit.py")


def test_app_streamlit_no_roi_artifacts():
    text = APP_PATH.read_text(encoding="utf-8")
    assert "streamlit_drawable_canvas" not in text
    assert "_build_full_image_roi_map" not in text
    assert ".get(\"roi_full_map\"" not in text


def test_run_vision_eval_call_uses_three_arguments():
    tree = ast.parse(APP_PATH.read_text(encoding="utf-8"))
    call_args_counts = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "run_vision_eval":
                call_args_counts.append(len(node.args))
    assert call_args_counts, "run_vision_eval should be called in app_streamlit.py"
    assert all(count == 3 for count in call_args_counts)
