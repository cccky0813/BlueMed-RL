"""影像工具的安全测试：image_id 校验与路径穿越防护（不加载模型）。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bluemed.tools.image_tools import analyze_image, resolve


def test_rejects_path_traversal():
    for bad in ["../../.env", "img_../../x", "/etc/passwd", "img_12345678/../../.env", ""]:
        assert resolve(bad) is None


def test_rejects_nonexistent_but_wellformed_id():
    assert resolve("img_00000000") is None


def test_tool_returns_friendly_message_for_bad_id():
    content, artifact = analyze_image.func("../../.env")
    assert "找不到图片" in content and artifact == []
