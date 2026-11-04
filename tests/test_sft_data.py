"""SFT 数据筛选测试（不需要网络）。运行：python -m pytest tests/test_sft_data.py -q"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl.sft_data import SFT_FRAC, good, is_sft


def _row(total=1.0, invalid=0, final='{"status": "sufficient", "answer": ["鸡肝"]}'):
    return {"reward": {"total": total, "invalid": invalid}, "final": final}


def test_split_is_stable_and_about_a_quarter():
    ids = [f"train-{i:05d}" for i in range(8000)]
    share = sum(map(is_sft, ids)) / len(ids)
    assert abs(share - SFT_FRAC) < 0.02
    assert all(is_sft(i) == is_sft(i + "-oral") for i in ids[:200])      # 口语版和模板版同一边


def test_good_trajectory_filter():
    assert good(_row())
    assert good(_row(final='{"status": "insufficient", "missing": "没有饮食数据"}'))
    assert not good(_row(total=0.75))                                    # 分数不够
    assert not good(_row(invalid=1))                                     # 有非法调用：不教模型乱调
    assert not good(_row(final='查到了。\n{"status": "sufficient", "answer": ["鸡肝"]}'))   # 前面有废话
    assert not good(_row(final="```json\n{}\n```"))
    assert not good(_row(final=None))
