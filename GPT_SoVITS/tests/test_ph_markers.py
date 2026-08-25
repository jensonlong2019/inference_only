"""Prepared clip markers: 【PH试纸】 must splice as one clip, not go to TTS."""
import ast
import re
import unittest
from pathlib import Path

WEBROOT = Path(__file__).resolve().parents[1]


def _load():
    src = (WEBROOT / "inference_webui.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    wanted_fn = {"split_by_ph_marker"}
    wanted_assign = {"PREPARED_CLIP_MARKERS", "PH_MARKER"}
    chunks = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if any(n in wanted_assign for n in names):
                chunks.append(ast.get_source_segment(src, node))
        if isinstance(node, ast.FunctionDef) and node.name in wanted_fn:
            chunks.append(ast.get_source_segment(src, node))
    ns = {"re": re}
    exec("\n\n".join(chunks), ns, ns)
    return ns


_NS = _load()
split_by_ph_marker = _NS["split_by_ph_marker"]


class SplitPreparedMarkersTests(unittest.TestCase):
    def test_ph_shizhi_is_single_clip(self):
        segs = split_by_ph_marker("把【PH试纸】浸入液体")
        self.assertEqual(
            segs,
            [("text", "把"), ("ph", "【PH试纸】"), ("text", "浸入液体")],
        )

    def test_ph_marker_still_splits(self):
        segs = split_by_ph_marker("【PH】等于7")
        self.assertEqual(segs, [("ph", "【PH】"), ("text", "等于7")])

    def test_both_markers_prefer_longer_first(self):
        segs = split_by_ph_marker("【PH试纸】和【PH】")
        self.assertEqual(
            segs,
            [("ph", "【PH试纸】"), ("text", "和"), ("ph", "【PH】")],
        )


if __name__ == "__main__":
    unittest.main()
