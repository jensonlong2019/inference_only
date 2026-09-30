"""Regression tests for batch preview checkbox/text merge.

Gradio DataFrame virtualization only keeps visible <tr> in the DOM (plus spacer
rows). Flags/texts collected from that DOM must be applied by filename, not by
positional index onto the full DataFrame.
"""
import ast
import json
import sys
import unittest
from pathlib import Path

import pandas as pd

WEBROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WEBROOT))


def _load_apply_fns():
    src_path = WEBROOT / "inference_webui.py"
    src = src_path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    wanted = {
        "_normalize_row_name",
        "_normalize_name_map",
        "_lookup_named_value",
        "_apply_process_flags_to_df",
        "_apply_preview_texts_to_df",
        "_results_sync_payload",
    }
    chunks = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            chunks.append(ast.get_source_segment(src, node))
    ns = {"json": json, "pd": pd, "_results_sync_seq": 0}
    exec("\n\n".join(chunks), ns, ns)
    return ns


_FNS = _load_apply_fns()
_apply_process_flags_to_df = _FNS["_apply_process_flags_to_df"]
_apply_preview_texts_to_df = _FNS["_apply_preview_texts_to_df"]
_results_sync_payload = _FNS["_results_sync_payload"]


def _sample_df():
    texts = [
        "实验原理：【PH】【试纸】含有多种酸碱指示剂成分。",
        "把变色后的试纸和标准比色卡对比。",
        "【PH】的数值一般在0到14之间：",
        "【PH】等于7是中性。",
        "【PH】小于7是酸性，数字越小酸性越强。",
        "【PH】大于7是碱性，数字越大碱性越强。",
    ]
    return pd.DataFrame(
        {
            "是否处理": [True] * 6,
            "序号": ["互动1"] * 6,
            "类型": ["点击播动画"] * 3 + ["录音题"] * 3,
            "命名": ["1", "2", "3", "4", "5", "6"],
            "配音内容": texts,
        }
    )


class ApplyProcessFlagsTests(unittest.TestCase):
    def test_by_name_selects_checked_rows_not_visible_slice(self):
        df = _sample_df()
        payload = json.dumps(
            {
                "version": 2,
                "default": False,
                "by_name": {"2": True, "3": True, "4": True, "5": True, "6": True},
            },
            ensure_ascii=False,
        )
        out = _apply_process_flags_to_df(df, payload, name_column="命名")
        selected = out.loc[out["是否处理"], "命名"].tolist()
        self.assertEqual(selected, ["2", "3", "4", "5", "6"])

    def test_uncheck_first_row_keeps_remaining_with_default_true(self):
        df = _sample_df()
        payload = json.dumps(
            {"version": 2, "default": True, "by_name": {"1": False}},
            ensure_ascii=False,
        )
        out = _apply_process_flags_to_df(df, payload, name_column="命名")
        selected = out.loc[out["是否处理"], "命名"].tolist()
        self.assertEqual(selected, ["2", "3", "4", "5", "6"])

    def test_visible_dom_slice_must_not_remap_onto_first_rows(self):
        """Reproduce the screenshot bug: spacer + visible rows 5/6 applied by index.

        Old behavior selected filenames 2 and 3 (and skipped 4/5/6).
        """
        df = _sample_df()
        # Virtual table DOM: spacer tr + two visible data rows
        legacy_visible_flags = json.dumps([False, True, True])
        out = _apply_process_flags_to_df(df, legacy_visible_flags, name_column="命名")
        selected = out.loc[out["是否处理"], "命名"].tolist()
        self.assertNotEqual(selected, ["2", "3"])
        self.assertEqual(selected, ["1", "2", "3", "4", "5", "6"])

    def test_integer_filename_matches_string_key(self):
        df = _sample_df()
        df["命名"] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        payload = json.dumps(
            {"version": 2, "default": False, "by_name": {"2": True, "3": True}},
            ensure_ascii=False,
        )
        out = _apply_process_flags_to_df(df, payload, name_column="命名")
        selected = [str(int(v)) if float(v).is_integer() else str(v) for v in out.loc[out["是否处理"], "命名"].tolist()]
        self.assertEqual(selected, ["2", "3"])

    def test_js_float_string_keys_match_integer_names(self):
        df = _sample_df()
        payload = json.dumps(
            {"version": 2, "default": False, "by_name": {"2.0": True, "3.0": True, "4": True}},
            ensure_ascii=False,
        )
        out = _apply_process_flags_to_df(df, payload, name_column="命名")
        selected = out.loc[out["是否处理"], "命名"].tolist()
        self.assertEqual(selected, ["2", "3", "4"])


class ApplyPreviewTextsTests(unittest.TestCase):
    def test_by_name_does_not_shift_text_onto_other_filenames(self):
        df = _sample_df()
        payload = json.dumps(
            {
                "version": 2,
                "by_name": {
                    "2": "把变色后的试纸和标准比色卡对比。（已改）",
                    "5": "【PH】小于7是酸性，数字越小酸性越强。",
                    "6": "【PH】大于7是碱性，数字越大碱性越强。",
                },
            },
            ensure_ascii=False,
        )
        out = _apply_preview_texts_to_df(df, payload, "配音内容", name_column="命名")
        self.assertEqual(out.loc[out["命名"] == "2", "配音内容"].iloc[0], "把变色后的试纸和标准比色卡对比。（已改）")
        self.assertEqual(
            out.loc[out["命名"] == "5", "配音内容"].iloc[0],
            "【PH】小于7是酸性，数字越小酸性越强。",
        )
        self.assertEqual(
            out.loc[out["命名"] == "6", "配音内容"].iloc[0],
            "【PH】大于7是碱性，数字越大碱性越强。",
        )

    def test_visible_dom_texts_must_not_overwrite_by_index(self):
        df = _sample_df()
        # spacer + visible rows 5 and 6 collected as a short list
        legacy = json.dumps(["", "【PH】小于7是酸性，数字越小酸性越强。", "【PH】大于7是碱性，数字越大碱性越强。"])
        out = _apply_preview_texts_to_df(df, legacy, "配音内容", name_column="命名")
        self.assertEqual(out.loc[out["命名"] == "2", "配音内容"].iloc[0], "把变色后的试纸和标准比色卡对比。")
        self.assertEqual(out.loc[out["命名"] == "3", "配音内容"].iloc[0], "【PH】的数值一般在0到14之间：")


class BatchResultsSyncTests(unittest.TestCase):
    def test_replace_payload_keeps_every_generated_row(self):
        rows = [
            {"filename": "2", "text": "a", "audio_url": "/file=2.mp3"},
            {"filename": "3", "text": "b", "audio_url": "/file=3.mp3"},
            {"filename": "4", "text": "c", "audio_url": "/file=4.mp3"},
            {"filename": "5", "text": "d", "audio_url": "/file=5.mp3"},
        ]
        payload = json.loads(_results_sync_payload("replace", rows=rows))
        self.assertEqual(payload["action"], "replace")
        self.assertEqual([r["filename"] for r in payload["rows"]], ["2", "3", "4", "5"])

    def test_sync_payload_keeps_every_row_generated_so_far(self):
        rows = [
            {"filename": "1", "text": "a", "audio_url": "/file=1.mp3"},
            {"filename": "2", "text": "b", "audio_url": "/file=2.mp3"},
        ]
        payload = json.loads(_results_sync_payload("sync", row=rows[-1], rows=rows))
        self.assertEqual(payload["action"], "sync")
        self.assertEqual(payload["row"]["filename"], "2")
        self.assertEqual([r["filename"] for r in payload["rows"]], ["1", "2"])

    def test_batch_generation_finishes_with_full_replace(self):
        src = (WEBROOT / "inference_webui.py").read_text(encoding="utf-8")
        done = [line for line in src.splitlines() if "处理完成" in line and "yield" in line]
        self.assertTrue(done, "missing completion yield")
        self.assertTrue(
            any('_results_sync_payload("replace", rows=results_list)' in line for line in done),
            "完成时必须把已生成的全部行 replace 回结果表，不能只用 gr.update() 留下部分 append",
        )
        self.assertIn(
            '_results_sync_payload("sync", row=item, rows=results_list)',
            src,
            "每生成一条都要带上迄今全部行，切走再回来不能只剩最后一条",
        )
        self.assertIn('data.action === "sync"', src)
        self.assertIn("rememberResultPayload(data)", src)


if __name__ == "__main__":
    unittest.main()
