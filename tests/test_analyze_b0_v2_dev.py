"""R1: dữ liệu tổng hợp; không đọc audio/checkpoint/test thật."""
from __future__ import annotations

import contextlib
import copy
import csv
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from scripts import analyze_b0_v2_dev as analysis
from src.metrics.evaluation import evaluate_development_scores


class B0DevelopmentAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = self.root/"closed_dev_full_v1.csv"
        types = ["bonafide"]*2+["voice_conversion"]*2+["adversarial_attack"]*2+["replay"]*2
        self.rows = [{"file": f"s{i%4}/file{i}.wav", "speaker_id": f"s{i%4}",
                      "utt_type": t, "binary_label": int(t != "bonafide"),
                      "split": "closed_dev", "shard": f"shard{i%2}.parquet"}
                     for i,t in enumerate(types)]
        with self.manifest.open("w",encoding="utf-8",newline="") as stream:
            writer = csv.DictWriter(stream,fieldnames=list(self.rows[0]))
            writer.writeheader();writer.writerows(self.rows)
        checksum = hashlib.sha256(self.manifest.read_bytes()).hexdigest()
        self.reports: dict[int,dict] = {}
        self.paths = {seed:self.root/f"b0_v2_ext_seed{seed}.json" for seed in analysis.SEEDS}
        for seed in analysis.SEEDS:
            scores = [.1,.4,.3,.9,.2 if seed == 2027 else .8,.9,.2,.9]
            records = [{"file":r["file"],"speaker_id":r["speaker_id"],"utt_type":r["utt_type"],
                        "label":r["binary_label"],"score":score}
                       for r,score in zip(self.rows,scores)]
            dev = asdict(evaluate_development_scores([r["label"] for r in records],
                           scores,loss=.7,utt_types=types))
            dev["records"] = records
            report = {"training_seed":seed,"data_seed":2026,"development_samples":8,
                "train_samples":4,"development_manifest_sha256":checksum,"use_test":False,
                "artifacts":{"development":{"source_manifest_sha256":checksum}},
                "source_tree_sha256":"a"*64,"status":"early_stopped",
                "controller":{"completed_epochs":5,"global_step":10,"bad_evaluations":2,
                "history":[{"epoch":e,"global_step":e*2,"eer":dev["eer"]+(0 if e == 3 else .1),
                            "loss":[.9,.8,.7,.9,1.][e-1]} for e in range(1,6)],
                "best":{"epoch":3,"global_step":6,"development":dev}}}
            # JSON roundtrip như report thật (tuple labels/scores thành list).
            self.reports[seed] = json.loads(json.dumps(report))
        self.save()

    def save(self) -> None:
        for seed,report in self.reports.items():
            self.paths[seed].write_text(json.dumps(report),encoding="utf-8")

    def analyze(self) -> dict:
        with contextlib.redirect_stdout(io.StringIO()):
            return analysis.analyze(self.paths,self.manifest,n_resamples=12)

    def test_attack_eer_and_sample_standard_deviation(self) -> None:
        result = self.analyze()
        self.assertEqual(result["seeds"][0]["per_attack_eer"]["voice_conversion"]["eer"],.5)
        self.assertEqual(result["seeds"][0]["per_attack_eer"]["adversarial_attack"]["eer"],0)
        self.assertEqual(result["seeds"][1]["per_attack_eer"]["adversarial_attack"]["eer"],.5)
        aggregate = result["per_attack_aggregate"]["adversarial_attack"]
        self.assertAlmostEqual(aggregate["mean"],1/6)
        self.assertAlmostEqual(aggregate["sample_std"],(1/12)**.5)
        self.assertEqual(aggregate["ddof"],1)
        self.assertEqual(analysis.mean_std([1.,2.,3.])["sample_std"],1)

    def test_confusion_errors_quantiles_and_bootstrap(self) -> None:
        result = self.analyze()
        seed = result["seeds"][0]
        cm = seed["confusion"]
        self.assertEqual(sum(cm[k] for k in ("tn","fp","fn","tp")),8)
        self.assertEqual(sum(g["errors"] for g in seed["errors_by_utt_type"].values()),
                         cm["fp"]+cm["fn"])
        self.assertAlmostEqual(seed["score_percentiles"]["bonafide"]["50"],.25)
        self.assertEqual(seed["speaker_bootstrap"]["requested_resamples"],12)
        self.assertGreater(seed["speaker_bootstrap"]["valid_resamples"],0)
        self.assertLessEqual(seed["speaker_bootstrap"]["valid_resamples"],12)
        self.assertEqual(seed["speaker_bootstrap"]["seed"],2026)
        self.assertEqual(seed["per_attack_eer"]["replay"]["interpretation"],"không đủ để kết luận")
        self.assertEqual(self.analyze(),result)

    def test_reads_exactly_four_allowlisted_inputs_once(self) -> None:
        allowed = {self.manifest.resolve(),*[p.resolve() for p in self.paths.values()]}
        seen = []
        original = Path.read_bytes
        def tracked(path: Path) -> bytes:
            self.assertIn(path.resolve(),allowed)
            seen.append(path.resolve())
            return original(path)
        with patch.object(Path,"read_bytes",tracked):
            result = self.analyze()
        self.assertEqual(set(seen),allowed)
        self.assertEqual(len(seen),4)
        self.assertEqual(len(result["inputs"]),4)
        self.assertTrue(all(len(x["sha256"])==64 for x in result["inputs"]))

    def test_rejects_test_path_without_opening_any_input(self) -> None:
        with patch.object(Path,"read_bytes",side_effect=AssertionError("Không được mở")) as read:
            with self.assertRaisesRegex(ValueError,"test"):
                analysis.analyze(self.paths,self.root/"closed_test_full_v1.csv")
            read.assert_not_called()

    def test_rejects_test_split_in_manifest(self) -> None:
        text = self.manifest.read_text().replace("closed_dev","closed_test")
        self.manifest.write_text(text,encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"closed_dev"):
            self.analyze()

    def test_rejects_test_metadata_in_report(self) -> None:
        self.reports[2026]["artifacts"]["development"]["split"] = "closed_test"
        self.save()
        with self.assertRaisesRegex(ValueError,"split test"):
            self.analyze()

    def test_rejects_order_or_labels_differing_from_manifest(self) -> None:
        original = copy.deepcopy(self.reports)
        self.reports[2026]["controller"]["best"]["development"]["records"].reverse()
        self.save()
        with self.assertRaisesRegex(ValueError,"thứ tự/nhãn"):
            self.analyze()
        self.reports = original
        dev = self.reports[2026]["controller"]["best"]["development"]
        dev["records"][0]["label"] = dev["labels"][0] = 1
        self.save()
        with self.assertRaisesRegex(ValueError,"thứ tự/nhãn"):
            self.analyze()

    def test_rejects_manifest_checksum_mismatch(self) -> None:
        self.reports[2026]["development_manifest_sha256"] = "0"*64
        self.save()
        with self.assertRaisesRegex(ValueError,"checksum"):
            self.analyze()

    def test_rejects_record_arrays_or_saved_metric_mismatch(self) -> None:
        original = copy.deepcopy(self.reports)
        self.reports[2026]["controller"]["best"]["development"]["labels"][0] = 1
        self.save()
        with self.assertRaisesRegex(ValueError,"mảng"):
            self.analyze()
        self.reports = original
        self.reports[2026]["controller"]["best"]["development"]["eer"] += .01
        self.save()
        with self.assertRaisesRegex(ValueError,"metric eer"):
            self.analyze()

    def test_svg_xml_labels_units_and_best_stop_markers(self) -> None:
        result = self.analyze()
        for render in (analysis.epoch_svg,analysis.attacks_svg):
            svg = render(result)
            node = ET.fromstring(svg)
            self.assertEqual(node.tag,"{http://www.w3.org/2000/svg}svg")
            self.assertIn("EER (%)",svg)
            self.assertIn("dev-full, không phải test",svg)
            self.assertIn("reports/b0_v2_dev_analysis.json",svg)
        self.assertIn("best theo EER",analysis.epoch_svg(result))
        self.assertIn("không đủ để kết luận",analysis.attacks_svg(result))
        self.assertIn("CI",analysis.markdown(result))

    def test_runtime_probe_shapes_and_parameter_count(self) -> None:
        probe = analysis.runtime_probe()
        self.assertEqual(probe["trainable_parameters"],498113)
        self.assertEqual(probe["lfcc_shape"],[1,60,401])
        self.assertEqual(probe["module_shapes"]["feature_extractor"],[1,128,7,50])
        self.assertEqual(probe["logit_shape"],[1])
        self.assertTrue(probe["finite_forward"])


if __name__ == "__main__":
    unittest.main()
