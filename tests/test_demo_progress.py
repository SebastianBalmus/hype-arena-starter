"""Live progress must describe real outputs without changing inference."""
import contextlib
import csv
import io
import json
import os
import shutil
import unittest
from unittest.mock import patch

from demo.server import Demo, ROOT, make_handler


def handler_for(demo, path, rows, mode="full"):
    handler_cls = make_handler(demo)
    handler = handler_cls.__new__(handler_cls)
    body = json.dumps({"rows": rows, "mode": mode}).encode()
    handler.path = path
    handler.headers = {"Content-Length": str(len(body))}
    handler.rfile = io.BytesIO(body)
    handler.wfile = io.BytesIO()
    handler.request_version = "HTTP/1.1"
    handler.requestline = "POST " + path + " HTTP/1.1"
    handler.command = "POST"
    handler.client_address = ("127.0.0.1", 1234)
    return handler


class TestDemoProgress(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.demo = Demo(os.path.join(ROOT, "data/practice.csv"), os.path.join(ROOT, "data/lexicon.json"),
                        os.path.join(ROOT, "hooly/assets/calibration.json"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.demo.out_dir)

    def read_export(self, url, filename="output.csv"):
        handler = handler_for(self.demo, url, [])
        handler.do_GET()
        headers, body = handler.wfile.getvalue().split(b"\r\n\r\n", 1)
        self.assertIn(b"200", headers.splitlines()[0])
        self.assertIn(f'filename="{filename}"'.encode(), headers)
        return list(csv.DictReader(io.StringIO(body.decode())))

    def test_classification_skips_writer_and_keeps_the_full_export(self):
        full = self.demo.run(self.demo.sample)
        full_before = self.read_export(full["meta"]["csv_url"])
        events = []
        with patch("demo.server.generate_claims", side_effect=AssertionError("classification called writer")):
            classified = self.demo.run(self.demo.sample, mode="classify", progress=events.append)
        self.assertEqual(classified["rows"], full["rows"])
        self.assertEqual(classified["meta"]["claims"], 0)
        self.assertFalse({"writer", "referee", "claims"} & {e["node"] for e in events})
        exported = self.read_export(classified["meta"]["csv_url"], "classifications.csv")
        self.assertEqual(len(exported), len(self.demo.sample))
        self.assertTrue(all(not r["type"] for r in exported))
        self.assertEqual(full_before, self.read_export(full["meta"]["csv_url"]))
        self.assertEqual(sum(bool(r["type"]) for r in full_before), 1000)

    def test_generation_works_without_input_and_has_no_classifications(self):
        events = []
        result = self.demo.run([], mode="generate", progress=events.append)
        self.assertEqual(result["rows"], [])
        self.assertEqual(result["meta"]["mode"], "generate")
        self.assertEqual(result["meta"]["claims"], 1000)
        self.assertFalse({"calib", "labels"} & {e["node"] for e in events})
        exported = self.read_export(result["meta"]["csv_url"], "claims.csv")
        self.assertEqual(len(exported), 1000)
        self.assertTrue(all(r["type"] for r in exported))
        self.assertEqual(len(self.demo.claims()), 1000)

    def test_generation_shortfall_does_not_publish_an_incomplete_export(self):
        before = dict(self.demo.exports)
        with patch("demo.server.generate_claims", return_value=[]):
            with self.assertRaisesRegex(ValueError, "generated 0 of 1000"):
                self.demo.run([], mode="generate")
        self.assertEqual(self.demo.exports, before)

    def test_http_classification_mode_is_forwarded(self):
        handler = handler_for(self.demo, "/api/run?stream=1", self.demo.sample, "classify")
        handler.do_POST()
        messages = [json.loads(line) for line in handler.wfile.getvalue().split(b"\r\n\r\n", 1)[1].splitlines()]
        self.assertEqual(messages[-1]["result"]["meta"]["mode"], "classify")
        self.assertFalse(any(e.get("node") == "writer" for e in messages))

    def test_progress_is_serializable_and_preserves_predictions_and_csv(self):
        events = []
        def record(event):
            json.dumps(event)  # tuple-keyed internals cannot leak into JSON
            events.append(event)
        traced = self.demo.run(self.demo.sample, progress=record)
        path = os.path.join(self.demo.out_dir, "output.csv")
        with open(path, "rb") as f:
            csv_before = f.read()
        ordinary = self.demo.run(self.demo.sample)
        with open(path, "rb") as f:
            self.assertEqual(csv_before, f.read())
        self.assertEqual(traced["rows"], ordinary["rows"])
        expected = ["input", "lexicon", "practice", "entities", "reader", "facts", "truth", "calib",
                    "labels", "writer", "referee", "claims", "output"]
        self.assertEqual([(e["node"], e["status"]) for e in events],
                         [(node, status) for node in expected for status in ("running", "complete")])
        outputs = {e["node"]: e["output"] for e in events if e["status"] == "complete"}
        self.assertEqual(outputs["labels"]["rows"], traced["rows"])
        self.assertEqual(outputs["input"]["rows"], self.demo.sample)
        self.assertEqual(outputs["claims"]["total_claims"], 1000)
        self.assertEqual(outputs["referee"]["kept"], 1000)
        self.assertGreater(outputs["writer"]["total_candidates"], 1000)
        self.assertIn("probabilities", outputs["truth"]["rows"][0])
        self.assertIn("after", outputs["calib"]["rows"][0])
        self.assertEqual(outputs["output"]["bytes"], len(csv_before))
        self.assertEqual([e["elapsed_ms"] for e in events], sorted(e["elapsed_ms"] for e in events))

    def test_http_stream_has_flushed_stage_events_and_final_result(self):
        handler = handler_for(self.demo, "/api/run?stream=1", self.demo.sample)
        handler.do_POST()
        headers, body = handler.wfile.getvalue().split(b"\r\n\r\n", 1)
        self.assertIn(b"application/x-ndjson", headers)
        messages = [json.loads(line) for line in body.splitlines()]
        self.assertEqual(messages[0], {"type": "queued"})
        self.assertEqual(messages[-1]["type"], "result")
        self.assertEqual(messages[-1]["result"]["meta"]["claims"], 1000)
        self.assertEqual(messages[-2]["node"], "output")
        self.assertTrue(handler.close_connection)


class TestStreamFailures(unittest.TestCase):
    def test_failure_after_headers_is_an_error_event_not_another_http_response(self):
        class FailingDemo:
            def run(self, rows, progress, mode="full"):
                progress({"type": "stage", "node": "truth", "status": "running", "elapsed_ms": 0})
                raise RuntimeError("model failed")
        handler = handler_for(FailingDemo(), "/api/run?stream=1", [{"id": "a", "text": "x", "date": "2026-10-29"}])
        with contextlib.redirect_stderr(io.StringIO()):
            handler.do_POST()
        raw = handler.wfile.getvalue()
        self.assertEqual(raw.count(b"HTTP/1.0"), 1)
        messages = [json.loads(line) for line in raw.split(b"\r\n\r\n", 1)[1].splitlines()]
        self.assertEqual(messages[-1], {"type": "error", "error": "model failed"})

    def test_invalid_mode_and_empty_classification_are_rejected(self):
        for mode, rows in [("invalid", [{"id": "a", "text": "x"}]), ("classify", []), ("full", [])]:
            handler = handler_for(None, "/api/run", rows, mode)
            handler.do_POST()
            self.assertIn(b"400", handler.wfile.getvalue().splitlines()[0])


if __name__ == "__main__":
    unittest.main()
