from pathlib import Path

from splatfusion.cli import build_manifest, collect_images


def test_collect_images_filters_and_sorts(tmp_path: Path):
    (tmp_path / "b.JPG").write_bytes(b"")
    (tmp_path / "a.png").write_bytes(b"")
    (tmp_path / "notes.txt").write_text("ignore")
    assert [p.name for p in collect_images(tmp_path)] == ["a.png", "b.JPG"]


def test_manifest_records_pending_views(tmp_path: Path):
    image = tmp_path / "front.jpg"
    image.write_bytes(b"")
    manifest = build_manifest([image], tmp_path / "out")
    assert manifest["stage"] == "input_indexed"
    assert manifest["views"][0]["sharp_ply"] is None
    assert manifest["views"][0]["camera_to_world"] is None
