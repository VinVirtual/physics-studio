"""What a robot file can reach: paths, XML declarations, refused MJCF elements, mesh files."""
import os

import pytest

from ps import RecipeError
from ps.safety import AssetStore, check_mjcf_elements, inside, parse_xml_bytes


def test_inside_accepts_plain_paths(tmp_path):
    (tmp_path / "a").mkdir()
    assert inside(tmp_path, "a/b.stl") == (tmp_path / "a" / "b.stl").resolve()


@pytest.mark.parametrize("rel", ["/etc/passwd", "../x", "a/../../x", "~/x", "file:///etc/passwd", "C:\\x",
                                 "package://../x", "", "a\x00b"])
def test_inside_refuses_escapes(tmp_path, rel):
    with pytest.raises(RecipeError):
        inside(tmp_path, rel)


def test_inside_refuses_symlink_out(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.stl").write_text("solid x")
    base = tmp_path / "robot"
    base.mkdir()
    os.symlink(outside, base / "link")
    with pytest.raises(RecipeError, match="outside"):
        inside(base, "link/secret.stl")


def test_doctype_and_entities_refused():
    for doc in (b'<?xml version="1.0"?><!DOCTYPE r [<!ENTITY x SYSTEM "file:///etc/passwd">]><robot>&x;</robot>',
                b'<!doctype robot><robot/>', b'<robot><!ENTITY a "b"></robot>'):
        with pytest.raises(RecipeError, match="DOCTYPE"):
            parse_xml_bytes(doc)


def test_bad_xml():
    with pytest.raises(RecipeError, match="not valid XML"):
        parse_xml_bytes(b"<robot><link></robot>")


@pytest.mark.parametrize("tag", ["include", "plugin", "extension", "model", "skin", "instance", "flexcomp"])
def test_refused_mjcf_elements(tag):
    root = parse_xml_bytes(f'<mujoco><worldbody><body><{tag} file="x"/></body></worldbody></mujoco>'.encode())
    with pytest.raises(RecipeError, match=f"<{tag}>"):
        check_mjcf_elements(root)


def test_asset_store_renames_and_dedupes(tmp_path):
    (tmp_path / "m").mkdir()
    (tmp_path / "m" / "arm link.STL").write_bytes(b"solid a\nendsolid a\n")
    s = AssetStore(tmp_path)
    n1 = s.add("m/arm link.STL", (".stl",), "mesh")
    n2 = s.add("m/./arm link.STL", (".stl",), "mesh")
    assert n1 == n2 == "a000_arm_link.stl" and list(s.files) == [n1]


def test_asset_store_refuses_wrong_type_and_missing(tmp_path):
    (tmp_path / "x.so").write_bytes(b"\x7fELF")
    s = AssetStore(tmp_path)
    with pytest.raises(RecipeError, match="only .stl"):
        s.add("x.so", (".stl",), "mesh")
    with pytest.raises(RecipeError, match="not found"):
        s.add("none.stl", (".stl",), "mesh")


def test_asset_store_size_cap(tmp_path, monkeypatch):
    import ps.safety as sf
    monkeypatch.setattr(sf, "MAX_ASSET_BYTES", 10)
    (tmp_path / "big.stl").write_bytes(b"x" * 11)
    with pytest.raises(RecipeError, match="larger than"):
        sf.AssetStore(tmp_path).add("big.stl", (".stl",), "mesh")


def test_asset_store_count_cap(tmp_path, monkeypatch):
    import ps.safety as sf
    monkeypatch.setattr(sf, "MAX_ASSETS", 2)
    s = sf.AssetStore(tmp_path)
    for i in range(3):
        (tmp_path / f"{i}.stl").write_bytes(b"solid")
    s.add("0.stl", (".stl",), "mesh")
    s.add("1.stl", (".stl",), "mesh")
    with pytest.raises(RecipeError, match="too many"):
        s.add("2.stl", (".stl",), "mesh")


def test_utf16_and_nul_refused():
    doc = '<?xml version="1.0" encoding="UTF-16"?><!DOCTYPE r [<!ENTITY a "x">]><robot>&a;</robot>'
    with pytest.raises(RecipeError, match="UTF-8"):
        parse_xml_bytes(b"\xff\xfe" + doc.encode("utf-16-le"))
    with pytest.raises(RecipeError):
        parse_xml_bytes(b"<robot>\x00</robot>")
    with pytest.raises(RecipeError, match="only UTF-8"):
        parse_xml_bytes(b'<?xml version="1.0" encoding="latin-1"?><robot/>')


@pytest.mark.parametrize("xml, needle", [
    ('<asset><texture name="t" type="cube" fileright="/etc/hostname"/></asset>', "fileright"),
    ('<worldbody><replicate count="3000"><body/></replicate></worldbody>', "<replicate>"),
    ('<worldbody><composite type="grid" count="99 99 1"/></worldbody>', "<composite>"),
    ('<asset><texture name="t" type="2d" builtin="flat" width="30000" height="30000"/></asset>', "larger than"),
    ('<asset><hfield name="h" nrow="5000" ncol="5000" size="1 1 1 1"/></asset>', "larger than"),
    ('<worldbody><body><spaceship/></body></worldbody>', "not supported"),
])
def test_mjcf_resource_and_file_attacks_refused(xml, needle):
    root = parse_xml_bytes(f"<mujoco>{xml}</mujoco>".encode())
    with pytest.raises(RecipeError, match=needle):
        check_mjcf_elements(root)


def test_too_many_bodies_refused():
    root = parse_xml_bytes(("<mujoco><worldbody>" + "<body/>" * 401 + "</worldbody></mujoco>").encode())
    with pytest.raises(RecipeError, match="bodies"):
        check_mjcf_elements(root)
