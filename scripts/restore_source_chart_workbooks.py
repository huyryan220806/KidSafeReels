"""Preserve unchanged source chart parts lost by Artifact Tool import/export.

Only for the two benchmark charts in the supplied 11-slide deck. Their values,
formulas, styles, externalData relationships and embedded workbooks stay byte-identical.
"""
import argparse
import posixpath
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

REL = "http://schemas.openxmlformats.org/package/2006/relationships"


def chart_target(bundle, slide):
    rels = ET.fromstring(bundle.read(f"ppt/slides/_rels/slide{slide}.xml.rels"))
    targets = [item.attrib["Target"] for item in rels if item.attrib["Type"].endswith("/chart")]
    if len(targets) != 1:
        raise ValueError("Expected one benchmark chart on source slide")
    return targets[0].lstrip("/") if targets[0].startswith("/") else posixpath.normpath("ppt/slides/" + targets[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    with zipfile.ZipFile(args.source) as source, zipfile.ZipFile(args.candidate) as candidate:
        for slide in (5, 6):
            if chart_target(source, slide) != chart_target(candidate, slide):
                raise ValueError("Chart target mapping changed; cannot restore blindly")
        names = [name for name in source.namelist()
                 if name.startswith(("ppt/slides/charts/", "ppt/embeddings/"))]
        preserved = {name: source.read(name) for name in names}
        content = {name: candidate.read(name) for name in candidate.namelist()}
        content.update(preserved)
        types = ET.fromstring(content["[Content_Types].xml"])
        source_types = ET.fromstring(source.read("[Content_Types].xml"))
        existing = {(item.tag, item.attrib.get("PartName", item.attrib.get("Extension"))) for item in types}
        for item in source_types:
            part = item.attrib.get("PartName", "").lstrip("/")
            key = (item.tag, item.attrib.get("PartName", item.attrib.get("Extension")))
            if (part in preserved or item.attrib.get("Extension") == "xlsx") and key not in existing:
                types.append(item)
                existing.add(key)
        ET.register_namespace("", "http://schemas.openxmlformats.org/package/2006/content-types")
        content["[Content_Types].xml"] = ET.tostring(types, encoding="utf-8", xml_declaration=True)
    temporary = args.candidate.with_suffix(".restored.pptx")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as output:
        for name, value in content.items():
            output.writestr(name, value)
    temporary.replace(args.candidate)
    print(f"Preserved {len(preserved)} unchanged source chart/workbook parts")


if __name__ == "__main__":
    main()
