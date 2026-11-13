"""SVN working-copy metadata only. No URL targets and no network flags."""

from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET


def parse_svn(info_xml: str, status_xml: str, root: Path):
    files = {}

    def relative(value):
        path = Path(value)
        if not path.is_absolute():
            path = root / path
        try:
            return path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            return None

    for entry in ET.fromstring(info_xml).iter("entry"):
        name = relative(entry.attrib["path"])
        if name is None:
            continue
        files[name] = {
            "revision": entry.get("revision"),
            "last_changed_revision": entry.find("commit").get("revision")
            if entry.find("commit") is not None else None,
            "local_status": "normal", "property_status": "normal",
            "availability": "available",
        }
    for entry in ET.fromstring(status_xml).iter("entry"):
        name = relative(entry.attrib["path"])
        status = entry.find("wc-status")
        if name is None or status is None:
            continue
        item = files.setdefault(name, {
            "revision": status.get("revision"), "last_changed_revision": None,
            "availability": "available",
        })
        item.update(local_status=status.get("item"), property_status=status.get("props"))
    root_entry = ET.fromstring(info_xml).find("entry")
    if root_entry is None:
        return {}, {"availability": "unavailable", "reason": "empty-local-svn-metadata"}
    return files, {
        "availability": "available",
        "url": root_entry.findtext("url") if root_entry is not None else None,
        "network_access": False,
    }


def read_svn(root: Path):
    if not (root / ".svn").is_dir():
        return {}, {"availability": "unavailable", "reason": "no-working-copy-metadata"}
    try:
        def run(args):
            return subprocess.run(
                ["svn", *args, "."], cwd=root, check=True, capture_output=True,
                text=True, encoding="utf-8", timeout=120,
            ).stdout

        info = run(["info", "--xml", "--depth", "infinity"])
        status = run(["status", "--xml", "--ignore-externals", "--depth", "infinity"])
        return parse_svn(info, status, root)
    except (OSError, subprocess.SubprocessError, ET.ParseError):
        # A failed metadata lookup must never imply a clean working copy.
        return {}, {"availability": "unavailable", "reason": "local-svn-query-failed"}


def file_svn(path: str, files: dict, summary: dict):
    if path in files:
        return files[path]
    if summary["availability"] == "available":
        return {"availability": "available", "revision": None,
                "last_changed_revision": None, "local_status": "unversioned",
                "property_status": None}
    return {"availability": "unavailable", "revision": None,
            "last_changed_revision": None, "local_status": "unknown",
            "property_status": None}
