from datetime import UTC, datetime
from pathlib import Path

REPORT_SUFFIX = ".md"


def list_reports(reports_dir: Path) -> list[dict[str, str | int]]:
    if not reports_dir.is_dir():
        return []
    reports = [
        {
            "name": path.name,
            "size": path.stat().st_size,
            "modified": datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(),
        }
        for path in reports_dir.glob(f"*{REPORT_SUFFIX}")
        if path.is_file()
    ]
    return sorted(reports, key=lambda report: str(report["name"]), reverse=True)


def read_report(reports_dir: Path, name: str) -> dict[str, str]:
    path = reports_dir / name
    if Path(name).name != name or not name.endswith(REPORT_SUFFIX) or not path.is_file():
        raise FileNotFoundError(name)
    return {"name": name, "content": path.read_text(encoding="utf-8")}
