from pathlib import Path
import csv

def api_to_filename(api: str) -> str:
    name = api.strip()
    if not name:
        return ""
    # 去掉 torch. 前缀只是为了文件名更短；你也可以保留
    if name.startswith("torch."):
        name = name[len("torch."):]
    name = name.replace(".", "_")
    return f"test_{name}.py"

txt_path = Path("apis.txt")
csv_path = Path("api_manifest.csv")

rows = []
for line in txt_path.read_text(encoding="utf-8").splitlines():
    api = line.strip()
    if not api or api.startswith("#"):
        continue
    rows.append({
        "raw_api_name": api,
        "canonical_name": api,
        "file_name": api_to_filename(api),
        "status": "pending",
        "notes": "",
    })

with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["raw_api_name", "canonical_name", "file_name", "status", "notes"]
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"written: {csv_path} ({len(rows)} rows)")