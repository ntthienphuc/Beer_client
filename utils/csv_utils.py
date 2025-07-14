#utils/csv_utils.py

import csv
from pathlib import Path
import re, datetime as dt
import os


MENU_PATH = Path("data/menu.csv")

def load_menu() -> dict[str, float]:
    """
    Đọc data/menu.csv trả về {name: price}
    """
    menu = {}
    with open(MENU_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            menu[row["name"]] = float(row["price"])
    return menu

def init_bill(table_id: int, session_id: str) -> Path:
    """
    Tạo file mới data/bills/bill_<table>_<session>.csv,
    với header item,qty,unit_price,total_line
    """
    bills_dir = Path("data/bills")
    bills_dir.mkdir(parents=True, exist_ok=True)
    path = bills_dir / f"bill_{table_id}_{session_id}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["item","qty","unit_price","total_line"])
    return path

def append_bill(path: Path, item: str, qty: int, unit_price: float) -> None:
    """
    Ghi tiếp 1 dòng vào bill hiện tại
    """
    if qty == 0:
        return
    total_line = qty * unit_price
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([item, qty, unit_price, total_line])

def save_menu_rows(rows: list[dict[str, str | float]]) -> None:
    """
    Ghi nguyên danh sách rows (mỗi row = {'name':..., 'price':...})
    trở lại data/menu.csv, tự động đánh lại cột id liên tục từ 1.
    """
    with open(MENU_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "name", "price"])
        for idx, r in enumerate(rows, start=1):
            writer.writerow([idx, r["name"], int(r["price"])])

BILL_RE = re.compile(
    r"bill_(\d+)_"                 # table_id
    r"(\d{1,2}-\d{2})_"            # start HH-MM
    r"(?:(\d{1,2}-\d{2})_)?"       # optional end HH-MM
    r"(\d{1,2}-\d{1,2}-\d{4})"     # date dd-mm-YYYY
)

def parse_bill_name(fname: str):
    """
    Trả về dict {table,start,end,date} hoặc None nếu sai định dạng.
    """
    m = BILL_RE.fullmatch(Path(fname).stem)
    if not m:
        return None
    table, start, end, date = m.groups()
    return {
        "table": int(table),
        "start": start,
        "end":   end,
        "date":  date
    }

def bill_total(path: Path) -> float:
    """
    Tính tổng tiền của bill CSV.
    """
    total = 0.0
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            total += float(row["total_line"])
    return total

def list_bills(dir_path: Path) -> list[Path]:
    """
    Lấy toàn bộ bill_*.csv trong thư mục, đã sort theo thời gian mới → cũ.
    """
    return sorted(dir_path.glob("bill_*.csv"), key=os.path.getmtime, reverse=True)