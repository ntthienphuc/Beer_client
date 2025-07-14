# app.py
import os, csv, datetime as dt, threading, time, logging
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

import cv2, numpy as np
from ttkbootstrap import Style, ttk
from ttkbootstrap.constants import *

# ── GPIO (chỉ có trên Raspberry Pi) ───────────────────
try:
    import RPi.GPIO as GPIO
    ON_PI = True
except (ImportError, RuntimeError):
    ON_PI = False

# ─── MODULE CỦA BẠN ───────────────────────────────────
from models.inference import TFLiteModel
from utils.csv_utils import (
    load_menu, init_bill, append_bill, save_menu_rows,
    parse_bill_name, bill_total, list_bills
)
# from utils.tcp_client import send_bill               # mở khi cần

# ─── LOGGING ──────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("sensor.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Sensor")

# ─── CẤU HÌNH CHUNG ───────────────────────────────────
MENU_FILE  = "data/menu.csv"
MODEL_FILE = "best.tflite"
TCP_HOST   = "192.168.1.16"
TCP_PORT   = 9000
BILLS_DIR  = Path("data/bills")

# Cảm biến & ảnh
SENSOR_PIN   = 17
IMAGE_PATH   = "/tmp/beer_temp.jpg"
CAP_CMD      = f"libcamera-still -o {IMAGE_PATH} --width 640 --height 480 -t 500 --nopreview"

# ─── TKINTER ──────────────────────────────────────────
style = Style(theme='litera')
root  = style.master
root.title("🍺 Beer Manager")

# ─── MODEL ────────────────────────────────────────────
if not os.path.isfile(MODEL_FILE):
    raise FileNotFoundError(f"Model not found: {MODEL_FILE}")
model = TFLiteModel(MODEL_FILE)

table_id_var = tk.IntVar(master=root, value=1)

# ─── NOTEBOOK ─────────────────────────────────────────
nb = ttk.Notebook(root)
order_frame, admin_frame, history_frame = ttk.Frame(nb), ttk.Frame(nb), ttk.Frame(nb)
nb.add(order_frame,  text="📝 ORDER")
nb.add(admin_frame,  text="🔧 ADMIN")
nb.add(history_frame, text="📜 HISTORY")
nb.pack(fill=BOTH, expand=True)

# ─── TIỆN ÍCH CHUNG ──────────────────────────────────
def read_image_unicode(path: str):
    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None

def add_tree_scroll(tree: ttk.Treeview, container: ttk.Frame, side=RIGHT):
    vsb = ttk.Scrollbar(container, orient=VERTICAL, command=tree.yview)
    tree.configure(yscrollcommand=vsb.set)
    vsb.pack(side=side, fill=Y)

# ─── ORDER TAB ────────────────────────────────────────
order_state = {"active": False}   # lưu thông tin phiên ORDER

def sensor_loop(stop_evt):
    """Luồng nền: đợi cảm biến, chụp ảnh, predict, thêm món."""
    if not ON_PI:
        return
    logger.info("Sensor thread started")
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(SENSOR_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    try:
        while not stop_evt.is_set() and order_state.get("active"):
            if GPIO.input(SENSOR_PIN) == GPIO.HIGH:
                logger.info("Object detected by sensor – capturing image")
                os.system(CAP_CMD)
                img = cv2.imread(IMAGE_PATH)
                if img is None:
                    logger.warning("Captured image not readable")
                else:
                    try:
                        name = model.predict(img)
                        logger.info("Prediction: %s", name)
                    except Exception as e:
                        logger.error("Predict error: %s", e)
                        name = None
                    if name:
                        root.after(0, lambda n=name: add_item_threadsafe(n))
                time.sleep(5)           # chống double-trigger
            time.sleep(0.1)
    finally:
        GPIO.cleanup()
        logger.info("Sensor thread stopped")

def start_session():
    menu = load_menu()
    if not menu:
        messagebox.showwarning("Menu trống", "Bạn cần thêm món trước.")
        return
    start_str = dt.datetime.now().strftime("%H-%M")
    date_str  = dt.datetime.now().strftime("%d-%m-%Y")
    tmp       = init_bill(table_id_var.get(), "tmp")
    bill_path = tmp.with_name(f"bill_{table_id_var.get()}_{start_str}_{date_str}.csv")
    try:
        os.rename(tmp, bill_path)
    except FileExistsError:
        os.remove(tmp)
    order_state.update({
        "active": True,
        "menu":   menu,
        "qtys":   {n: 0 for n in menu},
        "bill":   bill_path,
        "start":  start_str,
        "stop_evt": threading.Event(),
        "thread":  None
    })

    # khởi luồng cảm biến
    if ON_PI:
        th = threading.Thread(target=sensor_loop,
                              args=(order_state["stop_evt"],),
                              daemon=True)
        order_state["thread"] = th
        th.start()

    build_order()

def finish_session():
    # dừng sensor thread
    if order_state.get("thread"):
        order_state["stop_evt"].set()
        order_state["thread"].join(timeout=1)

    end_str = dt.datetime.now().strftime("%H-%M")
    bill_path = order_state["bill"]
    stem = bill_path.stem.split("_")
    if len(stem) == 4:        # bill_table_start_date.csv  →  _end_
        new = bill_path.with_name(
            "_".join(stem[:3] + [end_str] + stem[3:]) + bill_path.suffix
        )
        os.rename(bill_path, new)
        bill_path = new

    # send_bill(TCP_HOST, TCP_PORT, bill_path)   # mở khi cần
    messagebox.showinfo("Đã kết thúc", f"Bill lưu tại:\n{bill_path}")
    order_state.clear()
    order_state["active"] = False
    build_order()

def build_order():
    frame = order_frame
    for w in frame.winfo_children():
        w.destroy()

    # CHƯA CÓ PHIÊN
    if not order_state["active"]:
        ttk.Button(frame, text="🚀  Bắt đầu", width=20, bootstyle="success",
                   padding=10, command=start_session).pack(pady=80)
        return

    # ĐANG CÓ PHIÊN
    menu, qtys, bill_path, start_str = (
        order_state[k] for k in ("menu", "qtys", "bill", "start")
    )

    ttk.Label(frame, text=f"Bàn {table_id_var.get()} • Bắt đầu {start_str}",
              font=('Segoe UI Semibold', 18),
              bootstyle="inverse-primary", padding=10
              ).pack(fill=X, padx=10, pady=(10, 0))

    style.configure('Treeview', rowheight=28, font=('Segoe UI', 11))
    style.configure('Treeview.Heading', font=('Segoe UI Semibold', 12))

    # LEFT: danh sách món
    tree_fr = ttk.Frame(frame)
    tree_fr.pack(side=LEFT, fill=BOTH, expand=True, padx=(10, 0), pady=10)

    cols = ("name", "qty", "price")
    tree = ttk.Treeview(tree_fr, columns=cols, show="headings")
    for c, w in zip(cols, (200, 60, 100)):
        tree.heading(c, text={"name": "Món", "qty": "SL", "price": "Đơn giá"}[c])
        tree.column(c, width=w, anchor=CENTER if c == "qty" else W)
    for n, p in menu.items():
        tree.insert("", "end", iid=n, values=(n, qtys[n], f"{p:.0f}"))
    tree.pack(side=LEFT, fill=BOTH, expand=True)
    add_tree_scroll(tree, tree_fr)

    # RIGHT: Panel
    ctrl = ttk.Frame(frame, padding=10)
    ctrl.pack(side=LEFT, fill=Y, padx=10, pady=10)

    total_var = tk.StringVar()

    def update_total():
        total_var.set(f"Tổng: {sum(qtys[n]*menu[n] for n in qtys):,.0f} ₫")
    update_total()

    ttk.Label(ctrl, textvariable=total_var,
              font=('Segoe UI Semibold', 14),
              bootstyle="primary", padding=(0, 0, 0, 10)
              ).pack()

    # Qty control
    grp = ttk.LabelFrame(ctrl, text="Chỉnh SL", bootstyle="secondary")
    grp.pack(fill=X, pady=(0, 10))

    canvas = tk.Canvas(grp, height=300, highlightthickness=0)
    inner  = ttk.Frame(canvas)
    vsb    = ttk.Scrollbar(grp, orient=VERTICAL, command=canvas.yview)
    canvas.configure(yscrollcommand=vsb.set)
    vsb.pack(side=RIGHT, fill=Y)
    canvas.pack(side=LEFT, fill=BOTH, expand=True)
    canvas.create_window((0, 0), window=inner, anchor="nw")
    inner.bind("<Configure>",
               lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    # hàm thêm món an toàn từ mọi luồng
    def add_item_threadsafe(n):
        if not order_state["active"]:
            return
        if n not in menu:
            messagebox.showerror("Không khớp menu", f"'{n}' chưa có trong menu.")
            return
        qtys[n] += 1
        tree.set(n, "qty", qtys[n])
        append_bill(bill_path, n, qtys[n], menu[n])
        update_total()

    # gọi trực tiếp trong GUI
    def add_item(n):
        add_item_threadsafe(n)

    def sub_item(n):
        if qtys[n] == 0:
            return
        qtys[n] -= 1
        tree.set(n, "qty", qtys[n])
        append_bill(bill_path, n, qtys[n], menu[n])
        update_total()

    for n in menu:
        row = ttk.Frame(inner)
        row.pack(fill=X, pady=2)
        ttk.Label(row, text=n, width=20).pack(side=LEFT)
        ttk.Button(row, text="+", bootstyle="success", width=3,
                   command=lambda i=n: add_item(i)
                   ).pack(side=LEFT, padx=4)
        ttk.Button(row, text="-", bootstyle="danger", width=3,
                   command=lambda i=n: sub_item(i)
                   ).pack(side=LEFT)

    # upload ảnh tay
    def upload_image():
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png")])
        if not path:
            return
        img = read_image_unicode(path)
        if img is None:
            messagebox.showerror("Lỗi", f"Không đọc được ảnh:\n{path}")
            return
        name = model.predict(img)
        add_item(name)
        messagebox.showinfo("Đã thêm", name)

    ttk.Button(ctrl, text="📷 Upload Ảnh Bia", width=20,
               bootstyle="info", command=upload_image
               ).pack(pady=(0, 10))

    ttk.Button(ctrl, text="✅ Kết thúc", width=20,
               bootstyle="primary", command=finish_session
               ).pack()

# ─── ADMIN TAB ────────────────────────────────────────
def build_admin():
    frame = admin_frame
    for w in frame.winfo_children():
        w.destroy()

    rows = []
    with open(MENU_FILE, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({"name": r["name"], "price": float(r["price"])})

    def refresh():
        tree.delete(*tree.get_children())
        for i, r in enumerate(rows, 1):
            tree.insert("", "end", iid=str(i),
                        values=(i, r["name"], f"{r['price']:.0f}"))
        save_menu_rows(rows)

    ttk.Label(frame, text="⚙️ TRANG ADMIN",
              font=('Segoe UI Semibold', 16),
              bootstyle="inverse-info", padding=10
              ).pack(fill=X, padx=10, pady=(10, 5))

    tbl_fr = ttk.Frame(frame)
    tbl_fr.pack(side=LEFT, fill=BOTH, expand=True, padx=(10, 0), pady=10)

    tree = ttk.Treeview(tbl_fr, columns=("id", "name", "price"), show="headings")
    for c, w in zip(("id", "name", "price"), (50, 220, 120)):
        tree.heading(c, text=c.upper())
        tree.column(c, width=w, anchor=CENTER if c == "id" else W)
    tree.pack(side=LEFT, fill=BOTH, expand=True)
    add_tree_scroll(tree, tbl_fr)
    refresh()

    panel = ttk.Frame(frame, padding=10)
    panel.pack(side=LEFT, fill=Y, padx=10, pady=10)

    def add_item():
        n = simpledialog.askstring("Tên món", "Nhập tên món:")
        if not n:
            return
        p = simpledialog.askfloat("Giá", f"Giá của {n} (VNĐ):", minvalue=0)
        if p is None:
            return
        rows.append({"name": n, "price": p})
        refresh()

    def edit_price():
        sel = tree.selection()
        if not sel:
            messagebox.showwarning("Chưa chọn", "Chọn món trước.")
            return
        idx = int(sel[0]) - 1
        cur = rows[idx]
        np = simpledialog.askfloat("Sửa giá",
                                   f"Giá mới cho {cur['name']}:",
                                   initialvalue=cur["price"], minvalue=0)
        if np is not None:
            cur["price"] = np
            refresh()

    def delete_item():
        sel = tree.selection()
        if not sel:
            messagebox.showwarning("Chưa chọn", "Chọn món trước.")
            return
        idx = int(sel[0]) - 1
        if messagebox.askyesno("Xóa", f"Xóa '{rows[idx]['name']}'?"):
            rows.pop(idx)
            refresh()

    for txt, cmd, style_btn in (
        ("➕ Thêm món", add_item, "success"),
        ("✏️  Sửa giá", edit_price, "info"),
        ("➖ Xóa món", delete_item, "danger"),
    ):
        ttk.Button(panel, text=txt, width=20,
                   bootstyle=style_btn, command=cmd
                   ).pack(pady=4)

    ttk.Separator(panel).pack(fill=X, pady=8)
    ttk.Label(panel, text="Table ID:", font=('Segoe UI', 12)).pack()
    ttk.Spinbox(panel, from_=1, to=100, textvariable=table_id_var,
                width=5, bootstyle="info").pack(pady=5)

# ─── HISTORY TAB ──────────────────────────────────────
def build_history():
    frame = history_frame
    for w in frame.winfo_children():
        w.destroy()

    # Bộ lọc
    filt = ttk.LabelFrame(frame, text="Bộ lọc", padding=10)
    filt.pack(fill=X, padx=10, pady=8)
    date_var, start_var, end_var = tk.StringVar(), tk.StringVar(), tk.StringVar()

    def _row(text, var, col, w=12):
        ttk.Label(filt, text=text).grid(row=0, column=col, sticky=W, padx=4)
        ttk.Entry(filt, textvariable=var, width=w).grid(row=0, column=col + 1, pady=2)

    _row("Ngày (dd-mm-YYYY):", date_var, 0)
    _row("Giờ bắt đầu (HH:MM):", start_var, 2, 8)
    _row("Giờ kết thúc (HH:MM):", end_var, 4, 8)
    ttk.Button(filt, text="🔍 Lọc", bootstyle="primary",
               command=lambda: load_rows()).grid(row=0, column=6, padx=(10, 0))

    tree_fr = ttk.Frame(frame)
    tree_fr.pack(fill=BOTH, expand=True, padx=10, pady=(0, 10))
    tree = ttk.Treeview(tree_fr, columns=("Bàn", "BĐ", "KT", "Ngày", "Tổng"),
                        show="headings")
    for h, w in zip(("Bàn", "BĐ", "KT", "Ngày", "Tổng"),
                    (50, 70, 70, 110, 120)):
        tree.heading(h, text=h)
        tree.column(h, width=w, anchor=CENTER if h in ("Bàn", "BĐ", "KT") else W)
    tree.pack(side=LEFT, fill=BOTH, expand=True)
    add_tree_scroll(tree, tree_fr)

    delta5 = dt.timedelta(minutes=5)

    def _parse(t):
        return dt.datetime.strptime(t, "%H:%M").time() if t else None

    def load_rows():
        tree.delete(*tree.get_children())
        try:
            udate = (dt.datetime.strptime(date_var.get(), "%d-%m-%Y").date()
                     if date_var.get() else None)
        except ValueError:
            messagebox.showerror("Sai định dạng", "Ngày phải dd-mm-YYYY")
            return
        try:
            us, ue = _parse(start_var.get()), _parse(end_var.get())
        except ValueError:
            messagebox.showerror("Sai định dạng", "Giờ phải HH:MM")
            return

        for p in list_bills(BILLS_DIR):
            meta = parse_bill_name(p.name)
            if not meta:
                continue
            date = dt.datetime.strptime(meta["date"], "%d-%m-%Y").date()
            st   = dt.datetime.strptime(meta["start"].replace('-', ':'), "%H:%M").time()
            et   = dt.datetime.strptime(
                     meta["end"].replace('-', ':'), "%H:%M").time() if meta["end"] else st
            if udate and date != udate:
                continue

            def within(t1, t2):
                return abs(
                    dt.datetime.combine(date, t1) -
                    dt.datetime.combine(date, t2)
                ) <= delta5

            if us and not within(st, us):
                continue
            if ue and not within(et, ue):
                continue

            tree.insert("", "end", iid=p.as_posix(),
                        values=(meta["table"], meta["start"], meta["end"] or "...",
                                meta["date"], f"{bill_total(p):,.0f}"))

    load_rows()

    def detail(evt):
        sel = tree.selection()
        if not sel:
            return
        path = Path(sel[0])
        win = tk.Toplevel(root)
        win.title(path.name)
        ttk.Label(win, text=path.name, font=('Segoe UI Semibold', 14)
                  ).pack(pady=(8, 4))

        tbl_fr = ttk.Frame(win)
        tbl_fr.pack(fill=BOTH, expand=True, padx=8, pady=8)
        tv = ttk.Treeview(tbl_fr, columns=("item", "qty", "unit", "line"),
                          show="headings")
        for cid, t, w in zip(
            ("item", "qty", "unit", "line"),
            ("Món", "SL", "Đơn giá", "Thành tiền"),
            (200, 50, 100, 110)
        ):
            tv.heading(cid, text=t)
            tv.column(cid, width=w, anchor=E if cid in ("unit", "line") else W)
        tv.pack(side=LEFT, fill=BOTH, expand=True)
        add_tree_scroll(tv, tbl_fr)

        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                tv.insert("", "end", values=(
                    r["item"], r["qty"],
                    f"{float(r['unit_price']):,.0f}",
                    f"{float(r['total_line']):,.0f}"
                ))
        ttk.Label(win, text=f"Tổng: {bill_total(path):,.0f} ₫",
                  font=('Segoe UI Semibold', 12)).pack(pady=(0, 8))

    tree.bind("<Double-1>", detail)

# ─── BIND & RUN ───────────────────────────────────────
nb.bind("<<NotebookTabChanged>>",
        lambda e: {0: build_order, 1: build_admin, 2: build_history}
        .get(nb.index("current"), lambda: None)())

build_order()
build_admin()
build_history()
root.mainloop()
