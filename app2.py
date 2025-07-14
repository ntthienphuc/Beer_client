# app.py  —  Beer-Manager (Raspberry Pi version)
# ─────────────────────────────────────────────────────────
# • Each can that blocks the optical sensor on BCM-17 triggers ONE photo
#   capture → TFLite classification → item is added to the bill.
# • Upload button still lets staff pick a photo manually.
# • Works on Raspberry Pi OS Bookworm (libgpiod); no deprecated /sys/class/gpio.

import os, csv, datetime as dt, threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

import cv2, numpy as np
from ttkbootstrap import Style, ttk
from ttkbootstrap.constants import *

from models.inference import TFLiteModel
from utils.csv_utils import (
    load_menu, init_bill, append_bill, save_menu_rows,
    parse_bill_name, bill_total, list_bills
)

# ───────────────────────────── GPIO ──────────────────────────────
try:
    from gpiozero import Button
    from gpiozero.pins.gpiochip import GPIOChipPinFactory   # libgpiod backend
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False

GPIO_PIN         = 17        # BCM number (physical pin-11)
GPIO_PULL_UP     = False     # True if sensor outputs LOW when blocked
GPIO_BOUNCE_TIME = 0.3       # seconds; one event per can

# ───────────────────── paths & model ─────────────────────────────
MENU_FILE  = "data/menu.csv"
MODEL_FILE = "best.tflite"
BILLS_DIR  = Path("data/bills")

# ──────────────────────── GUI root ───────────────────────────────
style = Style(theme="litera")
root  = style.master
root.title("🍺 Beer Manager")

# ─────────────────── load TFLite model ───────────────────────────
if not os.path.isfile(MODEL_FILE):
    raise FileNotFoundError(f"Model not found: {MODEL_FILE}")
model = TFLiteModel(MODEL_FILE)

table_id_var = tk.IntVar(master=root, value=1)

# ─────────────────── notebook skeleton ───────────────────────────
nb = ttk.Notebook(root)
order_frame   = ttk.Frame(nb)
admin_frame   = ttk.Frame(nb)
history_frame = ttk.Frame(nb)
nb.add(order_frame,  text="📝 ORDER")
nb.add(admin_frame,  text="🔧 ADMIN")
nb.add(history_frame, text="📜 HISTORY")
nb.pack(fill=BOTH, expand=True)

# ─────────────────── helper utilities ────────────────────────────
def read_image_unicode(path: str):
    """Read an image whose path may contain non-ASCII characters."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None

def add_tree_scroll(tree: ttk.Treeview, container: ttk.Frame, side=RIGHT):
    """Attach a vertical scrollbar to a Treeview."""
    vsb = ttk.Scrollbar(container, orient=VERTICAL, command=tree.yview)
    tree.configure(yscrollcommand=vsb.set)
    vsb.pack(side=side, fill=Y)

# ────────────────── global runtime state ─────────────────────────
order_state = {"active": False}
gpio_btn    = None
cap_lock    = threading.Lock()        # ensure one capture at a time

# ────────────────── silent capture + predict ─────────────────────
def _capture_and_predict(add_fn):
    """Take one frame, classify, call add_fn(name) silently."""
    if not cap_lock.acquire(blocking=False):
        return
    try:
        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        if not cap.isOpened():
            return
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return
        name = model.predict(frame)
        if name == "Món Chưa Xác Định":
            return
        root.after(0, lambda: add_fn(name))
    finally:
        if cap_lock.locked():
            cap_lock.release()

# ────────────────────── ORDER TAB ────────────────────────────────
def start_session():
    menu = load_menu()
    if not menu:
        messagebox.showwarning("Menu trống", "Bạn cần thêm món trước.", parent=root)
        return
    now = dt.datetime.now()
    tmp = init_bill(table_id_var.get(), "tmp")
    bill = tmp.with_name(f"bill_{table_id_var.get()}_{now:%H-%M}_{now:%d-%m-%Y}.csv")
    try:
        os.rename(tmp, bill)
    except FileExistsError:
        os.remove(tmp)
    order_state.update(
        active=True, menu=menu,
        qtys={n: 0 for n in menu},
        bill=bill, start=f"{now:%H-%M}"
    )
    build_order()

def finish_session():
    end   = dt.datetime.now().strftime("%H-%M")
    bill  = order_state["bill"]
    parts = bill.stem.split("_")
    if len(parts) == 4:
        os.rename(bill, bill.with_name("_".join(parts[:3] + [end] + parts[3:]) + bill.suffix))
    messagebox.showinfo("Đã kết thúc", f"Bill lưu tại:\n{bill}", parent=root)
    order_state.clear(); order_state["active"] = False
    build_order()

def build_order():
    """Redraw ORDER tab. Creates GPIO button on first entry."""
    global gpio_btn
    fr = order_frame
    for w in fr.winfo_children(): w.destroy()

    # No open session → show “Start” button only
    if not order_state["active"]:
        ttk.Button(fr, text="🚀  Bắt đầu", width=20, bootstyle="success",
                   padding=10, command=start_session).pack(pady=80)
        if gpio_btn:
            gpio_btn.close(); gpio_btn = None
        return

    # Active session UI
    menu, qtys, bill, start_str = (order_state[k] for k in ("menu","qtys","bill","start"))

    ttk.Label(fr, text=f"Bàn {table_id_var.get()} • Bắt đầu {start_str}",
              font=('Segoe UI Semibold', 18), bootstyle="inverse-primary",
              padding=10).pack(fill=X, padx=10, pady=(10, 0))

    style.configure("Treeview", rowheight=28, font=('Segoe UI', 11))
    style.configure("Treeview.Heading", font=('Segoe UI Semibold', 12))

    # ---------- left: item list ----------
    left = ttk.Frame(fr); left.pack(side=LEFT, fill=BOTH, expand=True, padx=(10, 0), pady=10)
    cols = ("name", "qty", "price")
    tree = ttk.Treeview(left, columns=cols, show="headings")
    for c, w in zip(cols, (200, 60, 100)):
        tree.heading(c, text={"name": "Món", "qty": "SL", "price": "Đơn giá"}[c])
        tree.column(c, width=w, anchor=CENTER if c == "qty" else W)
    for n, p in menu.items():
        tree.insert("", "end", iid=n, values=(n, qtys[n], f"{p:.0f}"))
    tree.pack(side=LEFT, fill=BOTH, expand=True)
    add_tree_scroll(tree, left)

    # ---------- right: control panel ----------
    ctrl = ttk.Frame(fr, padding=10); ctrl.pack(side=LEFT, fill=Y, padx=10, pady=10)
    total_var = tk.StringVar()
    def update_total():
        total_var.set(f"Tổng: {sum(qtys[n]*menu[n] for n in qtys):,.0f} ₫")
    update_total()
    ttk.Label(ctrl, textvariable=total_var, font=('Segoe UI Semibold', 14),
              bootstyle="primary", padding=(0, 0, 0, 10)).pack()

    # Quantity adjust list (scrollable)
    grp = ttk.LabelFrame(ctrl, text="Chỉnh SL", bootstyle="secondary")
    grp.pack(fill=X, pady=(0, 10))
    canvas = tk.Canvas(grp, height=300, highlightthickness=0)
    inner  = ttk.Frame(canvas)
    vsb    = ttk.Scrollbar(grp, orient=VERTICAL, command=canvas.yview)
    canvas.configure(yscrollcommand=vsb.set); vsb.pack(side=RIGHT, fill=Y)
    canvas.pack(side=LEFT, fill=BOTH, expand=True)
    canvas.create_window((0, 0), window=inner, anchor="nw")
    inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    def add_item(n):
        qtys[n] += 1
        tree.set(n, "qty", qtys[n])
        append_bill(bill, n, qtys[n], menu[n])
        update_total()
    def sub_item(n):
        if qtys[n] == 0:
            return
        qtys[n] -= 1
        tree.set(n, "qty", qtys[n])
        append_bill(bill, n, qtys[n], menu[n])
        update_total()

    for n in menu:
        rw = ttk.Frame(inner); rw.pack(fill=X, pady=2)
        ttk.Label(rw, text=n, width=20).pack(side=LEFT)
        ttk.Button(rw, text="+", bootstyle="success", width=3,
                   command=lambda i=n: add_item(i)).pack(side=LEFT, padx=4)
        ttk.Button(rw, text="-", bootstyle="danger",  width=3,
                   command=lambda i=n: sub_item(i)).pack(side=LEFT)

    # Manual upload button
    def upload():
        f = filedialog.askopenfilename(parent=root,
                                       filetypes=[("Images", "*.jpg *.jpeg *.png")])
        if not f:
            return
        img = read_image_unicode(f)
        if img is None:
            messagebox.showerror("Lỗi", f"Không đọc được ảnh:\n{f}", parent=root)
            return
        name = model.predict(img)
        add_item(name)
        messagebox.showinfo("Đã thêm", name, parent=root)
    ttk.Button(ctrl, text="📷 Upload Ảnh Bia", width=20,
               bootstyle="info", command=upload).pack(pady=(0, 10))

    ttk.Button(ctrl, text="✅ Kết thúc", width=20, bootstyle="primary",
               command=finish_session).pack()

    # ---------- GPIO setup (first time only) ----------
    if GPIO_AVAILABLE and gpio_btn is None:
        factory = GPIOChipPinFactory()              # force libgpiod backend
        gpio_btn = Button(GPIO_PIN,
                          pull_up=GPIO_PULL_UP,
                          bounce_time=GPIO_BOUNCE_TIME,
                          pin_factory=factory)
        def on_blocked():
            if cap_lock.locked():                   # camera busy
                return
            threading.Thread(target=_capture_and_predict,
                             args=(add_item,), daemon=True).start()
        gpio_btn.when_pressed = on_blocked

# ───────────────────────── ADMIN TAB ───────────────────────────────
def build_admin():
    f = admin_frame
    for w in f.winfo_children(): w.destroy()

    rows = []
    with open(MENU_FILE, newline="", encoding="utf-8") as csvf:
        for r in csv.DictReader(csvf):
            rows.append({"name": r["name"], "price": float(r["price"])})

    def refresh():
        tree.delete(*tree.get_children())
        for i, r in enumerate(rows, 1):
            tree.insert("", "end", iid=str(i),
                        values=(i, r["name"], f"{r['price']:.0f}"))
        save_menu_rows(rows)

    ttk.Label(f, text="⚙️ ADMIN PAGE", font=('Segoe UI Semibold', 16),
              bootstyle="inverse-info", padding=10).pack(fill=X, padx=10, pady=(10, 5))

    tbl = ttk.Frame(f); tbl.pack(side=LEFT, fill=BOTH, expand=True, padx=(10, 0), pady=10)
    tree = ttk.Treeview(tbl, columns=("id", "name", "price"), show="headings")
    for c, w in zip(("id", "name", "price"), (50, 220, 120)):
        tree.heading(c, text=c.upper())
        tree.column(c, width=w, anchor=CENTER if c == "id" else W)
    tree.pack(side=LEFT, fill=BOTH, expand=True)
    add_tree_scroll(tree, tbl)
    refresh()

    panel = ttk.Frame(f, padding=10); panel.pack(side=LEFT, fill=Y, padx=10, pady=10)
    def add_row():
        n = simpledialog.askstring("Tên món", "Nhập tên món:", parent=root)
        if not n:
            return
        p = simpledialog.askfloat("Giá", f"Giá của {n} (VNĐ):", minvalue=0, parent=root)
        if p is None:
            return
        rows.append({"name": n, "price": p}); refresh()
    def edit_price():
        sel = tree.selection()
        if not sel:
            messagebox.showwarning("Chưa chọn", "Chọn món trước.", parent=root); return
        idx = int(sel[0]) - 1
        cur = rows[idx]
        new = simpledialog.askfloat("Sửa giá",
                                    f"Giá mới cho {cur['name']}:",
                                    initialvalue=cur["price"], minvalue=0, parent=root)
        if new is not None:
            cur["price"] = new; refresh()
    def delete_row():
        sel = tree.selection()
        if not sel:
            messagebox.showwarning("Chưa chọn", "Chọn món trước.", parent=root); return
        idx = int(sel[0]) - 1
        if messagebox.askyesno("Xóa", f"Xóa '{rows[idx]['name']}'?", parent=root):
            rows.pop(idx); refresh()
    for txt, fn, sty in (("➕ Thêm món", add_row, "success"),
                         ("✏️  Sửa giá", edit_price, "info"),
                         ("➖ Xóa món", delete_row, "danger")):
        ttk.Button(panel, text=txt, width=20, bootstyle=sty,
                   command=fn).pack(pady=4)
    ttk.Separator(panel).pack(fill=X, pady=8)
    ttk.Label(panel, text="Table ID:", font=('Segoe UI', 12)).pack()
    ttk.Spinbox(panel, from_=1, to=100, textvariable=table_id_var,
                width=5, bootstyle="info").pack(pady=5)

# ─────────────────────── HISTORY TAB ───────────────────────────────
def build_history():
    f = history_frame
    for w in f.winfo_children(): w.destroy()

    # filter bar
    filt = ttk.LabelFrame(f, text="Filters", padding=10)
    filt.pack(fill=X, padx=10, pady=8)
    date_var, start_var, end_var = tk.StringVar(), tk.StringVar(), tk.StringVar()
    def _row(label, var, col, width=12):
        ttk.Label(filt, text=label).grid(row=0, column=col, sticky=W, padx=4)
        ttk.Entry(filt, textvariable=var, width=width).grid(row=0, column=col+1, pady=2)
    _row("Date (dd-mm-YYYY):", date_var, 0)
    _row("Start (HH:MM):",      start_var, 2, 8)
    _row("End (HH:MM):",        end_var,   4, 8)
    ttk.Button(filt, text="🔍 Search", bootstyle="primary",
               command=lambda: load_rows()).grid(row=0, column=6, padx=(10, 0))

    # result table
    lst = ttk.Frame(f); lst.pack(fill=BOTH, expand=True, padx=10, pady=(0, 10))
    tree = ttk.Treeview(lst, columns=("Table", "Start", "End", "Date", "Total"),
                        show="headings")
    for h, w in zip(("Table", "Start", "End", "Date", "Total"),
                    (50, 70, 70, 110, 120)):
        tree.heading(h, text=h)
        tree.column(h, width=w, anchor=CENTER if h in ("Table", "Start", "End") else W)
    tree.pack(side=LEFT, fill=BOTH, expand=True)
    add_tree_scroll(tree, lst)

    delta5 = dt.timedelta(minutes=5)
    def _parse(t): return dt.datetime.strptime(t, "%H:%M").time() if t else None

    def load_rows():
        tree.delete(*tree.get_children())
        try:
            udate = dt.datetime.strptime(date_var.get(), "%d-%m-%Y").date() if date_var.get() else None
        except ValueError:
            messagebox.showerror("Định dạng sai", "Ngày phải dd-mm-YYYY", parent=root); return
        try:
            us, ue = _parse(start_var.get()), _parse(end_var.get())
        except ValueError:
            messagebox.showerror("Định dạng sai", "Giờ phải HH:MM", parent=root); return

        for p in list_bills(BILLS_DIR):
            meta = parse_bill_name(p.name)
            if not meta:
                continue
            d  = dt.datetime.strptime(meta["date"], "%d-%m-%Y").date()
            st = dt.datetime.strptime(meta["start"].replace('-', ':'), "%H:%M").time()
            et = dt.datetime.strptime(meta["end"].replace('-', ':'), "%H:%M").time() if meta["end"] else st
            if udate and d != udate:
                continue
            def within(t1, t2):
                return abs(dt.datetime.combine(d, t1) - dt.datetime.combine(d, t2)) <= delta5
            if us and not within(st, us):
                continue
            if ue and not within(et, ue):
                continue
            tree.insert("", "end", iid=p.as_posix(),
                        values=(meta["table"], meta["start"], meta["end"] or "...",
                                meta["date"], f"{bill_total(p):,.0f}"))
    load_rows()

    def detail(_):
        sel = tree.selection()
        if not sel:
            return
        path = Path(sel[0])
        win = tk.Toplevel(root); win.title(path.name)
        ttk.Label(win, text=path.name, font=('Segoe UI Semibold', 14)).pack(pady=(8, 4))

        tbl = ttk.Frame(win); tbl.pack(fill=BOTH, expand=True, padx=8, pady=8)
        tv = ttk.Treeview(tbl, columns=("item", "qty", "unit", "line"), show="headings")
        for cid, lbl, w in zip(("item", "qty", "unit", "line"),
                               ("Item", "Qty", "Unit Price", "Line Total"),
                               (200, 50, 100, 110)):
            tv.heading(cid, text=lbl)
            tv.column(cid, width=w, anchor=E if cid in ("unit", "line") else W)
        tv.pack(side=LEFT, fill=BOTH, expand=True)
        add_tree_scroll(tv, tbl)

        with open(path, newline="", encoding="utf-8") as fcsv:
            for r in csv.DictReader(fcsv):
                tv.insert("", "end", values=(
                    r["item"], r["qty"],
                    f"{float(r['unit_price']):,.0f}",
                    f"{float(r['total_line']):,.0f}"
                ))
        ttk.Label(win, text=f"Total: {bill_total(path):,.0f} ₫",
                  font=('Segoe UI Semibold', 12)).pack(pady=(0, 8))
    tree.bind("<Double-1>", detail)

# ─────────────────── tab change binding ────────────────────────────
nb.bind("<<NotebookTabChanged>>",
        lambda e: {0: build_order, 1: build_admin, 2: build_history}
        .get(nb.index("current"), lambda: None)())

# ─────────────────── initial draw & mainloop ───────────────────────
build_order(); build_admin(); build_history()
root.mainloop()
