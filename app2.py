#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app.py — Beer-Manager (Raspberry Pi Edition)  🏝

Complete, one-file GUI application that

  • Lists menu items, edits prices (ADMIN tab)
  • Opens a session and builds a bill (ORDER tab)
    – A can blocking the IR sensor on GPIO-17 triggers a photo
      → TFLite classification → quantity increment
    – Manual “Upload Beer Image” button does the same from a file
  • Browses previous CSV bills (HISTORY tab)

This version adds **detailed logging** around the sensor / capture flow:

  – INFO:  beam blocked, capture start, classified name, busy-camera skips
  – DEBUG: chosen capture backend (OpenCV vs libcamera), etc.

Change log level at the top of the file:

    logging.basicConfig(level=logging.INFO, …)     # INFO and above
    logging.basicConfig(level=logging.DEBUG, …)    # show DEBUG too
"""

from __future__ import annotations

# ────────────────────── Standard Library ────────────────────────
import csv
import datetime as dt
import logging
import os
import subprocess
import tempfile
import threading
import uuid
from pathlib import Path

# ───────────────────────── 3rd-Party ────────────────────────────
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from ttkbootstrap import Style, ttk
from ttkbootstrap.constants import *

from models.inference import TFLiteModel, UNKNOWN_NAME
from utils.csv_utils import (
    append_bill,
    bill_total,
    init_bill,
    list_bills,
    load_menu,
    parse_bill_name,
    save_menu_rows,
)

# GPIO (present only on Pi)
try:
    from gpiozero import Button
    from gpiozero.pins.gpiochip import GPIOChipPinFactory

    GPIO_AVAILABLE = True
except ImportError:  # developing on PC / Mac
    GPIO_AVAILABLE = False

# ─────────────────────── Configuration ──────────────────────────
GPIO_PIN = 17                   # BCM pin number
GPIO_BOUNCE_TIME = 0.3          # seconds; one event per can

MENU_FILE = "data/menu.csv"
MODEL_FILE = "best.tflite"
BILLS_DIR = Path("data/bills")

logging.basicConfig(
    level=logging.INFO,         # switch to DEBUG for more detail
    format="%(asctime)s  %(levelname)8s — %(message)s",
)

# ────────────────────── Load TFLite model ───────────────────────
if not os.path.isfile(MODEL_FILE):
    raise FileNotFoundError(f"Model not found: {MODEL_FILE}")
model = TFLiteModel(MODEL_FILE)
logging.info("TFLite model loaded from %s", MODEL_FILE)

# ────────────────────────── Tk root ─────────────────────────────
style = Style(theme="litera")
root = style.master
root.title("🍺 Beer Manager")

table_id_var = ttk.IntVar(master=root, value=1)

# ──────────────────────── Notebook ──────────────────────────────
nb = ttk.Notebook(root)
order_frame   = ttk.Frame(nb)
admin_frame   = ttk.Frame(nb)
history_frame = ttk.Frame(nb)
nb.add(order_frame,  text="📝 ORDER")
nb.add(admin_frame,  text="🔧 ADMIN")
nb.add(history_frame, text="📜 HISTORY")
nb.pack(fill=BOTH, expand=True)

# ───────────────────── Helper Utilities ─────────────────────────
cap_lock = threading.Lock()  # ensure single concurrent capture


def read_image_unicode(path: str) -> np.ndarray | None:
    """Open image even if the path contains non-ASCII characters."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception as exc:
        logging.error("Failed to read %s: %s", path, exc)
        return None


def add_tree_scroll(tree: ttk.Treeview, container: ttk.Frame, side=RIGHT) -> None:
    vsb = ttk.Scrollbar(container, orient=VERTICAL, command=tree.yview)
    tree.configure(yscrollcommand=vsb.set)
    vsb.pack(side=side, fill=Y)


# ───────────────── Camera Capture (with fallback) ───────────────
def capture_frame() -> np.ndarray | None:
    """
    Returns one BGR frame or None.
    Tries OpenCV/V4L2 first; falls back to libcamera-still.
    """
    # Attempt 1: OpenCV (works for USB webcams and legacy driver)
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if cap.isOpened():
        ok, frame = cap.read()
        cap.release()
        if ok:
            logging.debug("Frame captured via OpenCV/V4L2")
            return frame

    # Attempt 2: libcamera-still (works on Pi Camera with Bookworm)
    tmp = Path(tempfile.gettempdir()) / f"{uuid.uuid4()}.jpg"
    try:
        subprocess.run(
            [
                "libcamera-still",
                "-o", str(tmp),
                "--width", "640", "--height", "480",
                "-t", "200", "--nopreview", "--flush", "--autowb",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        img = cv2.imread(str(tmp))
        if img is not None:
            logging.debug("Frame captured via libcamera-still")
        return img
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        logging.exception("Capture failed: %s", exc)
        return None
    finally:
        tmp.unlink(missing_ok=True)


# ─────────────── One-shot Capture + Classification ──────────────
def _capture_and_predict(add_fn):
    """
    Background thread: capture → classify → schedule GUI update.
    """
    if not cap_lock.acquire(blocking=False):
        logging.info("Sensor triggered but camera is busy – skipped")
        return
    try:
        logging.info("Starting capture…")
        frame = capture_frame()
        if frame is None:
            logging.warning("No frame captured – skipped")
            return

        name = model.predict(frame)
        if name == UNKNOWN_NAME:
            logging.info("Low-confidence result – ignored")
            return

        logging.info("✅ Classified: %s", name)
        root.after(0, lambda: add_fn(name))
    finally:
        if cap_lock.locked():
            cap_lock.release()


# ───────────────────────── ORDER TAB ────────────────────────────
order_state: dict[str, object] = {"active": False}
gpio_btn: Button | None = None  # initialised on first tab entry


def start_session() -> None:
    """Begin a new bill for the selected table."""
    menu = load_menu()
    if not menu:
        messagebox.showwarning("Empty menu", "Add items in ADMIN first.", parent=root)
        return

    now = dt.datetime.now()
    tmp = init_bill(table_id_var.get(), "tmp")
    bill = tmp.with_name(f"bill_{table_id_var.get()}_{now:%H-%M}_{now:%d-%m-%Y}.csv")
    try:
        os.rename(tmp, bill)
    except FileExistsError:
        os.remove(tmp)

    order_state.update(
        active=True,
        menu=menu,
        qtys={n: 0 for n in menu},
        bill=bill,
        start=f"{now:%H-%M}",
    )
    build_order()


def finish_session() -> None:
    """Close the active bill and reset the ORDER tab."""
    end = dt.datetime.now().strftime("%H-%M")
    bill: Path = order_state["bill"]  # type: ignore
    parts = bill.stem.split("_")
    if len(parts) == 4:  # only add end-time once
        bill.rename(bill.with_name("_".join(parts[:3] + [end] + parts[3:]) + bill.suffix))

    messagebox.showinfo("Session finished", f"Bill saved:\n{bill}", parent=root)
    order_state.clear()
    order_state["active"] = False
    build_order()


def build_order() -> None:
    """Redraw the ORDER tab."""
    global gpio_btn
    fr = order_frame
    for w in fr.winfo_children():
        w.destroy()

    # ——— No active session ————————————————————————————
    if not order_state["active"]:
        ttk.Button(
            fr, text="🚀  Start Session", width=20, bootstyle="success",
            padding=10, command=start_session,
        ).pack(pady=80)
        if gpio_btn:
            gpio_btn.close()
            gpio_btn = None
        return

    # ——— Active session ————————————————————————————————
    menu: dict[str, float] = order_state["menu"]  # type: ignore
    qtys: dict[str, int] = order_state["qtys"]  # type: ignore
    bill: Path = order_state["bill"]            # type: ignore
    start_str: str = order_state["start"]       # type: ignore

    ttk.Label(
        fr, text=f"Table {table_id_var.get()} • Started {start_str}",
        font=("Segoe UI Semibold", 18),
        bootstyle="inverse-primary",
        padding=10,
    ).pack(fill=X, padx=10, pady=(10, 0))

    # Configure table fonts
    style.configure("Treeview", rowheight=28, font=("Segoe UI", 11))
    style.configure("Treeview.Heading", font=("Segoe UI Semibold", 12))

    # Left pane — menu item list
    left = ttk.Frame(fr)
    left.pack(side=LEFT, fill=BOTH, expand=True, padx=(10, 0), pady=10)
    cols = ("name", "qty", "price")
    tree = ttk.Treeview(left, columns=cols, show="headings")
    for cid, width in zip(cols, (200, 60, 100)):
        tree.heading(cid, text={"name": "Item", "qty": "Qty", "price": "Unit"}[cid])
        tree.column(cid, width=width, anchor=CENTER if cid == "qty" else W)
    for n, p in menu.items():
        tree.insert("", "end", iid=n, values=(n, qtys[n], f"{p:.0f}"))
    tree.pack(side=LEFT, fill=BOTH, expand=True)
    add_tree_scroll(tree, left)

    # Right pane — controls
    ctrl = ttk.Frame(fr, padding=10)
    ctrl.pack(side=LEFT, fill=Y, padx=10, pady=10)

    total_var = ttk.StringVar()

    def update_total() -> None:
        total_var.set(f"Total: {sum(qtys[n] * menu[n] for n in qtys):,.0f} ₫")

    update_total()
    ttk.Label(
        ctrl, textvariable=total_var, font=("Segoe UI Semibold", 14),
        bootstyle="primary", padding=(0, 0, 0, 10)
    ).pack()

    # Quantity adjust list
    grp = ttk.LabelFrame(ctrl, text="Adjust Qty", bootstyle="secondary")
    grp.pack(fill=X, pady=(0, 10))
    canvas = tk.Canvas(grp, height=300, highlightthickness=0)
    inner  = ttk.Frame(canvas)
    vsb    = ttk.Scrollbar(grp, orient=VERTICAL, command=canvas.yview)
    canvas.configure(yscrollcommand=vsb.set)
    vsb.pack(side=RIGHT, fill=Y)
    canvas.pack(side=LEFT, fill=BOTH, expand=True)
    canvas.create_window((0, 0), window=inner, anchor="nw")
    inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

    def add_item(n: str) -> None:
        qtys[n] += 1
        tree.set(n, "qty", qtys[n])
        append_bill(bill, n, qtys[n], menu[n])
        update_total()

    def sub_item(n: str) -> None:
        if qtys[n] == 0:
            return
        qtys[n] -= 1
        tree.set(n, "qty", qtys[n])
        append_bill(bill, n, qtys[n], menu[n])
        update_total()

    for n in menu:
        row = ttk.Frame(inner)
        row.pack(fill=X, pady=2)
        ttk.Label(row, text=n, width=20).pack(side=LEFT)
        ttk.Button(row, text="+", bootstyle="success", width=3,
                   command=lambda i=n: add_item(i)).pack(side=LEFT, padx=4)
        ttk.Button(row, text="-", bootstyle="danger",  width=3,
                   command=lambda i=n: sub_item(i)).pack(side=LEFT)

    # Manual upload button
    def upload_image() -> None:
        fpath = filedialog.askopenfilename(
            parent=root, filetypes=[("Images", "*.jpg *.jpeg *.png")]
        )
        if not fpath:
            return
        img = read_image_unicode(fpath)
        if img is None:
            messagebox.showerror("Error", f"Cannot read image:\n{fpath}", parent=root)
            return
        name = model.predict(img)
        add_item(name)
        messagebox.showinfo("Added", name, parent=root)

    ttk.Button(ctrl, text="📷 Upload Beer Image", width=20,
               bootstyle="info", command=upload_image).pack(pady=(0, 10))

    ttk.Button(ctrl, text="✅ Finish Session", width=20,
               bootstyle="primary", command=finish_session).pack()

    # GPIO setup — run once
    if GPIO_AVAILABLE and gpio_btn is None:
        factory = GPIOChipPinFactory()
        gpio_btn = Button(
            GPIO_PIN,
            pull_up=True,            # sensor keeps line HIGH; LOW = blocked
            bounce_time=GPIO_BOUNCE_TIME,
            pin_factory=factory,
            active_state=False,      # pressed when line is LOW
        )
        logging.info("GPIO pin %d initialised (active LOW)", GPIO_PIN)

        def on_blocked():
            logging.info("📟  Beam blocked — sensor triggered")
            threading.Thread(
                target=_capture_and_predict,
                args=(add_item,),
                daemon=True
            ).start()

        gpio_btn.when_pressed = on_blocked


# ───────────────────────── ADMIN TAB ────────────────────────────
def build_admin() -> None:
    f = admin_frame
    for w in f.winfo_children():
        w.destroy()

    rows: list[dict[str, float | str]] = []
    with open(MENU_FILE, newline="", encoding="utf-8") as csvf:
        for r in csv.DictReader(csvf):
            rows.append({"name": r["name"], "price": float(r["price"])})

    def refresh() -> None:
        tree.delete(*tree.get_children())
        for i, r in enumerate(rows, 1):
            tree.insert("", "end", iid=str(i),
                        values=(i, r["name"], f"{r['price']:.0f}"))
        save_menu_rows(rows)

    ttk.Label(
        f, text="⚙️ ADMIN PAGE", font=("Segoe UI Semibold", 16),
        bootstyle="inverse-info", padding=10
    ).pack(fill=X, padx=10, pady=(10, 5))

    tbl = ttk.Frame(f)
    tbl.pack(side=LEFT, fill=BOTH, expand=True, padx=(10, 0), pady=10)
    tree = ttk.Treeview(tbl, columns=("id", "name", "price"), show="headings")
    for cid, width in zip(("id", "name", "price"), (50, 220, 120)):
        tree.heading(cid, text=cid.upper())
        tree.column(cid, width=width, anchor=CENTER if cid == "id" else W)
    tree.pack(side=LEFT, fill=BOTH, expand=True)
    add_tree_scroll(tree, tbl)
    refresh()

    panel = ttk.Frame(f, padding=10)
    panel.pack(side=LEFT, fill=Y, padx=10, pady=10)

    def add_row() -> None:
        n = simpledialog.askstring("Item name", "Enter new item name:", parent=root)
        if not n:
            return
        p = simpledialog.askfloat("Price", f"Unit price for {n} (₫):",
                                  minvalue=0, parent=root)
        if p is None:
            return
        rows.append({"name": n, "price": p})
        refresh()

    def edit_price() -> None:
        sel = tree.selection()
        if not sel:
            messagebox.showwarning("No selection", "Select an item first.", parent=root)
            return
        idx = int(sel[0]) - 1
        cur = rows[idx]
        new = simpledialog.askfloat(
            "Edit price",
            f"New price for {cur['name']}:",
            initialvalue=cur["price"],
            minvalue=0,
            parent=root,
        )
        if new is not None:
            cur["price"] = new
            refresh()

    def delete_row() -> None:
        sel = tree.selection()
        if not sel:
            messagebox.showwarning("No selection", "Select an item first.", parent=root)
            return
        idx = int(sel[0]) - 1
        if messagebox.askyesno("Delete", f"Delete '{rows[idx]['name']}'?", parent=root):
            rows.pop(idx)
            refresh()

    for txt, fn, sty in (
        ("➕ Add item", add_row, "success"),
        ("✏️  Edit price", edit_price, "info"),
        ("➖ Delete item", delete_row, "danger"),
    ):
        ttk.Button(panel, text=txt, width=20, bootstyle=sty,
                   command=fn).pack(pady=4)

    ttk.Separator(panel).pack(fill=X, pady=8)
    ttk.Label(panel, text="Table ID:", font=("Segoe UI", 12)).pack()
    ttk.Spinbox(panel, from_=1, to=100, textvariable=table_id_var,
                width=5, bootstyle="info").pack(pady=5)


# ─────────────────────── HISTORY TAB ────────────────────────────
def build_history() -> None:
    f = history_frame
    for w in f.winfo_children():
        w.destroy()

    filt = ttk.LabelFrame(f, text="Filters", padding=10)
    filt.pack(fill=X, padx=10, pady=8)
    date_var, start_var, end_var = ttk.StringVar(), ttk.StringVar(), ttk.StringVar()

    def _row(label: str, var: ttk.StringVar, col: int, width: int = 12):
        ttk.Label(filt, text=label).grid(row=0, column=col, sticky=W, padx=4)
        ttk.Entry(filt, textvariable=var, width=width).grid(row=0, column=col + 1, pady=2)

    _row("Date (dd-mm-YYYY):", date_var, 0)
    _row("Start (HH:MM):",      start_var, 2, 8)
    _row("End (HH:MM):",        end_var,   4, 8)
    ttk.Button(filt, text="🔍 Search", bootstyle="primary",
               command=lambda: load_rows()).grid(row=0, column=6, padx=(10, 0))

    lst = ttk.Frame(f)
    lst.pack(fill=BOTH, expand=True, padx=10, pady=(0, 10))
    tree = ttk.Treeview(lst, columns=("Table", "Start", "End", "Date", "Total"),
                        show="headings")
    for h, w in zip(("Table", "Start", "End", "Date", "Total"),
                    (50, 70, 70, 110, 120)):
        tree.heading(h, text=h)
        tree.column(h, width=w, anchor=CENTER if h in ("Table", "Start", "End") else W)
    tree.pack(side=LEFT, fill=BOTH, expand=True)
    add_tree_scroll(tree, lst)

    delta5 = dt.timedelta(minutes=5)

    def _parse(t: str) -> dt.time | None:
        return dt.datetime.strptime(t, "%H:%M").time() if t else None

    def load_rows() -> None:
        tree.delete(*tree.get_children())
        try:
            udate = dt.datetime.strptime(date_var.get(), "%d-%m-%Y").date() if date_var.get() else None
        except ValueError:
            messagebox.showerror("Invalid format", "Date must be dd-mm-YYYY", parent=root)
            return
        try:
            us, ue = _parse(start_var.get()), _parse(end_var.get())
        except ValueError:
            messagebox.showerror("Invalid format", "Time must be HH:MM", parent=root)
            return

        for path in list_bills(BILLS_DIR):
            meta = parse_bill_name(path.name)
            if not meta:
                continue
            d  = dt.datetime.strptime(meta["date"], "%d-%m-%Y").date()
            st = dt.datetime.strptime(meta["start"].replace('-', ':'), "%H:%M").time()
            et = dt.datetime.strptime(meta["end"].replace('-', ':'), "%H:%M").time() if meta["end"] else st

            def within(t1: dt.time, t2: dt.time) -> bool:
                return abs(dt.datetime.combine(d, t1) - dt.datetime.combine(d, t2)) <= delta5

            if udate and d != udate:
                continue
            if us and not within(st, us):
                continue
            if ue and not within(et, ue):
                continue

            tree.insert("", "end", iid=str(path),
                        values=(meta["table"], meta["start"], meta["end"] or "...",
                                meta["date"], f"{bill_total(path):,.0f}"))

    load_rows()

    def detail(_: object) -> None:
        sel = tree.selection()
        if not sel:
            return
        path = Path(sel[0])
        win = tk.Toplevel(root)
        win.title(path.name)
        ttk.Label(win, text=path.name,
                  font=("Segoe UI Semibold", 14)).pack(pady=(8, 4))

        tbl = ttk.Frame(win)
        tbl.pack(fill=BOTH, expand=True, padx=8, pady=8)
        tv = ttk.Treeview(tbl, columns=("item", "qty", "unit", "line"),
                          show="headings")
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
                  font=("Segoe UI Semibold", 12)).pack(pady=(0, 8))

    tree.bind("<Double-1>", detail)


# ─────────────────── Tab-change Lazy Redraw ──────────────────────
nb.bind("<<NotebookTabChanged>>",
        lambda e: {0: build_order, 1: build_admin, 2: build_history}
        .get(nb.index("current"), lambda: None)())

# ───────────────────────── Run UI ───────────────────────────────
build_order()
build_admin()
build_history()
root.mainloop()
