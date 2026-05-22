import tkinter as tk
import tkinter.ttk as ttk
import tkinter.messagebox
import tkinter.simpledialog
import tkinter.filedialog
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sps
import scipy.interpolate as spi
import wordcloud as wc
import nltk
import pypdf
import docx
import statistics
import random
import collections
import os
import re
import chardet
import copy

try:
    import openpyxl
    XL_OK = True
except ImportError:
    XL_OK = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_OK = True
except ImportError:
    VADER_OK = False

try:
    from dostoevsky.tokenization import RegexTokenizer
    from dostoevsky.models import FastTextSocialNetworkModel
    DOST_OK = True
except ImportError:
    DOST_OK = False

for res in ['tokenizers/punkt', 'corpora/stopwords']:
    try:
        nltk.data.find(res)
    except LookupError:
        nltk.download(res.split('/')[-1])

class App:
    def __init__(self, master):
        self.master = master
        master.title("Обработка результатов эксперимента")
        master.geometry("1100x750")
        master.protocol("WM_DELETE_WINDOW", self._close)

        top = ttk.Frame(master)
        top.pack(pady=5, fill='x')

        self.btn_num = ttk.Button(top, text="📊 Числовые данные", command=self._show_num)
        self.btn_num.pack(side=tk.LEFT, padx=10)

        self.btn_txt = ttk.Button(top, text="📄 Текстовые данные", command=self._show_txt)
        self.btn_txt.pack(side=tk.LEFT, padx=10)

        self.btn_help = ttk.Button(top, text="❓ Справка", command=self._help)
        self.btn_help.pack(side=tk.RIGHT, padx=10)

        self.content = ttk.Frame(master)
        self.content.pack(fill='both', expand=True, padx=10, pady=10)

        self.num_frame = ttk.Frame(self.content)
        self.txt_frame = ttk.Frame(self.content)

        self.num_tab = NumTab(self.num_frame, self)
        self.txt_tab = TxtTab(self.txt_frame, self)

        self._show_num()

    def _show_num(self):
        self.txt_frame.pack_forget()
        self.num_frame.pack(fill='both', expand=True)
        self.btn_num.config(state='disabled')
        self.btn_txt.config(state='normal')

    def _show_txt(self):
        self.num_frame.pack_forget()
        self.txt_frame.pack(fill='both', expand=True)
        self.btn_txt.config(state='disabled')
        self.btn_num.config(state='normal')

    def _help(self):
        text = """🔹 КАК РАБОТАТЬ С ПРОГРАММОЙ 🔹

1. ЧИСЛОВЫЕ ДАННЫЕ:
   • Введите число и нажмите Enter – добавится в список.
   • Двойной клик по числу – редактирование.
   • Кнопки: очистить, гистограмма, график, сортировка, фильтр, сброс фильтра.
   • Поиск, генерация случайных чисел, статистика.
   • Загрузка двумерных (X,Y) – «Открыть (2 столбца)».
   • Сохранение в TXT/CSV/Excel, открытие файлов (TXT, CSV, Excel, DOCX).
   • Математические методы – меню «📐 Мат. методы»:
        регрессия, корреляция, интерполяция, t-тест, выбросы,
        нормализация, стандартизация, доверительный интервал.
   • Отмена/повтор: Ctrl+Z, Ctrl+Y или кнопки ↶/↷.

2. ТЕКСТОВЫЕ ДАННЫЕ:
   • Загрузить TXT, CSV, PDF, DOCX или ввести вручную.
   • Выбрать язык (русский, английский, немецкий, французский, испанский).
   • Кнопки: статистика, поиск, облако слов, тональность, удалить стоп-слова, стемминг.
   • Отмена/повтор: Ctrl+Z, Ctrl+Y (встроены в текстовое поле).

3. ОБЩЕЕ:
   • Переключение между режимами сохраняет данные.
   • Математические методы описаны в меню.

Горячие клавиши:
   Enter – добавить число (числовой режим)
   Ctrl+Z – отменить
   Ctrl+Y – повторить
"""
        tk.messagebox.showinfo("Справка", text)

    def _close(self):
        try:
            plt.close('all')
        except:
            pass
        self.master.destroy()


class NumTab:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.data = []
        self.fdata = None
        self.f_active = False
        self.bivar = False
        self.x = []
        self.y = []

        self.hist = []
        self.hist_pos = -1
        self._save_hist()

        self._build()
        self.refresh_info()

    def _save_hist(self):
        self.hist = self.hist[:self.hist_pos+1]
        self.hist.append(copy.deepcopy(self.data))
        self.hist_pos += 1
        if len(self.hist) > 100:
            self.hist.pop(0)
            self.hist_pos -= 1

    def undo(self):
        if self.hist_pos > 0:
            self.hist_pos -= 1
            self.data = copy.deepcopy(self.hist[self.hist_pos])
            if self.f_active:
                self.unfilter()
            self.show()
            self.refresh_info()
            self.msg_lbl.config(text="Отменено", foreground="blue")

    def redo(self):
        if self.hist_pos < len(self.hist)-1:
            self.hist_pos += 1
            self.data = copy.deepcopy(self.hist[self.hist_pos])
            if self.f_active:
                self.unfilter()
            self.show()
            self.refresh_info()
            self.msg_lbl.config(text="Повторено", foreground="blue")

    def _build(self):
        btn_cont = ttk.Frame(self.parent)
        btn_cont.pack(pady=10, fill='x')

        def btn(parent, text, cmd):
            b = ttk.Button(parent, text=text, command=cmd)
            b.pack(side=tk.LEFT, padx=2, pady=2)
            return b

        row1 = ttk.Frame(btn_cont)
        row1.pack(fill='x', pady=2)
        for (text, cmd) in [
            ("🗑️ Очистить", self.clear),
            ("📊 Гистограмма", self.hist_plot),
            ("📈 График", self.graph),
            ("🔽 Сортировать", self.sort),
            ("🔍 Фильтр", self.filter_),
            ("🔄 Сброс", self.unfilter),
        ]:
            btn(row1, text, cmd)

        ttk.Separator(btn_cont, orient='horizontal').pack(fill='x', pady=5)

        row2 = ttk.Frame(btn_cont)
        row2.pack(fill='x', pady=2)
        for (text, cmd) in [
            ("🔎 Поиск", self.search),
            ("🎲 Генерировать", self.gen_rand),
            ("📋 Статистика", self.stats),
            ("📂 Открыть (2 столбца)", self.load_xy),
            ("💾 Сохранить", self.save),
            ("📁 Открыть файлы", self.load_many),
        ]:
            btn(row2, text, cmd)

        ttk.Separator(btn_cont, orient='horizontal').pack(fill='x', pady=5)

        row3 = ttk.Frame(btn_cont)
        row3.pack(fill='x', pady=2)

        self.btn_math = ttk.Button(row3, text="📐 Мат. методы", command=self._math_menu)
        self.btn_math.pack(side=tk.LEFT, padx=2, pady=2)

        self.menu = tk.Menu(self.parent, tearoff=0)
        reg_menu = tk.Menu(self.menu, tearoff=0)
        reg_menu.add_command(label="Линейная регрессия", command=self.regress)
        reg_menu.add_command(label="Полином 4-й степени", command=self.poly_reg)
        self.menu.add_cascade(label="Регрессия", menu=reg_menu)

        self.menu.add_command(label="Корреляция Пирсона", command=self.correl)

        interp_menu = tk.Menu(self.menu, tearoff=0)
        interp_menu.add_command(label="Лагранж", command=self.lagrange)
        interp_menu.add_command(label="Ньютон", command=self.newton)
        interp_menu.add_command(label="Канонический полином", command=self.canon)
        interp_menu.add_command(label="Линейный сплайн", command=self.lin_spline)
        interp_menu.add_command(label="Кубический сплайн", command=self.cub_spline)
        self.menu.add_cascade(label="Интерполяция", menu=interp_menu)

        self.menu.add_separator()
        self.menu.add_command(label="t-тест", command=self.ttest)
        self.menu.add_command(label="Удалить выбросы", command=self.outliers)
        self.menu.add_command(label="Нормализация [0,1]", command=self.norm)
        self.menu.add_command(label="Стандартизация (z)", command=self.stand)
        self.menu.add_command(label="Доверительный интервал", command=self.conf_int)

        for (text, cmd) in [
            ("📈 Корреляция", self.correl),
            ("📉 Регрессия", self.regress),
            ("🔬 t-тест", self.ttest),
            ("🧹 Выбросы", self.outliers),
        ]:
            btn(row3, text, cmd)

        btn(row3, "↶", self.undo)
        btn(row3, "↷", self.redo)

        main = ttk.Frame(self.parent)
        main.pack(fill='both', expand=True, padx=10, pady=10)

        left = ttk.Frame(main)
        left.pack(side=tk.LEFT, fill='both', expand=True)

        right = ttk.Frame(main, width=320)
        right.pack(side=tk.RIGHT, fill='y', padx=(10,0))
        right.pack_propagate(False)

        inp = ttk.Frame(left)
        inp.pack(fill='x', padx=5, pady=5)
        self.entry = ttk.Entry(inp, width=30)
        self.entry.pack(side=tk.LEFT, padx=5)
        self.entry.bind("<Return>", lambda e: self.add())

        self.msg_lbl = ttk.Label(left, text="", foreground="green")
        self.msg_lbl.pack(pady=5)

        ttk.Label(left, text="Список (двойной клик – правка):").pack(anchor='w', padx=5)
        self.text_box = tk.Text(left, height=18, width=70, state='disabled')
        self.text_box.pack(fill='both', expand=True, padx=5, pady=5)
        self.text_box.bind("<Double-Button-1>", lambda e: self.edit())

        self.stat_box = ttk.LabelFrame(right, text="Статистика")
        self.stat_box.pack(fill='x', pady=5)
        self.stat_lbl = ttk.Label(self.stat_box, text="Нет данных", justify=tk.LEFT)
        self.stat_lbl.pack(padx=10, pady=10)

        self.info_box = ttk.LabelFrame(right, text="Информация")
        self.info_box.pack(fill='x', pady=5)
        self.info_lbl = ttk.Label(self.info_box, text="Элементов: 0", justify=tk.LEFT)
        self.info_lbl.pack(padx=10, pady=10)

        self.recent_box = ttk.LabelFrame(right, text="Последние 5")
        self.recent_box.pack(fill='x', pady=5)
        self.recent_lbl = ttk.Label(self.recent_box, text="—", justify=tk.LEFT)
        self.recent_lbl.pack(padx=10, pady=10)

        self.parent.bind("<Return>", lambda e: self.add())
        self.parent.bind("<Control-z>", lambda e: self.undo())
        self.parent.bind("<Control-y>", lambda e: self.redo())

    def _math_menu(self):
        try:
            self.menu.post(self.btn_math.winfo_rootx(),
                           self.btn_math.winfo_rooty() + self.btn_math.winfo_height())
        except:
            pass

    def refresh_info(self):
        arr = self.fdata if self.f_active else self.data
        n = len(arr)
        self.info_lbl.config(text=f"Элементов: {n}")
        if n > 0:
            rec = arr[-5:] if n >= 5 else arr
            if self.bivar:
                txt = ", ".join([f"({x:.2f}→{y:.2f})" for (x, y) in rec])
            else:
                txt = ", ".join(str(v) for v in rec)
            self.recent_lbl.config(text=txt)
        else:
            self.recent_lbl.config(text="—")

    def _float(self, s):
        s = s.strip().replace(',', '.')
        try:
            return float(s)
        except:
            raise ValueError("Не число")

    def add(self):
        if self.bivar:
            tk.messagebox.showinfo("Режим", "Сбросьте двумерный режим")
            return
        s = self.entry.get().strip()
        if not s:
            return
        try:
            val = self._float(s)
            self.data.append(val)
            self._save_hist()
            if self.f_active:
                self.unfilter()
            self.entry.delete(0, tk.END)
            self.show()
            self.refresh_info()
            self.msg_lbl.config(text="Добавлено", foreground="green")
        except ValueError:
            self.msg_lbl.config(text="Ошибка! Введите число", foreground="red")

    def show(self):
        arr = self.fdata if self.f_active else self.data
        self.text_box.config(state='normal')
        self.text_box.delete("1.0", tk.END)
        if self.bivar and arr:
            for x, y in arr:
                self.text_box.insert(tk.END, f"{x:.6f}\t→\t{y:.6f}\n")
        else:
            for r in arr:
                self.text_box.insert(tk.END, f"{r}\n")
        self.text_box.config(state='disabled')

    def edit(self):
        self.text_box.config(state='normal')
        try:
            first = self.text_box.index(tk.SEL_FIRST)
        except:
            tk.messagebox.showinfo("Правка", "Выделите значение")
            self.text_box.config(state='disabled')
            return
        line = int(first.split('.')[0]) - 1
        arr = self.fdata if self.f_active else self.data
        if line < 0 or line >= len(arr):
            self.text_box.config(state='disabled')
            return
        if self.bivar:
            old_x, old_y = arr[line]
            new_x = tk.simpledialog.askfloat("X", f"Текущий X = {old_x:.6f}", initialvalue=old_x)
            if new_x is not None:
                new_y = tk.simpledialog.askfloat("Y", f"Текущий Y = {old_y:.6f}", initialvalue=old_y)
                if new_y is not None:
                    if self.f_active:
                        idx = self.data.index((old_x, old_y))
                        self.data[idx] = (new_x, new_y)
                        self._save_hist()
                        self._apply_filter()
                    else:
                        self.data[line] = (new_x, new_y)
                        self._save_hist()
        else:
            old = arr[line]
            new = tk.simpledialog.askfloat("Изменить", f"Текущее {old}", initialvalue=old)
            if new is not None:
                if self.f_active:
                    idx = self.data.index(old)
                    self.data[idx] = new
                    self._save_hist()
                    self._apply_filter()
                else:
                    self.data[line] = new
                    self._save_hist()
        self.show()
        self.refresh_info()
        self.text_box.config(state='disabled')

    def clear(self):
        self.data.clear()
        self._save_hist()
        self.f_active = False
        self.fdata = None
        self.bivar = False
        self.x = []
        self.y = []
        self.show()
        self.stat_lbl.config(text="Нет данных")
        self.refresh_info()

    def stats(self):
        if self.bivar:
            tk.messagebox.showinfo("Статистика", "Для двумерных данных используйте корреляцию/регрессию")
            return
        arr = self.fdata if self.f_active else self.data
        if not arr:
            return
        avg = sum(arr)/len(arr)
        med = statistics.median(arr)
        mn = min(arr)
        mx = max(arr)
        std = statistics.stdev(arr) if len(arr) > 1 else 0
        self.stat_lbl.config(
            text=f"Среднее: {avg:.3f}\nМедиана: {med:.3f}\nСт. откл.: {std:.3f}\nMin: {mn}\nMax: {mx}"
        )

    def correl(self):
        if not self.bivar:
            tk.messagebox.showinfo("Корреляция", "Сначала загрузите X,Y")
            return
        r, p = sps.pearsonr(self.x, self.y)
        tk.messagebox.showinfo("Корреляция Пирсона",
                               f"r = {r:.4f}\np = {p:.4f}\n"
                               f"Интерпретация: |r|>0.7 – сильная, >0.3 – умеренная")

    def regress(self):
        if not self.bivar:
            tk.messagebox.showinfo("Регрессия", "Загрузите X,Y")
            return
        a, b, r_val, p_val, _ = sps.linregress(self.x, self.y)
        msg = f"y = {a:.4f} * x + {b:.4f}\nR² = {r_val**2:.4f}\np = {p_val:.4f}"
        tk.messagebox.showinfo("Регрессия", msg)
        plt.figure()
        plt.scatter(self.x, self.y, alpha=0.6, label='Данные')
        xr = np.array([min(self.x), max(self.x)])
        plt.plot(xr, a*xr + b, 'r-', label='Регрессия')
        plt.xlabel('X'); plt.ylabel('Y'); plt.title('Линейная регрессия')
        plt.legend(); plt.grid(True); plt.show(block=False)

    def poly_reg(self):
        if not self.bivar:
            tk.messagebox.showinfo("Полином", "Загрузите X,Y")
            return
        x = np.array(self.x)
        y = np.array(self.y)
        A = np.column_stack([np.ones_like(x), x, x**4])
        try:
            coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            K, D, A4 = coef
            y_pred = K + D*x + A4*x**4
            mean_y = np.mean(y)
            ss_res = np.sum((y - y_pred)**2)
            ss_tot = np.sum((y - mean_y)**2)
            R = np.sqrt(1 - ss_res/ss_tot) if ss_tot != 0 else 0
            msg = (f"Полином 4-й степени:\nZ = {K:.6f} + {D:.6f}*t + {A4:.6f}*t^4\n"
                   f"R = {R:.6f}\nСумма кв. ошибок = {ss_res:.6f}")
            tk.messagebox.showinfo("Результат", msg)
            plt.figure()
            plt.scatter(x, y, label="Данные")
            x_sm = np.linspace(min(x), max(x), 200)
            plt.plot(x_sm, K + D*x_sm + A4*x_sm**4, 'r-', label="Полином")
            plt.xlabel('t'); plt.ylabel('Z'); plt.title('Полиномиальная регрессия')
            plt.legend(); plt.grid(True); plt.show(block=False)
        except Exception as e:
            tk.messagebox.showerror("Ошибка", str(e))

    def _plot_interp(self, t, res, func, title):
        plt.figure()
        plt.scatter(self.x, self.y, color='red', label='Точки', zorder=5)
        xs = np.linspace(min(self.x), max(self.x), 200)
        plt.plot(xs, func(xs), 'b-', label=title)
        plt.plot(t, res, 'go', markersize=8, label=f't={t}')
        plt.xlabel('t'); plt.ylabel('Z'); plt.title(title)
        plt.legend(); plt.grid(True); plt.show(block=False)

    def lagrange(self):
        if not self.bivar:
            tk.messagebox.showinfo("Лагранж", "Загрузите X,Y")
            return
        t = tk.simpledialog.askfloat("Точка", "Введите t:")
        if t is None: return
        try:
            poly = spi.lagrange(self.x, self.y)
            res = poly(t)
            self._plot_interp(t, res, poly, "Интерполяция Лагранжа")
            tk.messagebox.showinfo("Результат", f"Значение в t={t}: {res:.6f}")
        except Exception as e:
            tk.messagebox.showerror("Ошибка", str(e))

    def newton(self):
        if not self.bivar:
            tk.messagebox.showinfo("Ньютон", "Загрузите X,Y")
            return
        t = tk.simpledialog.askfloat("Точка", "Введите t:")
        if t is None: return
        x = self.x
        y = self.y
        n = len(x)
        F = [[0]*n for _ in range(n)]
        for i in range(n):
            F[i][0] = y[i]
        for j in range(1, n):
            for i in range(n-j):
                F[i][j] = (F[i+1][j-1] - F[i][j-1]) / (x[i+j] - x[i])
        def pol(tt):
            res = F[0][0]
            term = 1.0
            for i in range(1, n):
                term *= (tt - x[i-1])
                res += F[0][i] * term
            return res
        res = pol(t)
        self._plot_interp(t, res, pol, "Интерполяция Ньютона")
        tk.messagebox.showinfo("Результат", f"Значение в t={t}: {res:.6f}")

    def canon(self):
        if not self.bivar:
            tk.messagebox.showinfo("Канонический", "Загрузите X,Y")
            return
        t = tk.simpledialog.askfloat("Точка", "Введите t:")
        if t is None: return
        x = self.x
        y = self.y
        n = len(x)
        A = np.vander(x, n, increasing=True)
        try:
            coef = np.linalg.solve(A, y)
            def pol(tt):
                return sum(coef[i] * (tt**i) for i in range(n))
            res = pol(t)
            self._plot_interp(t, res, pol, "Канонический полином")
            tk.messagebox.showinfo("Результат", f"Значение в t={t}: {res:.6f}")
        except np.linalg.LinAlgError:
            tk.messagebox.showerror("Ошибка", "Матрица вырождена (точки должны быть различны)")

    def lin_spline(self):
        if not self.bivar:
            tk.messagebox.showinfo("Линейный сплайн", "Загрузите X,Y")
            return
        t = tk.simpledialog.askfloat("Точка", "Введите t:")
        if t is None: return
        try:
            f = spi.interp1d(self.x, self.y, kind='linear', fill_value="extrapolate")
            res = f(t)
            self._plot_interp(t, res, f, "Линейный сплайн")
            tk.messagebox.showinfo("Результат", f"Значение в t={t}: {res:.6f}")
        except Exception as e:
            tk.messagebox.showerror("Ошибка", str(e))

    def cub_spline(self):
        if not self.bivar:
            tk.messagebox.showinfo("Кубический сплайн", "Загрузите X,Y")
            return
        t = tk.simpledialog.askfloat("Точка", "Введите t:")
        if t is None: return
        try:
            cs = spi.CubicSpline(self.x, self.y, bc_type='natural')
            res = cs(t)
            self._plot_interp(t, res, cs, "Кубический сплайн")
            tk.messagebox.showinfo("Результат", f"Значение в t={t}: {res:.6f}")
        except Exception as e:
            tk.messagebox.showerror("Ошибка", str(e))

    def ttest(self):
        if self.bivar:
            tk.messagebox.showinfo("t-тест", "Используйте одномерные данные")
            return
        if len(self.data) == 0:
            tk.messagebox.showwarning("Нет данных", "Добавьте числа")
            return
        if tk.messagebox.askyesno("t-тест", "Сравнить первую половину со второй?"):
            mid = len(self.data)//2
            g1 = self.data[:mid]
            g2 = self.data[mid:]
            if len(g1) < 2 or len(g2) < 2:
                tk.messagebox.showerror("Ошибка", "Мало данных в группах")
                return
            t_stat, p_val = sps.ttest_ind(g1, g2)
            msg = (f"Группа 1: n={len(g1)}, среднее={np.mean(g1):.3f}\n"
                   f"Группа 2: n={len(g2)}, среднее={np.mean(g2):.3f}\n"
                   f"t = {t_stat:.4f}, p = {p_val:.4f}\n"
                   f"{'Значимы' if p_val < 0.05 else 'Не значимы'}")
            tk.messagebox.showinfo("t-тест", msg)
        else:
            s = tk.simpledialog.askstring("Вторая выборка", "Введите числа через запятую")
            if s:
                try:
                    g2 = [float(x.strip().replace(',','.')) for x in s.split(',')]
                    t_stat, p_val = sps.ttest_ind(self.data, g2)
                    tk.messagebox.showinfo("t-тест", f"t = {t_stat:.4f}, p = {p_val:.4f}")
                except:
                    tk.messagebox.showerror("Ошибка", "Неверный формат")

    def outliers(self):
        if self.bivar:
            tk.messagebox.showinfo("Выбросы", "Только одномерные данные")
            return
        d = self.data[:]
        if len(d) < 4:
            tk.messagebox.showwarning("Мало данных", "Нужно минимум 4 точки")
            return
        q1 = np.percentile(d, 25)
        q3 = np.percentile(d, 75)
        iqr = q3 - q1
        lo = q1 - 1.5*iqr
        hi = q3 + 1.5*iqr
        filt = [x for x in d if lo <= x <= hi]
        rem = len(d) - len(filt)
        if tk.messagebox.askyesno("Выбросы", f"Удалено {rem} выбросов. Продолжить?"):
            self.data = filt
            self._save_hist()
            if self.f_active:
                self.unfilter()
            self.show()
            self.refresh_info()
            self.msg_lbl.config(text=f"Удалено {rem} выбросов", foreground="blue")

    def norm(self):
        if self.bivar:
            tk.messagebox.showinfo("Нормализация", "Только одномерные данные")
            return
        if not self.data: return
        mn = min(self.data)
        mx = max(self.data)
        if mx == mn:
            tk.messagebox.showwarning("Нет разброса", "Все числа одинаковы")
            return
        self.data = [(x - mn)/(mx - mn) for x in self.data]
        self._save_hist()
        self.show()
        self.refresh_info()
        self.msg_lbl.config(text="Нормализовано [0,1]")

    def stand(self):
        if self.bivar:
            tk.messagebox.showinfo("Стандартизация", "Только одномерные данные")
            return
        if len(self.data) < 2: return
        mean = np.mean(self.data)
        std = np.std(self.data)
        if std == 0:
            tk.messagebox.showwarning("Нет разброса", "σ = 0")
            return
        self.data = [(x - mean)/std for x in self.data]
        self._save_hist()
        self.show()
        self.refresh_info()
        self.msg_lbl.config(text="Стандартизировано (z-score)")

    def conf_int(self):
        if self.bivar:
            tk.messagebox.showinfo("Дов. интервал", "Только одномерные данные")
            return
        arr = self.fdata if self.f_active else self.data
        if len(arr) < 2: return
        mean = np.mean(arr)
        sem = sps.sem(arr)
        ci = sps.t.interval(0.95, len(arr)-1, loc=mean, scale=sem)
        tk.messagebox.showinfo("95% ДИ для среднего",
                               f"Среднее = {mean:.3f}\nДИ = [{ci[0]:.3f}, {ci[1]:.3f}]")

    def load_xy(self):
        path = tk.filedialog.askopenfilename(filetypes=[("CSV или TXT", "*.csv *.txt")])
        if not path: return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            xv, yv = [], []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        xv.append(float(parts[0].replace(',','.')))
                        yv.append(float(parts[1].replace(',','.')))
                    except: pass
            if len(xv) > 1:
                self.x = xv
                self.y = yv
                self.bivar = True
                self.data = list(zip(self.x, self.y))
                self._save_hist()
                self.show()
                self.refresh_info()
                tk.messagebox.showinfo("Успех", f"Загружено {len(xv)} пар")
            else:
                tk.messagebox.showerror("Ошибка", "Файл должен содержать два числовых столбца")
        except Exception as e:
            tk.messagebox.showerror("Ошибка загрузки", str(e))

    def save(self):
        arr = self.fdata if self.f_active else self.data
        if not arr: return
        types = [("Текстовый файл", "*.txt"), ("CSV", "*.csv")]
        if XL_OK: types.append(("Excel", "*.xlsx"))
        path = tk.filedialog.asksaveasfilename(defaultextension=".txt", filetypes=types)
        if not path: return
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == '.txt':
                with open(path, 'w', encoding='utf-8') as f:
                    if self.bivar:
                        for x, y in arr: f.write(f"{x}\t{y}\n")
                    else:
                        for v in arr: f.write(f"{v}\n")
            elif ext == '.csv':
                import csv
                with open(path, 'w', encoding='utf-8', newline='') as f:
                    w = csv.writer(f)
                    if self.bivar:
                        w.writerow(["X","Y"])
                        for x, y in arr: w.writerow([x,y])
                    else:
                        w.writerow(["Value"])
                        for v in arr: w.writerow([v])
            elif ext == '.xlsx' and XL_OK:
                wb = openpyxl.Workbook()
                ws = wb.active
                if self.bivar:
                    ws.title = "XY"
                    ws.append(["X","Y"])
                    for x, y in arr: ws.append([x,y])
                else:
                    ws.title = "Data"
                    ws.append(["Value"])
                    for v in arr: ws.append([v])
                wb.save(path)
            else:
                tk.messagebox.showerror("Ошибка", "Формат не поддерживается")
                return
            tk.messagebox.showinfo("Сохранено", f"Сохранено {len(arr)} записей")
        except Exception as e:
            tk.messagebox.showerror("Ошибка сохранения", str(e))

    def _extract_nums(self, text):
        nums = []
        for t in re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', text):
            try:
                nums.append(float(t))
            except: pass
        return nums

    def load_many(self):
        if self.bivar:
            tk.messagebox.showinfo("Пакетная загрузка", "Сначала очистите данные")
            return
        paths = tk.filedialog.askopenfilenames(
            filetypes=[("Все поддерживаемые", "*.txt *.csv *.xlsx *.docx"),
                       ("Текстовые", "*.txt"), ("CSV", "*.csv"),
                       ("Excel", "*.xlsx"), ("Word", "*.docx")]
        )
        if not paths: return
        total = []
        for path in paths:
            ext = os.path.splitext(path)[1].lower()
            try:
                if ext in ('.txt', '.csv'):
                    with open(path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    total += self._extract_nums(text)
                elif ext == '.xlsx' and XL_OK:
                    wb = openpyxl.load_workbook(path, data_only=True)
                    for sheet in wb.worksheets:
                        for row in sheet.iter_rows(values_only=True):
                            for cell in row:
                                if isinstance(cell, (int, float)):
                                    total.append(float(cell))
                elif ext == '.docx':
                    doc = docx.Document(path)
                    text = '\n'.join([p.text for p in doc.paragraphs])
                    total += self._extract_nums(text)
            except Exception as e:
                tk.messagebox.showerror("Ошибка", f"{os.path.basename(path)}:\n{str(e)}")
                return
        if total:
            self.data += total
            self._save_hist()
            self.show()
            self.refresh_info()
            self.msg_lbl.config(text=f"Загружено {len(total)} чисел", foreground="green")
        else:
            tk.messagebox.showwarning("Нет данных", "Не удалось извлечь числа")

    def gen_rand(self):
        if self.bivar:
            tk.messagebox.showinfo("Генерация", "Сначала очистите данные")
            return
        n = tk.simpledialog.askinteger("Генерация", "Количество чисел", minvalue=1)
        if not n: return
        dist = tk.simpledialog.askstring("Распределение", "normal / uniform", initialvalue="uniform")
        if dist == 'normal':
            vals = np.random.normal(50, 15, n).tolist()
        else:
            vals = [random.uniform(0, 100) for _ in range(n)]
        self.data += vals
        self._save_hist()
        self.show()
        self.refresh_info()

    def hist_plot(self):
        if self.bivar:
            tk.messagebox.showinfo("Гистограмма", "Только одномерные данные")
            return
        arr = self.fdata if self.f_active else self.data
        if not arr: return
        plt.figure()
        plt.hist(arr, bins=10, alpha=0.7, color='blue')
        plt.title('Гистограмма')
        plt.show(block=False)

    def graph(self):
        if self.bivar:
            tk.messagebox.showinfo("График", "Только одномерные данные")
            return
        arr = self.fdata if self.f_active else self.data
        if not arr: return
        plt.figure()
        plt.plot(arr, marker='o')
        plt.title('Линейный график')
        plt.show(block=False)

    def sort(self):
        if self.bivar:
            tk.messagebox.showinfo("Сортировка", "Пары X,Y не сортируются")
            return
        arr = self.fdata if self.f_active else self.data
        if not arr: return
        arr.sort()
        if self.f_active:
            self.fdata = arr
        else:
            self.data = arr
        self._save_hist()
        self.show()

    def search(self):
        if self.bivar:
            tk.messagebox.showinfo("Поиск", "Только одномерные данные")
            return
        s = self.entry.get()
        if not s: return
        try:
            num = self._float(s)
            arr = self.fdata if self.f_active else self.data
            if num in arr:
                tk.messagebox.showinfo("Поиск", f"{num} найдено")
            else:
                tk.messagebox.showinfo("Поиск", "Не найдено")
        except:
            tk.messagebox.showerror("Ошибка", "Введите число")

    def filter_(self):
        if self.bivar:
            tk.messagebox.showinfo("Фильтр", "Только одномерные данные")
            return
        if not self.data: return
        dlg = tk.Toplevel(self.parent)
        dlg.title("Фильтр")
        dlg.geometry("300x250")
        ttk.Label(dlg, text="Условие:").pack(pady=5)
        cond_var = tk.StringVar(value=">")
        for c, d in (">",">"), ("<","<"), (">=",">="), ("<=","<="):
            ttk.Radiobutton(dlg, text=d, variable=cond_var, value=c).pack(anchor='w')
        ttk.Label(dlg, text="Значение:").pack()
        val_ent = ttk.Entry(dlg)
        val_ent.pack()
        def apply():
            try:
                thr = float(val_ent.get().replace(',','.'))
                cond = cond_var.get()
                self.fdata = [x for x in self.data if eval(f"{x}{cond}{thr}")]
                self.f_active = True
                self.show()
                self.refresh_info()
                dlg.destroy()
            except:
                tk.messagebox.showerror("Ошибка", "Неверное число")
        ttk.Button(dlg, text="Применить", command=apply).pack(pady=10)

    def unfilter(self):
        self.f_active = False
        self.fdata = None
        self.show()
        self.refresh_info()

    def _apply_filter(self):
        self.unfilter()


class TxtTab:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.lang = 'russian'
        self.langs = {
            'russian': 'Русский', 'english': 'English', 'german': 'Deutsch',
            'french': 'Français', 'spanish': 'Español'
        }
        self.stops = set()
        self.stem = None

        self.analyzers = {}
        if VADER_OK:
            try:
                self.analyzers['english'] = SentimentIntensityAnalyzer()
            except: pass
        if DOST_OK:
            try:
                tokenizer = RegexTokenizer()
                model = FastTextSocialNetworkModel(tokenizer=tokenizer)
                self.analyzers['russian'] = model
            except: pass

        self._build()
        self._set_lang()

    def _build(self):
        btn_frame = ttk.Frame(self.parent)
        btn_frame.pack(pady=10, fill='x')

        def btn(text, cmd):
            b = ttk.Button(btn_frame, text=text, command=cmd)
            b.pack(side=tk.LEFT, padx=2, pady=2)
            return b

        for (text, cmd) in [
            ("📂 Загрузить", self.load),
            ("💾 Сохранить", self.save),
            ("🗑️ Очистить", self.clear),
            ("📋 Статистика", self.show_stats),
            ("🔍 Поиск", self.search),
            ("☁️ Облако слов", self.cloud),
            ("😊 Тональность", self.sentiment),
            ("🚫 Удалить стоп-слова", self.del_stops),
            ("🌱 Стемминг", self.stem_words),
        ]:
            btn(text, cmd)

        btn("↶", self.undo)
        btn("↷", self.redo)

        main = ttk.Frame(self.parent)
        main.pack(fill='both', expand=True, padx=10, pady=10)

        left = ttk.Frame(main)
        left.pack(side=tk.LEFT, fill='both', expand=True)
        right = ttk.Frame(main, width=350)
        right.pack(side=tk.RIGHT, fill='y', padx=(10,0))
        right.pack_propagate(False)

        self.text_box = tk.Text(left, height=25, font=("Consolas", 10),
                                undo=True, autoseparators=True, maxundo=100)
        self.text_box.pack(fill='both', expand=True)
        self.text_box.bind("<KeyRelease>", lambda e: self.refresh_stats())
        self.text_box.bind("<Control-z>", lambda e: self.undo())
        self.text_box.bind("<Control-y>", lambda e: self.redo())

        self.stat_box = ttk.LabelFrame(right, text="Статистика")
        self.stat_box.pack(fill='x', pady=5)
        self.stat_lbl = ttk.Label(self.stat_box, text="Нет данных", justify=tk.LEFT)
        self.stat_lbl.pack(padx=10, pady=10)

        self.sent_box = ttk.LabelFrame(right, text="Тональность")
        self.sent_box.pack(fill='x', pady=5)
        self.sent_lbl = ttk.Label(self.sent_box, text="—", justify=tk.LEFT)
        self.sent_lbl.pack(padx=10, pady=10)

        self.lang_box = ttk.LabelFrame(right, text="Язык текста")
        self.lang_box.pack(fill='x', pady=5)
        self.lang_var = tk.StringVar(value='russian')
        self.lang_cb = ttk.Combobox(self.lang_box, textvariable=self.lang_var,
                                    values=list(self.langs.keys()), state='readonly')
        self.lang_cb.pack(padx=10, pady=10, fill='x')
        self.lang_cb.bind('<<ComboboxSelected>>', self._chg_lang)

    def undo(self, event=None):
        try:
            self.text_box.edit_undo()
            self.refresh_stats()
        except: pass

    def redo(self, event=None):
        try:
            self.text_box.edit_redo()
            self.refresh_stats()
        except: pass

    def _set_lang(self):
        try:
            self.stops = set(nltk.corpus.stopwords.words(self.lang))
        except:
            self.stops = set()
        try:
            self.stem = nltk.stem.SnowballStemmer(self.lang)
        except:
            self.stem = None
        self.refresh_stats()

    def _chg_lang(self, event=None):
        self.lang = self.lang_var.get()
        self._set_lang()
        self.sentiment()

    def get_txt(self):
        return self.text_box.get("1.0", tk.END).strip()

    def words_from(self, text):
        return re.findall(r'\b\w+\b', text.lower(), flags=re.UNICODE)

    def refresh_stats(self):
        text = self.get_txt()
        if not text:
            self.stat_lbl.config(text="Нет текста")
            return
        chars = len(text)
        words = self.words_from(text)
        wc = len(words)
        sents = text.count('.') + text.count('!') + text.count('?')
        self.stat_lbl.config(text=f"Символов: {chars}\nСлов: {wc}\nПредложений: {sents}")
        self.sentiment()

    def load(self):
        path = tk.filedialog.askopenfilename(
            filetypes=[("Все поддерживаемые", "*.txt *.csv *.pdf *.docx"),
                       ("Текстовые", "*.txt"), ("CSV", "*.csv"),
                       ("PDF", "*.pdf"), ("Word", "*.docx")]
        )
        if not path: return
        content = ""
        try:
            if path.lower().endswith(('.txt', '.csv')):
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
            elif path.lower().endswith('.pdf'):
                with open(path, 'rb') as f:
                    reader = pypdf.PdfReader(f)
                    for page in reader.pages:
                        content += page.extract_text()
            elif path.lower().endswith('.docx'):
                doc = docx.Document(path)
                content = '\n'.join([p.text for p in doc.paragraphs])
            else:
                tk.messagebox.showerror("Ошибка", "Неподдерживаемый формат")
                return
        except Exception as e:
            tk.messagebox.showerror("Ошибка загрузки", str(e))
            return
        if content:
            self.text_box.delete("1.0", tk.END)
            self.text_box.insert("1.0", content)
            self.refresh_stats()
        else:
            tk.messagebox.showwarning("Пустой файл", "Не удалось извлечь текст")

    def save(self):
        text = self.get_txt()
        if not text:
            tk.messagebox.showwarning("Нет данных", "Нечего сохранять")
            return
        types = [("Текстовый", "*.txt"), ("CSV", "*.csv")]
        if XL_OK: types.append(("Excel", "*.xlsx"))
        path = tk.filedialog.asksaveasfilename(defaultextension=".txt", filetypes=types)
        if not path: return
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == '.txt':
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(text)
            elif ext == '.csv':
                import csv
                with open(path, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Text"])
                    writer.writerow([text])
            elif ext == '.xlsx' and XL_OK:
                wb = openpyxl.Workbook()
                ws = wb.active
                ws.title = "Text"
                ws.append(["Content"])
                for line in text.split('\n'):
                    ws.append([line])
                wb.save(path)
            else:
                tk.messagebox.showerror("Ошибка", "Неподдерживаемый формат")
                return
            tk.messagebox.showinfo("Сохранено", "Текст сохранён")
        except Exception as e:
            tk.messagebox.showerror("Ошибка сохранения", str(e))

    def clear(self):
        self.text_box.delete("1.0", tk.END)
        self.refresh_stats()

    def show_stats(self):
        self.refresh_stats()
        tk.messagebox.showinfo("Статистика", self.stat_lbl.cget("text"))

    def search(self):
        word = tk.simpledialog.askstring("Поиск", "Слово/фрагмент:")
        if word:
            count = self.get_txt().lower().count(word.lower())
            tk.messagebox.showinfo("Поиск", f"Вхождений: {count}")

    def cloud(self):
        text = self.get_txt()
        if not text: return
        wc_obj = wc.WordCloud(width=800, height=400,
                              background_color='white',
                              stopwords=self.stops).generate(text)
        plt.figure()
        plt.imshow(wc_obj, interpolation='bilinear')
        plt.axis('off')
        plt.show(block=False)

    def sentiment(self):
        lang = self.lang
        text = self.get_txt()
        if not text:
            self.sent_lbl.config(text="Нет текста")
            return

        if lang in self.analyzers:
            try:
                if lang == 'english' and VADER_OK:
                    scores = self.analyzers['english'].polarity_scores(text)
                    comp = scores['compound']
                    if comp >= 0.05:
                        sent = f"Позитивное ({comp:.2f})"
                    elif comp <= -0.05:
                        sent = f"Негативное ({comp:.2f})"
                    else:
                        sent = "Нейтральное"
                    det = f"pos={scores['pos']:.2f} neg={scores['neg']:.2f} neu={scores['neu']:.2f}"
                    self.sent_lbl.config(text=f"{sent}\n{det}")
                    return
                if lang == 'russian' and DOST_OK:
                    model = self.analyzers['russian']
                    res = model.predict([text])
                    if res and len(res[0]) > 0:
                        em = res[0][0]
                        probs = {k: v for k, v in em.items() if k in ('positive','negative','neutral')}
                        if probs:
                            dom = max(probs, key=probs.get)
                            sent = f"{dom.capitalize()} ({probs[dom]:.2f})"
                            det = f"pos={probs['positive']:.2f} neg={probs['negative']:.2f} neu={probs['neutral']:.2f}"
                            self.sent_lbl.config(text=f"{sent}\n{det}")
                            return
            except:
                pass

        words = self.words_from(text)
        pos_dict = {
            'russian': {'хороший','отличный','прекрасный','замечательный','великий','любовь','счастье','радость',
                        'добрый','великолепный','чудесный','восхитительный','солнечный','успешный','позитивный'},
            'english': {'good','great','excellent','wonderful','fantastic','love','happy','joy',
                        'amazing','beautiful','nice','pleased','glorious','superb','brilliant'},
            'german': {'gut','großartig','ausgezeichnet','wunderbar','fantastisch','liebe','glücklich','freude',
                       'hervorragend','prima','positiv','schön','erfolgreich','angenehm'},
            'french': {'bon','excellent','merveilleux','fantastique','amour','bonheur','joie',
                       'magnifique','superbe','positif','agréable','charmant','heureux'},
            'spanish': {'bueno','excelente','maravilloso','fantástico','amor','felicidad','alegría',
                        'genial','positivo','bonito','espléndido','encantador','agradable'}
        }
        neg_dict = {
            'russian': {'плохой','ужасный','отвратительный','грустный','ненависть','зло','печаль',
                        'мерзкий','скверный','негативный','паршивый','унылый','тоскливый'},
            'english': {'bad','terrible','awful','horrible','sad','hate','angry','pain',
                        'nasty','negative','poor','disgusting','miserable','unpleasant'},
            'german': {'schlecht','schrecklich','furchtbar','traurig','hass','wut','schmerz',
                       'übel','negativ','elend','miserabel','unangenehm'},
            'french': {'mauvais','terrible','horrible','triste','haine','colère','douleur',
                       'méchant','négatif','pénible','désagréable','misérable'},
            'spanish': {'malo','terrible','horrible','triste','odio','enfado','dolor',
                        'negativo','desagradable','pésimo','lamentable','penoso'}
        }
        pos_set = pos_dict.get(lang, set())
        neg_set = neg_dict.get(lang, set())
        if not pos_set or not neg_set:
            self.sent_lbl.config(text="Тональность не реализована")
            return
        pos = sum(1 for w in words if w in pos_set)
        neg = sum(1 for w in words if w in neg_set)
        if pos + neg == 0:
            sent = "Нейтрально"
        else:
            ratio = pos / (pos+neg)
            if ratio > 0.6:
                sent = f"Позитивное {ratio*100:.1f}%"
            elif ratio < 0.4:
                sent = f"Негативное {(1-ratio)*100:.1f}%"
            else:
                sent = "Нейтрально"
        self.sent_lbl.config(text=f"Позитив: {pos}, Негатив: {neg}\n{sent}")

    def del_stops(self):
        text = self.get_txt()
        words = self.words_from(text)
        filt = [w for w in words if w not in self.stops]
        self.text_box.delete("1.0", tk.END)
        self.text_box.insert("1.0", ' '.join(filt))

    def stem_words(self):
        if not self.stem:
            tk.messagebox.showwarning("Стемминг", "Для выбранного языка стемминг недоступен")
            return
        text = self.get_txt()
        words = self.words_from(text)
        stemmed = [self.stem.stem(w) for w in words]
        self.text_box.delete("1.0", tk.END)
        self.text_box.insert("1.0", ' '.join(stemmed))


if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style()
    style.theme_use('clam')

    default_font = ('Segoe UI Emoji', 10)  
    style.configure('TButton', font=default_font, padding=6)
    style.configure('TLabel', font=default_font)
    style.configure('TLabelframe.Label', font=default_font)
    style.configure('TEntry', font=default_font)

    app = App(root)
    root.mainloop()