import tkinter as tk
from tkinter import ttk, messagebox
import threading
import json
import os
import datetime

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from boroondara_search import find_routes, build_travel_time_graph

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.json')

ALGORITHMS = {
    'A* (AS)'                      : 'AS',
    'Breadth-First Search (BFS)'   : 'BFS',
    'Depth-First Search (DFS)'     : 'DFS',
    'Greedy Best-First (GBFS)'     : 'GBFS',
    'Bidirectional BFS (CUS1)'     : 'CUS1',
    'Recursive Best-First (CUS2)'  : 'CUS2',
}

MULTI_ROUTE_ALGORITHMS = {'AS'}
ROUTE_COLOURS = ['#e63946', '#f4a261', '#2a9d8f', '#6a4c93', '#457b9d']


def load_config() -> dict:
    with open(CONFIG_PATH, 'r') as f:
        return json.load(f)


def _try_import_predict():
    try:
        from predictor import predict
        return predict
    except Exception:
        return None


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Boroondara Traffic Route Guidance System')
        self.resizable(True, True)
        self.config = load_config()

        self._node_list, self._sites = build_travel_time_graph()
        self._scats_numbers = sorted(self._sites.keys())
        self._scats_labels = [
            f"{s} – {self._sites[s]['loc']}" for s in self._scats_numbers
        ]

        self._current_path = None
        self._pan_start = None
        self._route_artists = []
        self._current_routes = []
        self._predict = _try_import_predict()

        # pre-compute base map edge data once so _draw_base_map is fast
        self._edge_segments = []
        drawn = set()
        for scats, node in self._node_list.items():
            for nb, _ in node.edges:
                pair = (min(scats, nb), max(scats, nb))
                if pair in drawn:
                    continue
                drawn.add(pair)
                lon1, lat1 = self._sites[scats]['lon'], self._sites[scats]['lat']
                lon2, lat2 = self._sites[nb]['lon'], self._sites[nb]['lat']
                self._edge_segments.append(([lon1, lon2], [lat1, lat2]))

        self._build_ui()
        self._draw_base_map()

    def _build_ui(self):
        left = ttk.Frame(self, padding=8)
        left.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(left, text='Origin SCATS').pack(anchor=tk.W)
        self._origin_var = tk.StringVar()
        self._origin_cb = ttk.Combobox(
            left, textvariable=self._origin_var,
            values=self._scats_labels, width=38, state='readonly'
        )
        self._origin_cb.pack(pady=(0, 6), fill=tk.X)

        ttk.Label(left, text='Destination SCATS').pack(anchor=tk.W)
        self._dest_var = tk.StringVar()
        self._dest_cb = ttk.Combobox(
            left, textvariable=self._dest_var,
            values=self._scats_labels, width=38, state='readonly'
        )
        self._dest_cb.pack(pady=(0, 6), fill=tk.X)

        ttk.Label(left, text='Algorithm').pack(anchor=tk.W)
        self._algo_var = tk.StringVar(value=list(ALGORITHMS.keys())[0])
        self._algo_cb = ttk.Combobox(
            left, textvariable=self._algo_var,
            values=list(ALGORITHMS.keys()), width=38, state='readonly'
        )
        self._algo_cb.pack(pady=(0, 6), fill=tk.X)
        self._algo_cb.bind('<<ComboboxSelected>>', self._on_algo_changed)

        ttk.Label(left, text='Prediction Model').pack(anchor=tk.W)
        self._model_var = tk.StringVar(value='gru')
        self._model_cb = ttk.Combobox(
            left, textvariable=self._model_var,
            values=['gru', 'lstm', 'rnn', 'updated_rnn'], width=38, state='readonly'
        )
        self._model_cb.pack(pady=(0, 6), fill=tk.X)

        # date/time picker
        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)
        now = datetime.datetime.now()

        ttk.Label(left, text='Date', foreground='grey', font=('TkDefaultFont', 8)).pack(anchor=tk.W)
        date_row = ttk.Frame(left)
        date_row.pack(fill=tk.X, pady=(2, 4))
        self._date_day_cb = ttk.Combobox(date_row, values=[f'{d:02d}' for d in range(1, 32)], width=3, state='readonly')
        self._date_day_cb.set(f'{now.day:02d}')
        self._date_day_cb.pack(side=tk.LEFT)
        ttk.Label(date_row, text='/').pack(side=tk.LEFT, padx=2)
        self._date_month_cb = ttk.Combobox(date_row, values=[f'{m:02d}' for m in range(1, 13)], width=3, state='readonly')
        self._date_month_cb.set(f'{now.month:02d}')
        self._date_month_cb.pack(side=tk.LEFT)
        ttk.Label(date_row, text='/').pack(side=tk.LEFT, padx=2)
        self._date_year_cb = ttk.Combobox(date_row, values=[str(y) for y in range(2000, 2027)], width=5, state='readonly')
        self._date_year_cb.set(str(now.year))
        self._date_year_cb.pack(side=tk.LEFT)

        ttk.Label(left, text='Time', foreground='grey', font=('TkDefaultFont', 8)).pack(anchor=tk.W)
        time_row = ttk.Frame(left)
        time_row.pack(fill=tk.X, pady=(2, 4))
        self._time_hour_cb = ttk.Combobox(time_row, values=[f'{h:02d}' for h in range(24)], width=3, state='readonly')
        self._time_hour_cb.set(f'{now.hour:02d}')
        self._time_hour_cb.pack(side=tk.LEFT)
        ttk.Label(time_row, text=':').pack(side=tk.LEFT, padx=2)
        self._time_min_cb = ttk.Combobox(time_row, values=[f'{m:02d}' for m in range(60)], width=3, state='readonly')
        self._time_min_cb.set(f'{now.minute:02d}')
        self._time_min_cb.pack(side=tk.LEFT)

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)

        self._routes_note = ttk.Label(left, text='Returns up to 5 routes', foreground='grey', font=('TkDefaultFont', 8))
        self._routes_note.pack(anchor=tk.W, pady=(0, 4))

        self._find_btn = ttk.Button(left, text='Find Routes', command=self._on_find)
        self._find_btn.pack(fill=tk.X, pady=(0, 3))
        ttk.Button(left, text='Exit', command=self._exit).pack(fill=tk.X, pady=(0, 4))

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)
        results_header = ttk.Frame(left)
        results_header.pack(fill=tk.X)
        ttk.Label(results_header, text='Results').pack(side=tk.LEFT, anchor=tk.W)
        self._show_all_btn = ttk.Button(results_header, text='Show all routes', command=self._show_all_routes)

        # scrollable results
        results_outer = ttk.Frame(left)
        results_outer.pack(fill=tk.BOTH, expand=True)

        self._results_canvas = tk.Canvas(results_outer, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(results_outer, orient=tk.VERTICAL, command=self._results_canvas.yview)
        self._results_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._results_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._results_frame = ttk.Frame(self._results_canvas)
        self._results_win_id = self._results_canvas.create_window((0, 0), window=self._results_frame, anchor='nw')
        self._results_frame.bind('<Configure>', lambda e: self._results_canvas.configure(scrollregion=self._results_canvas.bbox('all')))
        self._results_canvas.bind('<Configure>', lambda e: self._results_canvas.itemconfig(self._results_win_id, width=e.width))
        self._results_canvas.bind('<Enter>', self._bind_scroll)
        self._results_canvas.bind('<Leave>', self._unbind_scroll)
        self._results_frame.bind('<Enter>', self._bind_scroll)
        self._results_frame.bind('<Leave>', self._unbind_scroll)

        self._status_var = tk.StringVar(value='Ready.')
        ttk.Label(left, textvariable=self._status_var, foreground='grey').pack(anchor=tk.W, pady=(4, 0))

        right = ttk.Frame(self, padding=10)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self._fig, self._ax = plt.subplots(figsize=(7, 6))
        self._canvas = FigureCanvasTkAgg(self._fig, master=right)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._canvas.get_tk_widget().bind('<MouseWheel>', self._on_map_zoom)
        self._canvas.get_tk_widget().bind('<ButtonPress-1>', self._on_pan_start)
        self._canvas.get_tk_widget().bind('<B1-Motion>', self._on_pan_move)

    def _bind_scroll(self, _event):
        self._results_canvas.bind_all('<MouseWheel>', lambda e: self._results_canvas.yview_scroll(int(-1*(e.delta/120)), 'units'))
        self._results_canvas.bind_all('<Up>',   lambda e: self._results_canvas.yview_scroll(-1, 'units'))
        self._results_canvas.bind_all('<Down>', lambda e: self._results_canvas.yview_scroll(1, 'units'))

    def _unbind_scroll(self, _event):
        self._results_canvas.unbind_all('<MouseWheel>')
        self._results_canvas.unbind_all('<Up>')
        self._results_canvas.unbind_all('<Down>')

    def _on_map_zoom(self, event):
        factor = 1.1 if event.delta > 0 else 0.9
        xlim = self._ax.get_xlim()
        ylim = self._ax.get_ylim()
        x_mid = (xlim[0] + xlim[1]) / 2
        y_mid = (ylim[0] + ylim[1]) / 2
        self._ax.set_xlim(x_mid - (xlim[1]-xlim[0])*factor/2, x_mid + (xlim[1]-xlim[0])*factor/2)
        self._ax.set_ylim(y_mid - (ylim[1]-ylim[0])*factor/2, y_mid + (ylim[1]-ylim[0])*factor/2)
        self._canvas.draw_idle()

    def _on_pan_start(self, event):
        self._pan_start = (event.x, event.y)

    def _on_pan_move(self, event):
        if self._pan_start is None:
            return
        dx_px = event.x - self._pan_start[0]
        dy_px = event.y - self._pan_start[1]
        self._pan_start = (event.x, event.y)
        fig_w, fig_h = self._fig.get_size_inches() * self._fig.dpi
        xlim = self._ax.get_xlim()
        ylim = self._ax.get_ylim()
        ax_bbox = self._ax.get_position()
        dx_data = -(dx_px / (ax_bbox.width  * fig_w)) * (xlim[1] - xlim[0])
        dy_data =  (dy_px / (ax_bbox.height * fig_h)) * (ylim[1] - ylim[0])
        self._ax.set_xlim(xlim[0]+dx_data, xlim[1]+dx_data)
        self._ax.set_ylim(ylim[0]+dy_data, ylim[1]+dy_data)
        self._canvas.draw_idle()

    def _exit(self):
        self.destroy()
        raise SystemExit(0)

    def _on_algo_changed(self, _event=None):
        algo_code = ALGORITHMS.get(self._algo_var.get(), 'AS')
        if algo_code in MULTI_ROUTE_ALGORITHMS:
            self._routes_note.config(text='Returns up to 5 routes')
            self._find_btn.config(text='Find Routes')
        else:
            self._routes_note.config(text='Returns 1 route')
            self._find_btn.config(text='Find Route')

    def _draw_base_map(self):
        """Draw grey network using pre-computed edge data. Called once per search."""
        self._ax.clear()
        sites = self._sites

        for lons, lats in self._edge_segments:
            self._ax.plot(lons, lats, color='#cccccc', linewidth=1, zorder=1)

        lons = [info['lon'] for info in sites.values()]
        lats = [info['lat'] for info in sites.values()]
        self._ax.scatter(lons, lats, color='#457b9d', s=30, zorder=3)

        for scats, info in sites.items():
            self._ax.annotate(
                str(scats), (info['lon'], info['lat']),
                fontsize=5.5, ha='center', va='bottom',
                xytext=(0, 4), textcoords='offset points', color='#1d3557'
            )

        self._ax.set_xlabel('Longitude', fontsize=8)
        self._ax.set_ylabel('Latitude', fontsize=8)
        self._ax.set_title('Boroondara Road Network', fontsize=10)
        self._ax.tick_params(labelsize=7)
        self._fig.tight_layout()
        self._canvas.draw()

    def _draw_routes(self, highlight_routes: list, colour_offset: int = 0):
        sites = self._sites

        for artist in self._route_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self._route_artists = []

        for idx, path in enumerate(highlight_routes):
            colour = ROUTE_COLOURS[(colour_offset + idx) % len(ROUTE_COLOURS)]
            if len(path) > 1:
                for i in range(len(path) - 1):
                    a, b = path[i], path[i + 1]
                    lon1, lat1 = sites[a]['lon'], sites[a]['lat']
                    lon2, lat2 = sites[b]['lon'], sites[b]['lat']
                    line, = self._ax.plot(
                        [lon1, lon2], [lat1, lat2],
                        color=colour, linewidth=2.5, zorder=2,
                        label=f'Route {idx+1}' if i == 0 else '_nolegend_'
                    )
                    self._route_artists.append(line)

        first_path = highlight_routes[0]
        origin, dest = first_path[0], first_path[-1]
        for scats, colour, label in [
            (origin, '#2a9d8f', 'Origin'),
            (dest,   '#e76f51', 'Destination'),
        ]:
            lon, lat = sites[scats]['lon'], sites[scats]['lat']
            sc = self._ax.scatter(lon, lat, color=colour, s=80, zorder=4, label=label)
            self._route_artists.append(sc)

        self._ax.legend(fontsize=7, loc='upper left')
        self._canvas.draw_idle()

    def _show_route_with_refresh(self, path: list):
        if self._current_path == path:
            return
        self._current_path = path
        idx = next((i for i, r in enumerate(self._current_routes) if r['path'] == path), 0)
        for artist in self._route_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self._route_artists = []
        self._canvas.draw_idle()
        self.after(120, lambda: self._draw_routes([path], colour_offset=idx))

    def _show_all_routes(self):
        if not self._current_routes:
            return
        self._current_path = None
        for artist in self._route_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self._route_artists = []
        self._canvas.draw_idle()
        self.after(120, lambda: self._draw_routes([r['path'] for r in self._current_routes]))

    def _parse_scats(self, label: str):
        try:
            return int(label.split('–')[0].strip())
        except (ValueError, IndexError):
            return None

    def _on_find(self):
        origin = self._parse_scats(self._origin_var.get())
        dest   = self._parse_scats(self._dest_var.get())

        if origin is None or dest is None:
            messagebox.showwarning('Input required', 'Please select both an origin and a destination.')
            return
        if origin == dest:
            messagebox.showwarning('Invalid input', 'Origin and destination must be different.')
            return

        try:
            selected_dt = datetime.datetime(
                int(self._date_year_cb.get()),
                int(self._date_month_cb.get()),
                int(self._date_day_cb.get()),
                int(self._time_hour_cb.get()),
                int(self._time_min_cb.get()),
            )
        except ValueError:
            messagebox.showwarning('Invalid date/time', 'Please select a valid date and time.')
            return

        algo_code  = ALGORITHMS[self._algo_var.get()]
        model_name = self._model_var.get()

        self._find_btn.configure(state=tk.DISABLED)
        self._status_var.set('Searching…')
        self._clear_results()
        self._current_path = None

        threading.Thread(
            target=self._run_search,
            args=(origin, dest, algo_code, selected_dt, model_name),
            daemon=True
        ).start()

    def _run_search(self, origin: int, dest: int, algorithm: str, dt: datetime.datetime, model_name: str):
        try:
            flow_dict = self._predict(dt, model_name=model_name) if self._predict else None
            routes = find_routes(origin, dest, flow_dict=flow_dict, algorithm=algorithm)
            self.after(0, self._display_results, routes, algorithm)
        except Exception as e:
            self.after(0, self._show_error, str(e))

    def _display_results(self, routes: list, algorithm: str):
        self._find_btn.configure(state=tk.NORMAL)
        self._clear_results()

        if not routes:
            self._status_var.set('No route found.')
            return

        self._status_var.set(f'{len(routes)} route(s) found via {algorithm}.')
        self._current_routes = routes
        self._show_all_btn.pack(side=tk.RIGHT)
        self._draw_base_map()
        self._draw_routes([r['path'] for r in routes])

        for i, route in enumerate(routes):
            path    = route['path']
            minutes = route['cost_minutes']
            sites   = route['sites']
            colour  = ROUTE_COLOURS[i % len(ROUTE_COLOURS)]

            frame = ttk.LabelFrame(self._results_frame, text=f'Route {i+1}  –  {minutes:.1f} min', padding=6)
            frame.pack(fill=tk.X, pady=4)
            frame.bind('<Enter>', self._bind_scroll)
            frame.bind('<Leave>', self._unbind_scroll)

            indicator = tk.Frame(frame, background=colour, width=10)
            indicator.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 6))

            info_frame = ttk.Frame(frame)
            info_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            path_str = ' → '.join(str(s) for s in path)
            ttk.Label(info_frame, text=path_str, wraplength=260, justify=tk.LEFT, font=('Courier', 8)).pack(anchor=tk.W)

            via = [sites[s]['loc'] for s in path[1:-1]]
            if via:
                via_str = 'Via: ' + ', '.join(via[:4]) + ('…' if len(via) > 4 else '')
                ttk.Label(info_frame, text=via_str, foreground='grey', wraplength=260, font=('TkDefaultFont', 8)).pack(anchor=tk.W)

            ttk.Button(
                info_frame, text='Show this route only',
                command=lambda p=path: self._show_route_with_refresh(p)
            ).pack(anchor=tk.E, pady=(4, 0))

    def _clear_results(self):
        for widget in self._results_frame.winfo_children():
            widget.destroy()
        self._show_all_btn.pack_forget()
        self._route_artists = []
        self._current_path = None
        self._current_routes = []

    def _show_error(self, message: str):
        self._find_btn.configure(state=tk.NORMAL)
        self._status_var.set('Error.')
        messagebox.showerror('Search error', message)


if __name__ == '__main__':
    app = App()
    app.mainloop()
