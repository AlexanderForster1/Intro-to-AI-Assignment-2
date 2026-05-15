import tkinter as tk
from tkinter import ttk, messagebox
import threading
import json
import os

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


def load_config() -> dict:
    with open(CONFIG_PATH, 'r') as f:
        return json.load(f)


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

        self._build_ui()
        self._draw_base_map()

    def _build_ui(self):
        left = ttk.Frame(self, padding=10)
        left.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(left, text='Origin SCATS').pack(anchor=tk.W)
        self._origin_var = tk.StringVar()
        self._origin_cb = ttk.Combobox(
            left, textvariable=self._origin_var,
            values=self._scats_labels, width=42, state='readonly'
        )
        self._origin_cb.pack(pady=(0, 8))

        ttk.Label(left, text='Destination SCATS').pack(anchor=tk.W)
        self._dest_var = tk.StringVar()
        self._dest_cb = ttk.Combobox(
            left, textvariable=self._dest_var,
            values=self._scats_labels, width=42, state='readonly'
        )
        self._dest_cb.pack(pady=(0, 8))

        ttk.Label(left, text='Algorithm').pack(anchor=tk.W)
        self._algo_var = tk.StringVar(value=list(ALGORITHMS.keys())[0])
        self._algo_cb = ttk.Combobox(
            left, textvariable=self._algo_var,
            values=list(ALGORITHMS.keys()), width=42, state='readonly'
        )
        self._algo_cb.pack(pady=(0, 12))
        self._algo_cb.bind('<<ComboboxSelected>>', self._on_algo_changed)

        self._routes_note = ttk.Label(left, text='Returns up to 5 routes', foreground='grey', font=('TkDefaultFont', 8))
        self._routes_note.pack(anchor=tk.W, pady=(0, 8))

        self._find_btn = ttk.Button(left, text='Find Routes', command=self._on_find)
        self._find_btn.pack(fill=tk.X, pady=(0, 12))

        ttk.Separator(left, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)
        ttk.Label(left, text='Results').pack(anchor=tk.W)

        self._results_frame = ttk.Frame(left)
        self._results_frame.pack(fill=tk.BOTH, expand=True)

        self._status_var = tk.StringVar(value='Ready.')
        ttk.Label(left, textvariable=self._status_var, foreground='grey').pack(
            anchor=tk.W, pady=(8, 0)
        )

        right = ttk.Frame(self, padding=10)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self._fig, self._ax = plt.subplots(figsize=(7, 6))
        self._canvas = FigureCanvasTkAgg(self._fig, master=right)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _on_algo_changed(self, _event=None):
        algo_key = self._algo_var.get()
        algo_code = ALGORITHMS.get(algo_key, 'AS')
        if algo_code in MULTI_ROUTE_ALGORITHMS:
            self._routes_note.config(text='Returns up to 5 routes')
            self._find_btn.config(text='Find Routes')
        else:
            self._routes_note.config(text='Returns 1 route')
            self._find_btn.config(text='Find Route')

    def _draw_base_map(self, highlight_path: list[int] | None = None):
        self._ax.clear()
        node_list = self._node_list
        sites = self._sites

        drawn = set()
        for scats, node in node_list.items():
            for nb, _ in node.edges:
                pair = (min(scats, nb), max(scats, nb))
                if pair in drawn:
                    continue
                drawn.add(pair)
                lon1, lat1 = sites[scats]['lon'], sites[scats]['lat']
                lon2, lat2 = sites[nb]['lon'], sites[nb]['lat']
                self._ax.plot([lon1, lon2], [lat1, lat2], color='#cccccc', linewidth=1, zorder=1)

        if highlight_path and len(highlight_path) > 1:
            for i in range(len(highlight_path) - 1):
                a, b = highlight_path[i], highlight_path[i + 1]
                lon1, lat1 = sites[a]['lon'], sites[a]['lat']
                lon2, lat2 = sites[b]['lon'], sites[b]['lat']
                self._ax.plot([lon1, lon2], [lat1, lat2], color='#e63946', linewidth=2.5, zorder=2)

        for scats, info in sites.items():
            self._ax.scatter(info['lon'], info['lat'], color='#457b9d', s=30, zorder=3)
            self._ax.annotate(
                str(scats), (info['lon'], info['lat']),
                fontsize=5.5, ha='center', va='bottom',
                xytext=(0, 4), textcoords='offset points', color='#1d3557'
            )

        if highlight_path:
            origin, dest = highlight_path[0], highlight_path[-1]
            for scats, colour, label in [
                (origin, '#2a9d8f', 'Origin'),
                (dest,   '#e76f51', 'Destination'),
            ]:
                lon, lat = sites[scats]['lon'], sites[scats]['lat']
                self._ax.scatter(lon, lat, color=colour, s=80, zorder=4, label=label)
            self._ax.legend(fontsize=7, loc='upper left')

        self._ax.set_xlabel('Longitude', fontsize=8)
        self._ax.set_ylabel('Latitude', fontsize=8)
        self._ax.set_title('Boroondara Road Network', fontsize=10)
        self._ax.tick_params(labelsize=7)
        self._fig.tight_layout()
        self._canvas.draw()

    def _parse_scats(self, label: str) -> int | None:
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

        algo_code = ALGORITHMS[self._algo_var.get()]

        self._find_btn.configure(state=tk.DISABLED)
        self._status_var.set('Searching…')
        self._clear_results()

        threading.Thread(
            target=self._run_search,
            args=(origin, dest, algo_code),
            daemon=True
        ).start()

    def _run_search(self, origin: int, dest: int, algorithm: str):
        try:
            routes = find_routes(origin, dest, algorithm=algorithm)
            self.after(0, self._display_results, routes, algorithm)
        except Exception as e:
            self.after(0, self._show_error, str(e))

    def _display_results(self, routes: list[dict], algorithm: str):
        self._find_btn.configure(state=tk.NORMAL)
        self._clear_results()

        if not routes:
            self._status_var.set('No route found.')
            return

        self._status_var.set(f'{len(routes)} route(s) found via {algorithm}.')

        for i, route in enumerate(routes):
            path    = route['path']
            minutes = route['cost_minutes']
            sites   = route['sites']

            frame = ttk.LabelFrame(
                self._results_frame,
                text=f'Route {i + 1}  –  {minutes:.1f} min',
                padding=6
            )
            frame.pack(fill=tk.X, pady=4)

            path_str = ' → '.join(str(s) for s in path)
            ttk.Label(frame, text=path_str, wraplength=320, justify=tk.LEFT, font=('Courier', 8)).pack(anchor=tk.W)

            via = [sites[s]['loc'] for s in path[1:-1]]
            if via:
                via_str = 'Via: ' + ', '.join(via[:4]) + ('…' if len(via) > 4 else '')
                ttk.Label(frame, text=via_str, foreground='grey', wraplength=320, font=('TkDefaultFont', 8)).pack(anchor=tk.W)

            btn = ttk.Button(
                frame, text='Show on map',
                command=lambda p=path: self._draw_base_map(highlight_path=p)
            )
            btn.pack(anchor=tk.E, pady=(4, 0))

        self._draw_base_map(highlight_path=routes[0]['path'])

    def _clear_results(self):
        for widget in self._results_frame.winfo_children():
            widget.destroy()

    def _show_error(self, message: str):
        self._find_btn.configure(state=tk.NORMAL)
        self._status_var.set('Error.')
        messagebox.showerror('Search error', message)


if __name__ == '__main__':
    app = App()
    app.mainloop()