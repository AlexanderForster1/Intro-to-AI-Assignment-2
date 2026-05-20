import csv
import math 
import re 
import os
from collections import defaultdict

DEFAULT_CSV = os.path.join(os.path.dirname(__file__), "data", "map_data.csv")

class Node:
    """Represents a SCATS intersection node in the road network."""
    def __init__(self, name: int, coordinates: tuple):
        self.name = name
        self.coordinates = coordinates  
        self.edges: list[tuple[int, float]] = []
 
    def add_edge(self, neighbour: int, cost: float) -> None:
        self.edges.append((neighbour, cost))
 
    def has_edge(self, neighbour: int) -> bool:
        return neighbour in (e[0] for e in self.edges)
 
    def __repr__(self):
        return f"Node({self.name}, coords={self.coordinates}, edges={self.edges})"

# helpers
def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance in kilometres between two lat/lon points."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)

    return R * 2 * math.asin(math.sqrt(a))

def _parse_primary_road(location: str) -> str:
    """Extract the primary road name from a SCATS location string."""
    parts = re.split(
        r'\s+(?:NE|NW|SE|SW|N|S|E|W)\s+(?:of|OF)\b',
        location,
        maxsplit=1,
        flags=re.IGNORECASE,
    )
    return parts[0].strip()
 
def _parse_cross_road(location: str) -> str:
    """Extract the cross road name from a SCATS location string."""
    parts = re.split(
        r'\s+(?:NE|NW|SE|SW|N|S|E|W)\s+(?:of|OF)\b',
        location,
        maxsplit=1,
        flags=re.IGNORECASE,
    )
    return parts[1].strip() if len(parts) > 1 else ''

def _load_sites(csv_path: str) -> dict[int, dict]:
    """Load SCATS site metadata from map_data.csv.
    Returns a dict keyed by SCATS number:"""
    sites = {}
    with open(csv_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            scats = int(row['SCATS Number'])
            lat = float(row['NB_LATITUDE'])
            lon = float(row['NB_LONGITUDE'])
            if lat == 0.0 and lon == 0.0:
                continue
            loc = row['Location'].strip()
            sites[scats] = {
                'lat'  : lat,
                'lon'  : lon,
                'loc'  : loc,
                'road' : _parse_primary_road(loc),
                'cross': _parse_cross_road(loc),
            }
    return sites

# Edge constructions 
def _add_edge(node_list: dict[int, Node], a: int, b: int, dist_km: float) -> None:
    """Add a bidirectional edge between nodes a and b with cost dist_km."""
    if not node_list[a].has_edge(b):
        node_list[a].add_edge(b, dist_km)
    if not node_list[b].has_edge(a):
        node_list[b].add_edge(a, dist_km)

def _build_same_road_edges(
    sites: dict[int, dict],
    node_list: dict[int, Node],
    ) -> None:
    """Connect sites that share the same primary road name."""
    road_groups: dict[str, list[int]] = defaultdict(list)
    for scats, info in sites.items():
        road_groups[info['road']].append(scats)
 
    for road, members in road_groups.items():
        if len(members) < 2:
            continue

        lats = [sites[s]['lat'] for s in members]
        lons = [sites[s]['lon'] for s in members]
        lat_spread = max(lats) - min(lats)
        lon_spread = max(lons) - min(lons)
 
        if lon_spread >= lat_spread:
            members.sort(key=lambda s: sites[s]['lon'])
        else:
            members.sort(key=lambda s: sites[s]['lat'])
 
        for i in range(len(members) - 1):
            a, b = members[i], members[i + 1]
            dist = _haversine_km(
                sites[a]['lat'], sites[a]['lon'],
                sites[b]['lat'], sites[b]['lon'],
            )
            _add_edge(node_list, a, b, dist)

def _build_cross_road_edges(
    sites: dict[int, dict],
    node_list: dict[int, Node],
    max_dist_km: float = 2.5,) -> None:
    """connect sites that meet at a common interesection via t heir cross-road."""

    scats_list = list(sites.keys())
 
    for i, a in enumerate(scats_list):
        for b in scats_list[i + 1:]:
            road_a = sites[a]['road']
            road_b = sites[b]['road']
            cross_a = sites[a]['cross']
            cross_b = sites[b]['cross']
 
            connected = (
                road_a == cross_b   # A's road is B's cross road
                or road_b == cross_a  # B's road is A's cross road
            )
 
            if connected:
                dist = _haversine_km(
                    sites[a]['lat'], sites[a]['lon'],
                    sites[b]['lat'], sites[b]['lon'],
                )
                if dist <= max_dist_km:
                    _add_edge(node_list, a, b, dist)

def _build_manual_edges(
    sites: dict[int, dict],
    node_list: dict[int, Node],) -> None:
    """
    Add edges that cannot be inferred automatically from road/cross-road name
    matching.  These cover two situations:
 
      1. Sites that share a road corridor but use a different naming convention
      2. Outlier sites with no shared road tokens but clearly on the same.
         corridor by geographic evidence and the Boroondara map.
 
    Pairs listed here has been cross-checked against the PDF map and
    haversine distances, all are under 2.5 km.
    """
    manual_pairs: list[tuple[int, int]] = [
        # Western Burwood Rd corridor
        (4262, 4263),   
        (4263, 4264), 
        # Outer-west sites connecting to network 
        (4262, 4812),  
        (4812, 4263),   
        (4262, 4821),   
        # 3002 DENMARK_ST connects to BARKERS_RD cluster 
        (3002, 3001),  
        (3002, 4263),   
        # 2200 UNION_RD connects via WHITEHORSE_RD / BALWYN_RD 
        (2200, 4063),  
        # 2846 S.E.ARTERIAL connects near TOORAK_RD 
        (2846, 4043),   
        ]

    for a, b in manual_pairs:
        if a not in node_list or b not in node_list:
            continue
        dist = _haversine_km(
            sites[a]['lat'], sites[a]['lon'],
            sites[b]['lat'], sites[b]['lon'],
        )
        _add_edge(node_list, a, b, dist)

def _remove_long_same_road_edges(
    node_list: dict[int, Node],
    sites: dict[int: dict],
    threshold_km: float = 2.5) -> None:
    """Remove same-road edges that exceed the threshold."""
    for node in node_list.values():
        node.edges = [
            (nb, cost) for nb, cost in node.edges if cost <= threshold_km
        ]

def _sort_edges(node_list: dict[int, Node]) -> None:
    """Sort each node's edge list by neighbour number ascending"""
    for node in node_list.values():
        node.edges.sort(key=lambda e: e[0])

# Public API 
def build_graph(
    csv_path: str = DEFAULT_CSV,
    cross_road_max_dist_km: float = 2.5,
) -> tuple[dict[int, Node], dict[int, dict]]:
    """Build the Boroondara road network graph from map_data.csv"""
    sites = _load_sites(csv_path)

    node_list: dict[int, Node] = {}
    for scats, info in sites.items():
        node_list[scats] = Node(
            name=scats,
            coordinates=(info['lon'], info['lat']),
        )
 
    _build_same_road_edges(sites, node_list)
    _build_cross_road_edges(sites, node_list, max_dist_km=cross_road_max_dist_km)
    _build_manual_edges(sites, node_list)
    _remove_long_same_road_edges(node_list, sites, threshold_km=cross_road_max_dist_km)
    _sort_edges(node_list)
 
    return node_list, sites
