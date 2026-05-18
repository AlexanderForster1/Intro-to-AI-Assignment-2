import math

SPEED_LIMIT_KMH = 60.0
CAPACITY_FLOW_VEH_HR = 1500.0
CAPACITY_SPEED_KMH = 32.0
INTERSECTION_DELAY_S = 30.0

QUADRATIC_A = -CAPACITY_FLOW_VEH_HR / (CAPACITY_SPEED_KMH ** 2)
QUADRATIC_B = -2 * CAPACITY_SPEED_KMH * QUADRATIC_A
 
FLOW_THRESHOLD_VEH_HR = QUADRATIC_A * SPEED_LIMIT_KMH ** 2 + QUADRATIC_B * SPEED_LIMIT_KMH

def flow_to_speed(flow_veh_hr: float) -> float:
    """Convert predicted flow (veh/hr) to speed (km/h) using the free-flow branch."""
    if flow_veh_hr < 0:
        flow_veh_hr = 0.0
 
    if flow_veh_hr <= FLOW_THRESHOLD_VEH_HR:
        return SPEED_LIMIT_KMH
 
    discriminant = QUADRATIC_B ** 2 + 4 * QUADRATIC_A * flow_veh_hr
    if discriminant < 0:
        return CAPACITY_SPEED_KMH
 
    # QUADRATIC_A is negative, so the -sqrt root gives the higher (free-flow) speed
    speed = (-QUADRATIC_B - math.sqrt(discriminant)) / (2 * QUADRATIC_A)
    return max(CAPACITY_SPEED_KMH, min(speed, SPEED_LIMIT_KMH))

def travel_time_seconds(
    distance_km: float,
    flow_veh_hr: float,
    include_intersection_delay: bool = True,
) -> float:
    """Return estimated travel time in seconds for a segment given distance and flow."""
    speed = flow_to_speed(flow_veh_hr)
    time_s = (distance_km / speed) * 3600.0
    if include_intersection_delay:
        time_s += INTERSECTION_DELAY_S
    return time_s

def travel_time_minutes(
    distance_km: float,
    flow_veh_hr: float,
    include_intersection_delay: bool = True,
) -> float:
    """Return estimated travel time in minutes for a segment given distance and flow."""
    return travel_time_seconds(distance_km, flow_veh_hr, include_intersection_delay) / 60.0