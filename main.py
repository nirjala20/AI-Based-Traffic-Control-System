import streamlit as st
import time
from controller import TrafficSignalController
import asyncio
import torch
import nest_asyncio
import folium
from streamlit_folium import st_folium
import json
import pandas as pd
import requests
from typing import Tuple, Optional

# Fix for asyncio runtime error
try:
    nest_asyncio.apply()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    if torch.cuda.is_available():
        torch.cuda.init()
except Exception as e:
    st.error(f"Initialization error: {str(e)}")

# Initialize Session State
if 'current_direction_index' not in st.session_state:
    st.session_state.current_direction_index = 0
if 'remaining_time' not in st.session_state:
    st.session_state.remaining_time = 0
if 'signal_data' not in st.session_state:
    st.session_state.signal_data = None
if 'cycle_completed' not in st.session_state:
    st.session_state.cycle_completed = False
if 'controller' not in st.session_state:
    st.session_state.controller = TrafficSignalController(model_name="yolov8m")
if 'auto_restart' not in st.session_state:
    st.session_state.auto_restart = False
if 'page' not in st.session_state:
    st.session_state.page = None
if 'intersection_selected' not in st.session_state:
    st.session_state.intersection_selected = False
if 'intersection_confirmed' not in st.session_state:
    st.session_state.intersection_confirmed = False
if 'intersection_coords' not in st.session_state:
    st.session_state.intersection_coords = None
if 'map_center' not in st.session_state:
    st.session_state.map_center = [16.7050, 74.2433]  # Coordinates for Kolhapur city center

def verify_intersection(lat: float, lng: float) -> Tuple[bool, Optional[str], Optional[Tuple[float, float]]]:
    """Verify if coordinates represent a 4-way intersection using Overpass API"""
    overpass_url = "http://overpass-api.de/api/interpreter"
    query = f"""
    [out:json][timeout:25];
    (
        way(around:100,{lat},{lng})[highway];
        node(around:100,{lat},{lng})[highway=traffic_signals];
    );
    out body;
    >;
    out skel qt;
    """
    
    try:
        response = requests.post(overpass_url, data=query)
        data = response.json()
        
        # Find nearest intersection node
        intersections = []
        roads = [elem for elem in data['elements'] if elem['type'] == 'way']
        
        if len(roads) >= 4:
            # Find nodes where multiple roads meet
            node_counts = {}
            for road in roads:
                for node in road.get('nodes', []):
                    node_counts[node] = node_counts.get(node, 0) + 1
            
            # Get coordinates of potential intersections
            for elem in data['elements']:
                if elem['type'] == 'node':
                    if node_counts.get(elem['id'], 0) >= 3:  # Node where 3 or more roads meet
                        distance = calculate_distance(lat, lng, elem['lat'], elem['lon'])
                        if distance <= 0.1:  # Within 100 meters
                            intersections.append((elem['lat'], elem['lon'], distance))
            
            if intersections:
                # Get the nearest intersection
                nearest = min(intersections, key=lambda x: x[2])
                return True, None, (nearest[0], nearest[1])
            
        return False, "No 4-way intersection found nearby", None
    except Exception as e:
        return False, f"Error verifying intersection: {str(e)}", None

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points in kilometers using Haversine formula"""
    from math import radians, sin, cos, sqrt, atan2
    
    R = 6371  # Earth's radius in kilometers
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c
    
    return distance

def create_map():
    """Create and display the map for intersection selection"""
    st.markdown('<p class="subtitle-text">Select a 4-way intersection on the map</p>', unsafe_allow_html=True)
    
    m = folium.Map(
        location=st.session_state.map_center,
        zoom_start=13,
        tiles="CartoDB positron",
        control_scale=True
    )
    
    # Add zoom control
    m.add_child(folium.LatLngPopup())
    
    map_data = st_folium(
        m,
        height=500,
        width="100%",
        returned_objects=["last_clicked"]
    )
    
    if (map_data is not None and 
        "last_clicked" in map_data and 
        map_data["last_clicked"] is not None):
        
        lat = map_data["last_clicked"]["lat"]
        lng = map_data["last_clicked"]["lng"]
        
        is_intersection, error_msg, intersection_coords = verify_intersection(lat, lng)
        
        if is_intersection and intersection_coords:
            st.session_state.intersection_coords = intersection_coords
            st.session_state.map_center = intersection_coords
            st.session_state.intersection_confirmed = True
            st.session_state.intersection_selected = True
            st.rerun()
        else:
            st.warning(error_msg)

def main():
    st.set_page_config(
        page_title="AI Traffic Controller",
        layout="wide",
        page_icon="🚦",
        initial_sidebar_state="expanded"
    )

    st.markdown('<p class="header-text">AI Traffic Signal Control</p>', unsafe_allow_html=True)

    if st.session_state.page is None:
        st.markdown('<p class="subtitle-text">Choose your preferred operation mode</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Select Map Mode", key="map_mode", use_container_width=True):
                st.session_state.page = "map"
                st.rerun()

        with col2:
            if st.button("Select Direct Mode", key="direct_mode", use_container_width=True):
                st.session_state.page = "direct"
                st.rerun()
        return

    if st.session_state.page == "map":
        run_map_mode()
    else:
        run_direct_mode()

def run_map_mode():
    if not st.session_state.intersection_confirmed:
        create_map()
    else:
        show_traffic_control_interface(True)

def run_direct_mode():
    show_traffic_control_interface(False)

def show_traffic_control_interface(show_map=False):
    st.markdown("""
        <p class="subtitle-text">
            <strong>Real-time traffic management</strong> using YOLOv8 vehicle detection.<br>
            The system automatically adjusts signal timings based on detected vehicle density.
        </p>
    """, unsafe_allow_html=True)

    if show_map and not st.session_state.get('signal_data'):
        run_detection(st.session_state.controller)
    else:
        if st.button("▶️ Start New Detection Cycle", use_container_width=True):
            st.session_state.auto_restart = False
            run_detection(st.session_state.controller)

    if st.session_state.signal_data:
        show_current_signal_state()
        countdown_and_cycle_signals(st.session_state.signal_data['timings'])

def run_detection(controller):
    with st.status("🚦 **Processing Traffic Data**", expanded=True):
        counts, timings, images = controller.run_control_cycle()
        st.session_state.signal_data = {
            'counts': counts,
            'timings': timings,
            'images': images
        }

        st.session_state.current_direction_index = 0
        st.session_state.remaining_time = timings['Direction_1']

def show_current_signal_state():
    pass

def countdown_and_cycle_signals(timings):
    pass

if __name__ == "__main__":
    main()
    
