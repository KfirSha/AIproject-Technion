import pandas as pd
import folium
import os
import sys
from folium.plugins import HeatMap

# Read map data from CSV
map_data = pd.read_csv("world.csv")

# Find highest purchase amount
max_amount = float(map_data['s'].max())

# Makes a new map centered on the given location and zoom
startingLocation = [30, 35] # EDIT THIS WITH YOUR CITIES GPS COORDINATES!
hmap = folium.Map(location=startingLocation, zoom_start=7)

# Creates a heatmap element
hm_wide = HeatMap( list(zip(map_data.lat.values, map_data.lon.values, map_data.s.values)),
                    min_opacity=0.3,
                    max_val=max_amount,
                    radius=20, blur=15,
                    max_zoom=1)

# Adds the heatmap element to the map
hmap.add_child(hm_wide)

# Saves the map to heatmap.hmtl
hmap.save(os.path.join('.', 'heatmap.html'))