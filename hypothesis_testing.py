from scipy.stats import norm, gamma, f, chi2
import matplotlib.pyplot as plt
import numpy as np
import folium

def create_hypothesis_test_map(before, after, aoi, m, threshold):
    # Set the decision threshold 
    dt = f.ppf(threshold, 2*m, 2*m)

    # LRT statistics.
    q1 = before.divide(after)
    q2 = after.divide(before)

    # Change map with 0 = no change, 1 = decrease, 2 = increase in intensity.
    c_map = before.multiply(0).where(q2.lt(dt), 1)
    c_map = c_map.where(q1.lt(dt), 2)

    # Mask no-change pixels.
    c_map = c_map.updateMask(c_map.gt(0))

    # Display map with red for increase and blue for decrease in intensity.
    location = aoi.centroid().coordinates().getInfo()[::-1]
    mp = folium.Map(
        location=location, tiles='Stamen Toner',
        zoom_start=11)
    folium.TileLayer('OpenStreetMap').add_to(mp)
    mp.add_ee_layer(q2,
                    {'min': 0, 'max': 2, 'palette': ['black', 'white']}, 'Ratio')
    mp.add_ee_layer(c_map,
                    {'min': 0, 'max': 2, 'palette': ['black', 'blue', 'red']},
                    'Change Map')
    mp.add_child(folium.LayerControl())

    return mp