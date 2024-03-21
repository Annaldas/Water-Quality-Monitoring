import pickle

import ee
import joblib
import streamlit as st
import folium
from streamlit_folium import folium_static
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Authenticate to the Earth Engine servers.
ee.Authenticate()
ee.Initialize(project='ee-team2wqm')

model = joblib.load('wqm_logr.pkl')
# Function to get water quality parameters from Google Earth Engine API
def get_water_quality_data(latitude, longitude):
    point = ee.Geometry.Point(longitude, latitude)
    image = ee.ImageCollection('COPERNICUS/S2_SR').filterBounds(point).first()
    ndci = image.normalizedDifference(['B5', 'B4']).rename('ndci')
    latlon = ee.Image.pixelLonLat().addBands(ndci)
    latlon = latlon.reduceRegion(reducer=ee.Reducer.toList(), geometry=point, scale=100)
    data_ndci = ee.Array(latlon.get("ndci")).getInfo()

    # dissolved oxygen
    dissolvedoxygen = image.expression('(-0.0167 * B8 + 0.0067 * B9 + 0.0083 * B11) + 9.577', {
        'B8': image.select('B8'),
        'B9': image.select('B9'),
        'B11': image.select('B11')
    }).rename('dissolvedoxygen')
    latlon = ee.Image.pixelLonLat().addBands(dissolvedoxygen)
    latlon = latlon.reduceRegion(reducer=ee.Reducer.toList(), geometry=point, scale=100)

    data_do = ee.Array(latlon.get("dissolvedoxygen")).getInfo()

    # NDTI
    band11 = image.select('B11')
    band12 = image.select('B12')
    ndti = band11.subtract(band12).divide(band11.add(band12)).rename('NDTI')
    latlon = ee.Image.pixelLonLat().addBands(ndti)
    latlon = latlon.reduceRegion(reducer=ee.Reducer.toList(), geometry=point, scale=100)
    data_ndti = ee.Array(latlon.get("NDTI")).getInfo()

    # NDSI
    green_band = image.select('B3')
    swir_band = image.select('B11')
    ndsi = green_band.subtract(swir_band).divide(green_band.add(swir_band)).rename('NDSI')
    latlon = ee.Image.pixelLonLat().addBands(ndsi)
    latlon = latlon.reduceRegion(reducer=ee.Reducer.toList(), geometry=point, scale=100)
    data_ndsi = ee.Array(latlon.get("NDSI")).getInfo()

    # pH
    band7 = image.select('B1')
    band8 = image.select('B8')
    pH = band7.divide(band8).multiply(-0.827).add(8.339).rename('pH')
    latlon = ee.Image.pixelLonLat().addBands(pH)
    latlon = latlon.reduceRegion(reducer=ee.Reducer.toList(), geometry=point, scale=100)
    data_pH = ee.Array(latlon.get("pH")).getInfo()

    # this is for landsat dataset.

    landsat_img = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').filterBounds(point).first()
    tir_band = landsat_img.select('ST_B10')
    lst = tir_band.multiply(0.00341802).add(149.0).subtract(273.15).rename('LST')
    latlon = ee.Image.pixelLonLat().addBands(lst)
    latlon = latlon.reduceRegion(reducer=ee.Reducer.toList(), geometry=point, scale=100)
    data_LST = ee.Array(latlon.get("LST")).getInfo()

    # sentinel3 dataset

    # DOM
    sentinel3_img = ee.ImageCollection("COPERNICUS/S3/OLCI").filterBounds(point).first()
    oa08_radiance = sentinel3_img.select('Oa08_radiance')
    oa04_radiance = sentinel3_img.select('Oa04_radiance')
    # Calculate dissolved organic matter using the provided formula
    dom = oa08_radiance.divide(oa04_radiance).rename('DOM')
    latlon = ee.Image.pixelLonLat().addBands(dom)
    latlon = latlon.reduceRegion(reducer=ee.Reducer.toList(), geometry=point, scale=100)
    # print(latlon.get("DOM").getInfo())
    data_dom = latlon.get("DOM").getInfo()

    # SM
    oa08_radiance = sentinel3_img.select('Oa08_radiance')
    oa06_radiance = sentinel3_img.select('Oa06_radiance')
    # Calculate dissolved organic matter using the provided formula
    sm = oa08_radiance.divide(oa06_radiance).rename('SM')
    latlon = ee.Image.pixelLonLat().addBands(sm)
    latlon = latlon.reduceRegion(reducer=ee.Reducer.toList(), geometry=point, scale=100)
    # print(latlon.get("DOM").getInfo())
    data_sm = latlon.get("SM").getInfo()

    return {"Dissolved Oxygen": data_do[0],
            "Salinity": data_ndsi[0],
            "Temperature": data_LST[0],
            "pH": data_pH[0],
            "Turbidity": data_ndti[0],
            "Dissolved Organic Matter": data_dom[0] * 100,
            "Suspended Matter": data_sm[0] * 100,
            "Chlorophyll": data_ndci[0]}


# Streamlit UI
background_css = """
    <style>
        body {
            background-image: url("bg_image.jpg");
            background-size: cover;
        }
    </style>
"""
st.markdown(background_css, unsafe_allow_html=True)
st.title('Water Quality Monitoring')

# Input for latitude and longitude
latitude = st.number_input('Enter Latitude:')
longitude = st.number_input('Enter Longitude:')

if st.button('Get Water Quality Parameters'):
    # Get water quality parameters
    water_quality_data = get_water_quality_data(latitude, longitude)

    # Display retrieved parameters
    st.write('Chlorophyll:', water_quality_data['Chlorophyll'])
    st.write('pH:', water_quality_data['pH'])
    st.write('Temperature: ', water_quality_data['Temperature'])
    st.write('Turbidity: ', water_quality_data['Turbidity'])
    st.write('Dissolved Organic Matter: ', water_quality_data['Dissolved Organic Matter'])
    st.write('Suspended Matter: ', water_quality_data['Suspended Matter'])
    st.write('Salinity: ', water_quality_data['Salinity'])
    st.write('Dissolved Oxygen: ', water_quality_data['Dissolved Oxygen'])

    data = list(water_quality_data.values())
    data_array = np.array(data)
    # Reshape the array to (1, -1) to make it 2D with one row
    data_2d = data_array.reshape(1, -1)
    # Make prediction using the trained model
    prediction = model.predict(data_2d)

    # Display prediction
    res = "Good"
    if prediction == 1:
        res = "Needs Treatment"
    elif prediction == 2:
        res = "Poor"
    else:
        pass
    st.write('Predicted Water Quality:', res)
    m = folium.Map(location=[latitude, longitude], zoom_start=25)

    # Add a marker for the selected point
    folium.Marker(
        location=[latitude, longitude],
        popup='Selected Point',
        icon=folium.Icon(color='blue')
    ).add_to(m)

    # Display the map
    folium_static(m)