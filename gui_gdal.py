import tkinter as tk
from tkinter import ttk
import serial
import pynmea2
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import matplotlib.patches as patches
from osgeo import gdal
import pyproj
import random
import time
import webbrowser
from mpl_toolkits.mplot3d import Axes3D


# Set up the transformation from WGS84 to Lambert93
wgs84 = pyproj.CRS("EPSG:4326")
lambert93 = pyproj.CRS("EPSG:2154")
transformer_to_lambert93 = pyproj.Transformer.from_crs(wgs84, lambert93)
transformer_to_wgs84 = pyproj.Transformer.from_crs(lambert93, wgs84)

# Function to open the link in a web browser
def open_link(url):
    webbrowser.open_new(url)

def generate_random_gpgga():
    base_lat = 48.418303
    base_lon = -4.472370

    def generate_coordinate(base, offset):
        return base + random.uniform(-offset, offset)

    def format_lat_lon(value, is_latitude=True):
        degrees = int(value)
        minutes = (value - degrees) * 60
        direction = 'N' if degrees >= 0 else 'S' if is_latitude else 'E' if degrees >= 0 else 'W'
        return f"{abs(degrees):02d}{abs(minutes):07.4f}", direction

    def calculate_checksum(nmea_str):
        checksum = 0
        for char in nmea_str:
            checksum ^= ord(char)
        return f"{checksum:02X}"

    def generate_gpgga_sentence():
        lat, lat_dir = format_lat_lon(generate_coordinate(base_lat, 0.00001), is_latitude=True)
        lon, lon_dir = format_lat_lon(generate_coordinate(base_lon, 0.00001), is_latitude=False)
        time_str = time.strftime("%H%M%S.00", time.gmtime())
        gpgga = f"GPGGA,{time_str},{lat},{lat_dir},{lon},{lon_dir},1,08,0.9,545.4,M,46.9,M,,"
        checksum = calculate_checksum(gpgga)
        return f"${gpgga}*{checksum}"

    return generate_gpgga_sentence()

def fake_feed():
    while True:
        sentence = generate_random_gpgga()
        print(sentence)
        try:
            msg = pynmea2.parse(sentence)
            update_interface(msg)
        except pynmea2.nmea.ParseError as e:
            print(f"Parse error: {e}")
        except AttributeError as e:
            print(f"Attribute error: {e}")
        time.sleep(1)

# Serial port configuration
serial_port = 'COM4'
baud_rate = 4800

# Lists to hold latitude and longitude values
latitudes = []
longitudes = []
x_coords = []
y_coords = []

# Variables to hold satellite data
satellite_data = []

# Variables to hold partial GSV messages
gsv_messages = []
expected_gsv_messages = 0

# Function to read NMEA sentences from the serial port
def read_nmea():
    try:
        with serial.Serial(serial_port, baud_rate, timeout=1) as ser:
            while True:
                line = ser.readline().decode('ascii', errors='replace')
                if line.startswith('$'):
                    try:
                        msg = pynmea2.parse(line)
                        #print(msg)
                        update_interface(msg)
                    except pynmea2.nmea.ParseError as e:
                        print(f"Parse error: {e}")
                    except AttributeError as e:
                        print(f"Attribute error: {e}")
    except serial.SerialException as e:
        print(f"Error: {e}")

# Function to format latitude and longitude to 6 digits
def format_lat_lon(value, direction):
    if value:
        return f"{float(value):.6f} {direction}"
    return "N/A"

# Function to determine the quality of DOP values
def get_dop_quality(dop):
    if dop != '':
        dop = float(dop)
        if dop <= 1:
            return "Idéal"
        elif dop <= 2:
            return "Excellent"
        elif dop <= 5:
            return "Bon"
        elif dop <= 10:
            return "Modéré"
        else:
            return "Mauvais"
    else:
        return None

# Function to update the interface with the parsed NMEA data
def update_interface(msg):
    if isinstance(msg, pynmea2.types.talker.GGA):
        gga_vars['timestamp'].set(getattr(msg, 'timestamp', 'N/A'))
        lat = getattr(msg, 'latitude', '')
        lat_dir = getattr(msg, 'lat_dir', '')
        lon = getattr(msg, 'longitude', '')
        lon_dir = getattr(msg, 'lon_dir', '')
        gga_vars['latitude'].set(format_lat_lon(lat, lat_dir))
        gga_vars['longitude'].set(format_lat_lon(lon, lon_dir))
        gga_vars['fix_quality'].set(getattr(msg, 'gps_qual', 'N/A'))
        gga_vars['num_sats'].set(getattr(msg, 'num_sats', 'N/A'))
        hdop = getattr(msg, 'horizontal_dil', 'N/A')
        gga_vars['hdop'].set(f"{hdop} ({get_dop_quality(hdop)})" if hdop != 'N/A' else 'N/A')
        gga_vars['altitude'].set(f"{getattr(msg, 'altitude', 'N/A')} {getattr(msg, 'altitude_units', '')}")
        gga_vars['geo_sep'].set(f"{getattr(msg, 'geo_sep', 'N/A')} {getattr(msg, 'geo_sep_units', '')}")

        if lat and lon:
            latitudes.append(float(lat))
            longitudes.append(float(lon))
            x, y = transformer_to_lambert93.transform(lat, lon)
            x_coords.append(x)
            y_coords.append(y)
            update_plot()

    elif isinstance(msg, pynmea2.types.talker.RMC):
        rmc_vars['timestamp'].set(getattr(msg, 'timestamp', 'N/A'))
        rmc_vars['status'].set(getattr(msg, 'status', 'N/A'))
        rmc_vars['latitude'].set(format_lat_lon(getattr(msg, 'latitude', ''), getattr(msg, 'lat_dir', '')))
        rmc_vars['longitude'].set(format_lat_lon(getattr(msg, 'longitude', ''), getattr(msg, 'lon_dir', '')))
        rmc_vars['speed'].set(getattr(msg, 'spd_over_grnd', 'N/A'))
        rmc_vars['course'].set(getattr(msg, 'true_course', 'N/A'))
        rmc_vars['datestamp'].set(getattr(msg, 'datestamp', 'N/A'))
        rmc_vars['mode'].set(getattr(msg, 'mode_indicator', 'N/A'))
    elif isinstance(msg, pynmea2.types.talker.GSA):
        gsa_vars['mode'].set(getattr(msg, 'mode', 'N/A'))
        gsa_vars['fix_type'].set(getattr(msg, 'mode_fix_type', 'N/A'))
        pdop = getattr(msg, 'pdop', 'N/A')
        hdop = getattr(msg, 'hdop', 'N/A')
        vdop = getattr(msg, 'vdop', 'N/A')
        gsa_vars['pdop'].set(f"{pdop} ({get_dop_quality(pdop)})" if pdop != 'N/A' else 'N/A')
        gsa_vars['hdop'].set(f"{hdop} ({get_dop_quality(hdop)})" if hdop != 'N/A' else 'N/A')
        gsa_vars['vdop'].set(f"{vdop} ({get_dop_quality(vdop)})" if vdop != 'N/A' else 'N/A')
    elif isinstance(msg, pynmea2.types.talker.GSV):
        handle_gsv_message(msg)

def handle_gsv_message(msg):
    global gsv_messages, expected_gsv_messages
    
    # Append the message to the list of GSV messages
    gsv_messages.append(msg)

    # If this is the first message in the sequence, set the expected number of messages
    if int(msg.msg_num) == 1:
        print("--- GSV ---")
        expected_gsv_messages = int(msg.num_messages)
    
    print(f"({len(gsv_messages)}/{expected_gsv_messages}) | {msg}") 

    # If we have collected all the messages in the sequence, process them
    if len(gsv_messages) == expected_gsv_messages:
        # Clear the previous satellite data
        satellite_data.clear()
        
        # Collect satellite data from all GSV messages
        for gsv_msg in gsv_messages:
            for i in range(3, len(gsv_msg.data), 4):
                prn = gsv_msg.data[i]
                elevation = gsv_msg.data[i + 1]
                azimuth = gsv_msg.data[i + 2]
                snr = gsv_msg.data[i + 3] if i + 3 < len(gsv_msg.data) else 'N/A'
                satellite_data.append((prn, elevation, azimuth, snr))
        
        # Update the GSV variables if num_sv is present
        if hasattr(gsv_messages[0], 'num_sv'):
            gsv_vars['num_sv'].set(gsv_messages[0].num_sv)
        sv_prns = ', '.join([sat[0] for sat in satellite_data])
        gsv_vars['sv_prns'].set(sv_prns)
        
        # Update the satellite plot
        update_satellite_plot()
        
        # Clear the GSV messages for the next sequence
        gsv_messages = []

        # Print the list of PRNs for debugging purposes
        #print("PRNs of visible satellites:", [sat[0] for sat in satellite_data])



# Load the background image using GDAL and convert its coordinates
im = gdal.Open('ensta_2015.jpg')

geotransform = im.GetGeoTransform()

# Get image dimensions
nx = im.RasterXSize
ny = im.RasterYSize
nb = im.RasterCount

# Initialize the image array
image = np.zeros((ny, nx, nb), dtype=np.float32)

# Read each band into the image array and normalize to [0, 1]
image[:, :, 0] = im.GetRasterBand(1).ReadAsArray() / 255.0
image[:, :, 1] = im.GetRasterBand(2).ReadAsArray() / 255.0
image[:, :, 2] = im.GetRasterBand(3).ReadAsArray() / 255.0

# Example conversion of image corners from Lambert93 to WGS84
# Assuming the image covers a certain area in Lambert93 coordinates
top_left_x = geotransform[0]
top_left_y = geotransform[3]
bottom_right_x = top_left_x + nx * geotransform[1] + ny * geotransform[2]
bottom_right_y = top_left_y + nx * geotransform[4] + ny * geotransform[5]

top_left_lat, top_left_lon = transformer_to_wgs84.transform(top_left_x, top_left_y)
bottom_right_lat, bottom_right_lon = transformer_to_wgs84.transform(bottom_right_x, bottom_right_y)

print(top_left_lat, top_left_lon, bottom_right_lat, bottom_right_lon)

# Update the plot function to use the converted coordinates
def update_plot():
    if len(latitudes) > 1:
        lat_mean = np.mean(latitudes)
        lon_mean = np.mean(longitudes)

        # Convert latitude and longitude to Lambert93
        coords = np.array([transformer_to_lambert93.transform(lat, lon) for lat, lon in zip(latitudes, longitudes)])
        x_coords, y_coords = coords[:, 0], coords[:, 1]

        # Compute the covariance matrix
        cov = np.cov(x_coords, y_coords)
        
        # Compute the eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Sort the eigenvalues and eigenvectors
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        # Calculate the angle of the ellipse
        angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))

        # Standard deviation corresponds to the square root of eigenvalues
        std_dev_x, std_dev_y = np.sqrt(eigenvalues)

        ax.clear()

        # Plot the background image
        ax.imshow(image, extent=[top_left_x, bottom_right_x, bottom_right_y, top_left_y], aspect='auto')

        ax.scatter(x_coords, y_coords, label='Positions')
        x_mean, y_mean = transformer_to_lambert93.transform(lat_mean, lon_mean)
        ax.scatter(x_mean, y_mean, color='red', label='Position Moyenne')

        # Draw ellipse representing global standard deviation
        ellipse = patches.Ellipse((x_mean, y_mean), width=2*std_dev_x, height=2*std_dev_y, angle=angle, 
                                  edgecolor='red', facecolor='none', label='Ellipse Écart-Type Global')
        ax.add_patch(ellipse)

        # Mark the last position in green
        ax.scatter(x_coords[-1], y_coords[-1], color='green', label='Dernière Position')

        # Determine the limits of the plot
        min_x = min(x_coords)
        max_x = max(x_coords)
        min_y = min(y_coords)
        max_y = max(y_coords)

        zoom = 20

        ax.set_xlim(min_x - zoom, max_x + zoom)
        ax.set_ylim(min_y - zoom, max_y + zoom)

        ax.set_xlabel('X (Lambert93)')
        ax.set_ylabel('Y (Lambert93)')
        ax.set_title('Graphique de Position')
        ax.legend()

        # Ensure the same scale on both axes
        ax.set_aspect('equal', adjustable='datalim')

        plot_canvas.draw()

        std_var.set(f"Écart-Type: {std_dev_x:.2f} m (X), {std_dev_y:.2f} m (Y), {np.sqrt(std_dev_y**2+std_dev_x**2):.2f} m (global)\n"
                    f"Moyenne: {lat_mean:.6f}° (Latitude), {lon_mean:.6f}° (Longitude)")

        # Update the link label
        link_label.config(text=f"Open mean coordinates in google")
        link_label.bind("<Button-1>", lambda x: open_link(f"https://maps.google.com?q={lat_mean:.6f},{lon_mean:.6f}"))



def update_satellite_plot():
    ax_sat.clear()
    
    for sat in satellite_data:
        prn, elevation, azimuth, snr = sat
        if elevation and azimuth:
            elevation = float(elevation)
            azimuth = float(azimuth)
            ax_sat.scatter(np.radians(azimuth), 90 - elevation, label=f"PRN: {prn}, SNR: {snr}")
    
    ax_sat.set_theta_zero_location('N')  # 0° au nord
    ax_sat.set_theta_direction(-1)  # Sens horaire
    
    ax_sat.set_ylim(0, 90)
    ax_sat.set_yticks(range(0, 91, 15))
    ax_sat.set_yticklabels(map(str, range(90, -1, -15)))
    ax_sat.set_title("Trajectoire des Satellites")
    
    # Supprimer ou masquer la légende
    # ax_sat.legend(loc='upper right')
    ax_sat.legend().set_visible(False)
    
    satellite_canvas.draw()


# Function to start the thread for reading NMEA sentences
def start_reading():
    thread = threading.Thread(target=read_nmea, daemon=True)
    #thread = threading.Thread(target=fake_feed, daemon=True)
    thread.start()

# Setting up the main application window
root = tk.Tk()
root.title("Interface de Données NMEA")

# Close the application when the window is closed
root.protocol("WM_DELETE_WINDOW", root.quit)

# Variables to hold the data
gga_vars = {key: tk.StringVar() for key in ['timestamp', 'latitude', 'longitude', 'fix_quality', 'num_sats', 'hdop', 'altitude', 'geo_sep']}
rmc_vars = {key: tk.StringVar() for key in ['timestamp', 'status', 'latitude', 'longitude', 'speed', 'course', 'datestamp', 'mode']}
gsa_vars = {key: tk.StringVar() for key in ['mode', 'fix_type', 'pdop', 'hdop', 'vdop']}
gsv_vars = {key: tk.StringVar() for key in ['num_sv', 'sv_prns']}
std_var = tk.StringVar()

# Creating and placing the labels and values on the interface
explanations = {
    'timestamp': 'Heure UTC de la fixation',
    'latitude': 'Latitude de la position',
    'longitude': 'Longitude de la position',
    'fix_quality': 'Indicateur de qualité GPS',
    'num_sats': 'Nombre de satellites utilisés',
    'hdop': 'Dilution horizontale de la précision',
    'altitude': 'Altitude au-dessus du niveau moyen de la mer',
    'geo_sep': 'Séparation géoïde',
    'status': 'Statut des données (A=valide, V=non valide)',
    'speed': 'Vitesse par rapport au sol en nœuds',
    'course': 'Cap par rapport au sol en degrés',
    'datestamp': 'Date de la fixation',
    'mode': 'Indicateur de mode',
    'fix_type': 'Type de fixation (2D/3D)',
    'pdop': 'Dilution de la précision de position',
    'vdop': 'Dilution verticale de la précision',
    'num_sv': 'Nombre de satellites en vue',
    'sv_prns': 'PRNs des satellites en vue'
}

bold_font = ('TkDefaultFont', 10, 'bold')

ttk.Label(root, text="Données GGA", font=bold_font).grid(column=0, row=0, columnspan=3)
for idx, (key, var) in enumerate(gga_vars.items()):
    ttk.Label(root, text=key.replace('_', ' ').title()).grid(column=0, row=idx+1, sticky=tk.W)
    ttk.Label(root, textvariable=var).grid(column=1, row=idx+1, sticky=tk.W)
    ttk.Label(root, text=explanations[key]).grid(column=2, row=idx+1, sticky=tk.W)

ttk.Label(root, text="Données RMC", font=bold_font).grid(column=0, row=len(gga_vars) + 1, columnspan=3)
for idx, (key, var) in enumerate(rmc_vars.items()):
    ttk.Label(root, text=key.replace('_', ' ').title()).grid(column=0, row=len(gga_vars) + 2 + idx, sticky=tk.W)
    ttk.Label(root, textvariable=var).grid(column=1, row=len(gga_vars) + 2 + idx, sticky=tk.W)
    ttk.Label(root, text=explanations[key]).grid(column=2, row=len(gga_vars) + 2 + idx, sticky=tk.W)

ttk.Label(root, text="Données GSA", font=bold_font).grid(column=0, row=len(gga_vars) + len(rmc_vars) + 2, columnspan=3)
for idx, (key, var) in enumerate(gsa_vars.items()):
    ttk.Label(root, text=key.replace('_', ' ').title()).grid(column=0, row=len(gga_vars) + len(rmc_vars) + 3 + idx, sticky=tk.W)
    ttk.Label(root, textvariable=var).grid(column=1, row=len(gga_vars) + len(rmc_vars) + 3 + idx, sticky=tk.W)
    ttk.Label(root, text=explanations[key]).grid(column=2, row=len(gga_vars) + len(rmc_vars) + 3 + idx, sticky=tk.W)

ttk.Label(root, text="Données GSV", font=bold_font).grid(column=0, row=len(gga_vars) + len(rmc_vars) + len(gsa_vars) + 3, columnspan=3)
for idx, (key, var) in enumerate(gsv_vars.items()):
    ttk.Label(root, text=key.replace('_', ' ').title()).grid(column=0, row=len(gga_vars) + len(rmc_vars) + len(gsa_vars) + 4 + idx, sticky=tk.W)
    ttk.Label(root, textvariable=var).grid(column=1, row=len(gga_vars) + len(rmc_vars) + len(gsa_vars) + 4 + idx, sticky=tk.W)
    ttk.Label(root, text=explanations[key]).grid(column=2, row=len(gga_vars) + len(rmc_vars) + len(gsa_vars) + 4 + idx, sticky=tk.W)

# Plotting area for positions and satellite trajectories
fig, ax = plt.subplots(figsize=(5, 4))
plot_canvas = FigureCanvasTkAgg(fig, master=root)
plot_canvas.get_tk_widget().grid(column=0, row=len(gga_vars) + len(rmc_vars) + len(gsa_vars) + len(gsv_vars) + 4, columnspan=3)

fig_sat, ax_sat = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(5, 4))
satellite_canvas = FigureCanvasTkAgg(fig_sat, master=root)
satellite_canvas.get_tk_widget().grid(column=3, row=0, rowspan=len(gga_vars) + len(rmc_vars) + len(gsa_vars) + len(gsv_vars) + 4)

ttk.Label(root, textvariable=std_var).grid(column=0, row=len(gga_vars) + len(rmc_vars) + len(gsa_vars) + len(gsv_vars) + 5, columnspan=3)

link_label = tk.Label(root, text="", fg="blue", cursor="hand2")
link_label.grid(column=0, row=len(gga_vars) + len(rmc_vars) + len(gsa_vars) + len(gsv_vars) + 6, columnspan=3)

# Start the NMEA reading in a separate thread
start_reading()

# Start the GUI main loop
root.mainloop()
