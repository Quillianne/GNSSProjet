import tkinter as tk
from tkinter import ttk
import serial
import pynmea2
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import matplotlib.patches as patches

# Serial port configuration
serial_port = 'COM3'
baud_rate = 4800

# Lists to hold latitude and longitude values
latitudes = []
longitudes = []

# Function to read NMEA sentences from the serial port
def read_nmea():
    try:
        with serial.Serial(serial_port, baud_rate, timeout=1) as ser:
            while True:
                line = ser.readline().decode('ascii', errors='replace')
                if line.startswith('$'):
                    try:
                        msg = pynmea2.parse(line)
                        print(msg)
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
        gga_vars['hdop'].set(f"{getattr(msg, 'horizontal_dil', 'N/A')} ({get_dop_quality(getattr(msg, 'horizontal_dil', 'N/A'))})")
        gga_vars['altitude'].set(f"{getattr(msg, 'altitude', 'N/A')} {getattr(msg, 'altitude_units', '')}")
        gga_vars['geo_sep'].set(f"{getattr(msg, 'geo_sep', 'N/A')} {getattr(msg, 'geo_sep_units', '')}")

        if lat and lon:
            latitudes.append(float(lat))
            longitudes.append(float(lon))
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
        gsa_vars['pdop'].set(f"{getattr(msg, 'pdop', 'N/A')} ({get_dop_quality(getattr(msg, 'pdop', 'N/A'))})")
        gsa_vars['hdop'].set(f"{getattr(msg, 'hdop', 'N/A')} ({get_dop_quality(getattr(msg, 'hdop', 'N/A'))})")
        gsa_vars['vdop'].set(f"{getattr(msg, 'vdop', 'N/A')} ({get_dop_quality(getattr(msg, 'vdop', 'N/A'))})")
    elif isinstance(msg, pynmea2.types.talker.GSV):
        num_sv = msg.data[2]  # Number of satellites in view
        sv_prns = ', '.join([msg.data[i] for i in range(4, len(msg.data), 4)])  # Satellite PRNs
        gsv_vars['num_sv'].set(num_sv)
        gsv_vars['sv_prns'].set(sv_prns)

# Function to update the plot
def update_plot():
    if len(latitudes) > 1:
        lat_mean = np.mean(latitudes)
        lon_mean = np.mean(longitudes)
        lat_std = np.std(latitudes)
        lon_std = np.std(longitudes)
        
        # Conversion factors
        meters_per_degree_lat = 111139  # Approximate meters per degree latitude
        meters_per_degree_lon = 111139 * np.cos(np.radians(lat_mean))  # Approximate meters per degree longitude at the mean latitude
        
        # Convert standard deviation to meters
        lat_std_meters = lat_std * meters_per_degree_lat
        lon_std_meters = lon_std * meters_per_degree_lon

        # Calculate global standard deviation
        global_std = np.sqrt(lat_std_meters**2 + lon_std_meters**2)

        ax.clear()
        ax.scatter(longitudes, latitudes, label='Positions')
        ax.scatter(lon_mean, lat_mean, color='red', label='Position Moyenne')
        
        # Draw circle representing global standard deviation
        circle = patches.Circle((lon_mean, lat_mean), global_std / meters_per_degree_lat,
                                edgecolor='red', facecolor='none', label='Cercle Écart-Type Global')
        ax.add_patch(circle)
        
        # Mark the last position in green
        ax.scatter(longitudes[-1], latitudes[-1], color='green', label='Dernière Position')
        
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Graphique de Position')
        ax.legend()
        
        # Ensure the same scale on both axes
        ax.set_aspect('equal', adjustable='datalim')
        
        plot_canvas.draw()
        
        std_var.set(f"Écart-Type: {lat_std_meters:.2f} m (Latitude), {lon_std_meters:.2f} m (Longitude), {global_std:.2f} m (Global)\n"
                    f"Moyenne: {lat_mean:.6f}° (Latitude), {lon_mean:.6f}° (Longitude)")

# Function to start the thread for reading NMEA sentences
def start_reading():
    thread = threading.Thread(target=read_nmea, daemon=True)
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

# Plotting area
fig, ax = plt.subplots(figsize=(5, 4))
plot_canvas = FigureCanvasTkAgg(fig, master=root)
plot_canvas.get_tk_widget().grid(column=0, row=len(gga_vars) + len(rmc_vars) + len(gsa_vars) + len(gsv_vars) + 4, columnspan=3)

ttk.Label(root, textvariable=std_var).grid(column=0, row=len(gga_vars) + len(rmc_vars) + len(gsa_vars) + len(gsv_vars) + 5, columnspan=3)

# Start the NMEA reading in a separate thread
start_reading()

# Start the GUI main loop
root.mainloop()
