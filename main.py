import tkinter as tk
from tkinter import ttk, filedialog
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
import serial.tools.list_ports
import webbrowser
from mpl_toolkits.mplot3d import Axes3D

class NMEAGUI:
    def __init__(self, root, mode="statique", data=(False, "export.txt"), gnss_types=None):
        self.mode = mode
        self.data = data
        self.root = root
        self.gnss_types = gnss_types if gnss_types else ['GP', 'GA']

        # Set up the transformation from WGS84 to Lambert93
        self.wgs84 = pyproj.CRS("EPSG:4326")
        self.lambert93 = pyproj.CRS("EPSG:2154")
        self.transformer_to_lambert93 = pyproj.Transformer.from_crs(self.wgs84, self.lambert93)
        self.transformer_to_wgs84 = pyproj.Transformer.from_crs(self.lambert93, self.wgs84)

        # Serial port configuration
        self.baud_rates = [115200, 4800]

        # Lists to hold latitude and longitude values
        self.altitudes = []
        self.latitudes = []
        self.longitudes = []
        self.x_coords = []
        self.y_coords = []

        # Variables to hold satellite data
        self.satellite_data = {}

        # Variables to hold partial GSV messages
        self.gsv_messages = {}
        self.expected_gsv_messages = {}

        # Load the background image using GDAL and convert its coordinates
        self.im = gdal.Open('ensta_2015.jpg')

        self.geotransform = self.im.GetGeoTransform()

        # Get image dimensions
        self.nx = self.im.RasterXSize
        self.ny = self.im.RasterYSize
        self.nb = self.im.RasterCount

        # Initialize the image array
        self.image = np.zeros((self.ny, self.nx, self.nb), dtype=np.float32)

        # Read each band into the image array and normalize to [0, 1]
        self.image[:, :, 0] = self.im.GetRasterBand(1).ReadAsArray() / 255.0
        self.image[:, :, 1] = self.im.GetRasterBand(2).ReadAsArray() / 255.0
        self.image[:, :, 2] = self.im.GetRasterBand(3).ReadAsArray() / 255.0

        # Example conversion of image corners from Lambert93 to WGS84
        self.top_left_x = self.geotransform[0]
        self.top_left_y = self.geotransform[3]
        self.bottom_right_x = self.top_left_x + self.nx * self.geotransform[1] + self.ny * self.geotransform[2]
        self.bottom_right_y = self.top_left_y + self.nx * self.geotransform[4] + self.ny * self.geotransform[5]

        self.top_left_lat, self.top_left_lon = self.transformer_to_wgs84.transform(self.top_left_x, self.top_left_y)
        self.bottom_right_lat, self.bottom_right_lon = self.transformer_to_wgs84.transform(self.bottom_right_x, self.bottom_right_y)

        self.constellation_colors = {
            'GP': 'blue',   # GPS
            'GL': 'green',  # GLONASS
            'GA': 'red',    # Galileo
            'GB': 'orange', # BeiDou
            'GI': 'purple', # IRNSS
            'GS': 'cyan',   # SBAS
        }

        self.setup_ui()

    # Function to open the link in a web browser
    def open_link(self, url):
        webbrowser.open_new(url)

    # Function to read NMEA sentences from the serial port
    def read_nmea(self):
        serial_port = self.data[1]
        for baud_rate in self.baud_rates:
            print(f"ON ESSAYE UN BAUDRATE DE {baud_rate}")
            try:
                with serial.Serial(serial_port, baud_rate, timeout=1) as ser:
                    buffer = ""
                    no_data_count = 0
                    while True:
                        data = ser.read(1).decode('ascii', errors='replace')
                        if data:
                            buffer += data
                            if data == '\n':
                                line = buffer.strip()
                                buffer = ""
                                if any(line.startswith(f'${gnss}') for gnss in self.gnss_types):
                                    try:
                                        msg = pynmea2.parse(line)
                                        print(msg)
                                        self.update_interface(msg)
                                        no_data_count = 0  # Reset count if valid data is received
                                    except pynmea2.nmea.ParseError as e:
                                        print(f"Parse error: {e}")
                                    except AttributeError as e:
                                        print(f"Attribute error: {e}")
                            else:
                                no_data_count += 1
                                if no_data_count > 120:  # Adjust the threshold as needed
                                    print("NO DATA")
                                    break

            except serial.SerialException as e:
                print(f"Error with baud rate {baud_rate}: {e}")

    def read_nmea_from_file(self):
        filename = self.data[1]
        try:
            with open(filename, 'r') as file:
                while True:
                    line = file.readline()
                    if not line:
                        break
                    if any(line.startswith(f'${gnss}') for gnss in self.gnss_types):
                        try:
                            msg = pynmea2.parse(line)
                            print(msg)
                            self.update_interface(msg)
                        except pynmea2.nmea.ParseError as e:
                            print(f"Parse error: {e}")
                        except AttributeError as e:
                            print(f"Attribute error: {e}")
                    time.sleep(0.0001)  # Sleep for 0.1 milliseconds
        except IOError as e:
            print(f"File error: {e}")

    # Function to format latitude and longitude to 6 digits
    def format_lat_lon(self, value, direction):
        if value:
            return f"{float(value):.6f} {direction}"
        return "N/A"

    # Function to determine the quality of DOP values
    def get_dop_quality(self, dop):
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
    def update_interface(self, msg):
        if isinstance(msg, pynmea2.types.talker.GGA):
            self.gga_vars['timestamp'].set(getattr(msg, 'timestamp', 'N/A'))
            lat = getattr(msg, 'latitude', '')
            lat_dir = getattr(msg, 'lat_dir', '')
            lon = getattr(msg, 'longitude', '')
            lon_dir = getattr(msg, 'lon_dir', '')
            alt = getattr(msg, 'altitude', 'N/A')
            self.gga_vars['latitude'].set(self.format_lat_lon(lat, lat_dir))
            self.gga_vars['longitude'].set(self.format_lat_lon(lon, lon_dir))
            self.gga_vars['fix_quality'].set(getattr(msg, 'gps_qual', 'N/A'))
            self.gga_vars['num_sats'].set(getattr(msg, 'num_sats', 'N/A'))
            hdop = getattr(msg, 'horizontal_dil', 'N/A')
            self.gga_vars['hdop'].set(f"{hdop} ({self.get_dop_quality(hdop)})" if hdop != 'N/A' else 'N/A')
            self.gga_vars['altitude'].set(f"{alt} {getattr(msg, 'altitude_units', '')}")
            self.gga_vars['geo_sep'].set(f"{getattr(msg, 'geo_sep', 'N/A')} {getattr(msg, 'geo_sep_units', '')}")

            if lat and lon:
                self.latitudes.append(float(lat))
                self.longitudes.append(float(lon))
                self.altitudes.append(float(alt))
                x, y = self.transformer_to_lambert93.transform(lat, lon)
                self.x_coords.append(x)
                self.y_coords.append(y)
                self.update_plot()

        elif isinstance(msg, pynmea2.types.talker.RMC):
            self.rmc_vars['timestamp'].set(getattr(msg, 'timestamp', 'N/A'))
            self.rmc_vars['status'].set(getattr(msg, 'status', 'N/A'))
            self.rmc_vars['latitude'].set(self.format_lat_lon(getattr(msg, 'latitude', ''), getattr(msg, 'lat_dir', '')))
            self.rmc_vars['longitude'].set(self.format_lat_lon(getattr(msg, 'longitude', ''), getattr(msg, 'lon_dir', '')))
            self.rmc_vars['speed'].set(getattr(msg, 'spd_over_grnd', 'N/A'))
            self.rmc_vars['course'].set(getattr(msg, 'true_course', 'N/A'))
            self.rmc_vars['datestamp'].set(getattr(msg, 'datestamp', 'N/A'))
            self.rmc_vars['mode'].set(getattr(msg, 'mode_indicator', 'N/A'))
        elif isinstance(msg, pynmea2.types.talker.GSA):
            self.gsa_vars['mode'].set(getattr(msg, 'mode', 'N/A'))
            self.gsa_vars['fix_type'].set(getattr(msg, 'mode_fix_type', 'N/A'))
            pdop = getattr(msg, 'pdop', 'N/A')
            hdop = getattr(msg, 'hdop', 'N/A')
            vdop = getattr(msg, 'vdop', 'N/A')
            self.gsa_vars['pdop'].set(f"{pdop} ({self.get_dop_quality(pdop)})" if pdop != 'N/A' else 'N/A')
            self.gsa_vars['hdop'].set(f"{hdop} ({self.get_dop_quality(hdop)})" if hdop != 'N/A' else 'N/A')
            self.gsa_vars['vdop'].set(f"{vdop} ({self.get_dop_quality(vdop)})" if vdop != 'N/A' else 'N/A')
        elif isinstance(msg, pynmea2.types.talker.GSV):
            self.handle_gsv_message(msg)

    def handle_gsv_message(self, msg):
        constellation_type = msg.talker  # Assuming msg.talker provides 'GP', 'GL', etc.

        if constellation_type not in self.gsv_messages:
            self.gsv_messages[constellation_type] = []
            self.expected_gsv_messages[constellation_type] = int(msg.num_messages)

        self.gsv_messages[constellation_type].append(msg)

        if len(self.gsv_messages[constellation_type]) == self.expected_gsv_messages[constellation_type]:
            self.satellite_data[constellation_type] = []

            for gsv_msg in self.gsv_messages[constellation_type]:
                # Parcours dynamique des satellites dans le message
                for i in range(1, 4):  # Il y a jusqu'à 4 satellites par message GSV
                    prn = getattr(gsv_msg, f'sv_prn_num_{i}', None)
                    elevation = getattr(gsv_msg, f'elevation_deg_{i}', None)
                    azimuth = getattr(gsv_msg, f'azimuth_{i}', None)
                    snr = getattr(gsv_msg, f'snr_{i}', 'N/A')

                    # Ajouter les données du satellite si prn, elevation et azimuth sont valides
                    if prn and elevation and azimuth:
                        self.satellite_data[constellation_type].append((prn, elevation, azimuth, snr))

            self.gsv_messages[constellation_type] = []

            # Mettre à jour le graphique chaque fois qu'une constellation complète son cycle
            self.update_satellite_plot(self.satellite_data)

        # Mettre à jour les variables gsv
        self.gsv_vars['num_sv'].set(len([sat[0] for constellation in self.satellite_data.values() for sat in constellation]))
        sv_prns = ', '.join([sat[0] for constellation in self.satellite_data.values() for sat in constellation])
        self.gsv_vars['sv_prns'].set(sv_prns[:50])

    def update_satellite_plot(self, sat_data):
        self.ax_sat.clear()

        for constellation, sats in sat_data.items():
            color = self.constellation_colors.get(constellation, 'black')  # Default to black if not found

            for sat in sats:
                prn, elevation, azimuth, snr = sat
                if elevation and azimuth:
                    elevation = float(elevation)
                    azimuth = float(azimuth)

                    self.ax_sat.scatter(np.radians(azimuth), 90 - elevation, label=f"{constellation} PRN: {prn}, SNR: {snr}", color=color)

        self.ax_sat.set_theta_zero_location('N')  # 0° au nord
        self.ax_sat.set_theta_direction(-1)  # Sens horaire

        self.ax_sat.set_ylim(0, 90)
        self.ax_sat.set_yticks(range(0, 91, 15))
        self.ax_sat.set_yticklabels(map(str, range(90, -1, -15)))
        self.ax_sat.set_title("Trajectoire des Satellites")

        # Supprimer ou masquer la légende si nécessaire
        handles, labels = self.ax_sat.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax_sat.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(-0.7, 0.5), loc='center left', ncol=1)

        self.satellite_canvas.draw()

    # Update the plot function to use the converted coordinates
    def update_plot(self):
        if len(self.latitudes) > 1:
            self.ax.clear()

            # Convert latitude and longitude to Lambert93
            coords = np.array([self.transformer_to_lambert93.transform(lat, lon) for lat, lon in zip(self.latitudes, self.longitudes)])
            self.x_coords, self.y_coords = list(coords[:, 0]), list(coords[:, 1])

            # Plot the background image
            self.ax.imshow(self.image, extent=[self.top_left_x, self.bottom_right_x, self.bottom_right_y, self.top_left_y], aspect='auto')

            self.ax.scatter(self.x_coords, self.y_coords, label='Positions')

            if self.mode == "statique":
                lat_mean = np.mean(self.latitudes)
                lon_mean = np.mean(self.longitudes)

                # Compute the covariance matrix
                cov = np.cov(self.x_coords, self.y_coords)

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

                x_mean, y_mean = self.transformer_to_lambert93.transform(lat_mean, lon_mean)
                print("LAMBERT: ", x_mean, y_mean)
                self.ax.scatter(x_mean, y_mean, color='red', label='Position Moyenne')

                # Draw ellipse representing global standard deviation
                ellipse = patches.Ellipse((x_mean, y_mean), width=2*std_dev_x, height=2*std_dev_y, angle=angle,
                                          edgecolor='red', facecolor='none', label='Ellipse Écart-Type Global')
                self.ax.add_patch(ellipse)

                self.std_var.set(f"Écart-Type: {std_dev_x:.2f} m (X), {std_dev_y:.2f} m (Y), {np.sqrt(std_dev_y**2 + std_dev_x**2):.2f} m (global)\n"
                                 f"Moyenne: {lat_mean:.6f}° (Latitude), {lon_mean:.6f}° (Longitude)")

                # Update the link label
                self.link_label.config(text=f"Open mean coordinates in google")
                self.link_label.bind("<Button-1>", lambda x: self.open_link(f"https://maps.google.com?q={lat_mean:.6f},{lon_mean:.6f}"))

            # Mark the last position in green
            self.ax.scatter(self.x_coords[-1], self.y_coords[-1], color='green', label='Dernière Position')

            # Determine the limits of the plot
            min_x = min(self.x_coords)
            max_x = max(self.x_coords)
            min_y = min(self.y_coords)
            max_y = max(self.y_coords)

            zoom = 20

            self.ax.set_xlim(min_x - zoom, max_x + zoom)
            self.ax.set_ylim(min_y - zoom, max_y + zoom)

            self.ax.set_xlabel('X (Lambert93)')
            self.ax.set_ylabel('Y (Lambert93)')
            self.ax.set_title('Graphique de Position')
            self.ax.legend()

            # Ensure the same scale on both axes
            self.ax.set_aspect('equal', adjustable='datalim')

            self.plot_canvas.draw()
        else:
            print("Pas assez de données pour afficher le graphique.")

    # Function to start the thread for reading NMEA sentences
    def start_reading(self):
        if self.data[0] == False:
            thread = threading.Thread(target=lambda: self.read_nmea_from_file(), daemon=True)
        else:
            thread = threading.Thread(target=lambda: self.read_nmea(), daemon=True)
        thread.start()

    def setup_ui(self):
        # Setting up the main application window
        self.root.title("Interface de Données NMEA")

        # Close the application when the window is closed
        self.root.protocol("WM_DELETE_WINDOW", self.root.quit)

        # Variables to hold the data
        self.gga_vars = {key: tk.StringVar() for key in ['timestamp', 'latitude', 'longitude', 'fix_quality', 'num_sats', 'hdop', 'altitude', 'geo_sep']}
        self.rmc_vars = {key: tk.StringVar() for key in ['timestamp', 'status', 'latitude', 'longitude', 'speed', 'course', 'datestamp', 'mode']}
        self.gsa_vars = {key: tk.StringVar() for key in ['mode', 'fix_type', 'pdop', 'hdop', 'vdop']}
        self.gsv_vars = {key: tk.StringVar() for key in ['num_sv', 'sv_prns']}
        self.std_var = tk.StringVar()

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

        ttk.Label(self.root, text="Données GGA", font=bold_font).grid(column=0, row=0, columnspan=3)
        for idx, (key, var) in enumerate(self.gga_vars.items()):
            ttk.Label(self.root, text=key.replace('_', ' ').title()).grid(column=0, row=idx + 1, sticky=tk.W)
            ttk.Label(self.root, textvariable=var).grid(column=1, row=idx + 1, sticky=tk.W)
            ttk.Label(self.root, text=explanations[key]).grid(column=2, row=idx + 1, sticky=tk.W)

        ttk.Label(self.root, text="Données RMC", font=bold_font).grid(column=0, row=len(self.gga_vars) + 1, columnspan=3)
        for idx, (key, var) in enumerate(self.rmc_vars.items()):
            ttk.Label(self.root, text=key.replace('_', ' ').title()).grid(column=0, row=len(self.gga_vars) + 2 + idx, sticky=tk.W)
            ttk.Label(self.root, textvariable=var).grid(column=1, row=len(self.gga_vars) + 2 + idx, sticky=tk.W)
            ttk.Label(self.root, text=explanations[key]).grid(column=2, row=len(self.gga_vars) + 2 + idx, sticky=tk.W)

        ttk.Label(self.root, text="Données GSA", font=bold_font).grid(column=0, row=len(self.gga_vars) + len(self.rmc_vars) + 2, columnspan=3)
        for idx, (key, var) in enumerate(self.gsa_vars.items()):
            ttk.Label(self.root, text=key.replace('_', ' ').title()).grid(column=0, row=len(self.gga_vars) + len(self.rmc_vars) + 3 + idx, sticky=tk.W)
            ttk.Label(self.root, textvariable=var).grid(column=1, row=len(self.gga_vars) + len(self.rmc_vars) + 3 + idx, sticky=tk.W)
            ttk.Label(self.root, text=explanations[key]).grid(column=2, row=len(self.gga_vars) + len(self.rmc_vars) + 3 + idx, sticky=tk.W)

        ttk.Label(self.root, text="Données GSV", font=bold_font).grid(column=0, row=len(self.gga_vars) + len(self.rmc_vars) + len(self.gsa_vars) + 3, columnspan=3)
        for idx, (key, var) in enumerate(self.gsv_vars.items()):
            ttk.Label(self.root, text=key.replace('_', ' ').title()).grid(column=0, row=len(self.gga_vars) + len(self.rmc_vars) + len(self.gsa_vars) + 4 + idx, sticky=tk.W)
            ttk.Label(self.root, textvariable=var).grid(column=1, row=len(self.gga_vars) + len(self.rmc_vars) + len(self.gsa_vars) + 4 + idx, sticky=tk.W)
            ttk.Label(self.root, text=explanations[key]).grid(column=2, row=len(self.gga_vars) + len(self.rmc_vars) + len(self.gsa_vars) + 4 + idx, sticky=tk.W)

        # Plotting area for positions and satellite trajectories
        self.fig, self.ax = plt.subplots(figsize=(5, 4))
        self.plot_canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.plot_canvas.get_tk_widget().grid(column=0, row=len(self.gga_vars) + len(self.rmc_vars) + len(self.gsa_vars) + len(self.gsv_vars) + 4, columnspan=3)

        self.fig_sat, self.ax_sat = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 4))
        self.satellite_canvas = FigureCanvasTkAgg(self.fig_sat, master=self.root)
        self.satellite_canvas.get_tk_widget().grid(column=3, row=0, rowspan=len(self.gga_vars) + len(self.rmc_vars) + len(self.gsa_vars) + len(self.gsv_vars) + 4)

        ttk.Label(self.root, textvariable=self.std_var).grid(column=0, row=len(self.gga_vars) + len(self.rmc_vars) + len(self.gsa_vars) + len(self.gsv_vars) + 5, columnspan=3)

        self.link_label = tk.Label(self.root, text="", fg="blue", cursor="hand2")
        self.link_label.grid(column=0, row=len(self.gga_vars) + len(self.rmc_vars) + len(self.gsa_vars) + len(self.gsv_vars) + 6, columnspan=3)

        # Start the NMEA reading in a separate thread
        self.start_reading()

class SELECTGUI:
    def __init__(self):
        self.mode = None
        self.data = None
        
        self.app = tk.Tk()
        self.gnss_types = {'GP': tk.BooleanVar(), 'GL': tk.BooleanVar(), 'GA': tk.BooleanVar(), 'GB': tk.BooleanVar(), 'GI': tk.BooleanVar(), 'GS': tk.BooleanVar()}
        self.app.minsize(400, 300)
        self.app.title("Choix de la source des données")
        self.show_static_dynamic_menu()

    def show_static_dynamic_menu(self):
        for widget in self.app.winfo_children():
            widget.destroy()

        static_button = tk.Button(self.app, text="Statique", command=lambda: self.set_mode("statique"))
        static_button.pack(pady=10)

        dynamic_button = tk.Button(self.app, text="En Déplacement", command=lambda: self.set_mode("dynamique"))
        dynamic_button.pack(pady=10)
        gnss_frame = tk.Frame(self.app)
        gnss_frame.pack(pady=10)

        for gnss, var in self.gnss_types.items():
            tk.Checkbutton(gnss_frame, text=gnss, variable=var).pack(side=tk.LEFT)


    def set_mode(self, selected_mode):
        self.mode = selected_mode
        self.show_file_or_sensor_menu()

    def show_file_or_sensor_menu(self):
        for widget in self.app.winfo_children():
            widget.destroy()

        frame = tk.Frame(self.app)
        frame.pack(pady=10)

        file_button = tk.Button(frame, text="Lire depuis un fichier", command=self.select_file)
        file_button.pack(side=tk.LEFT, padx=10)

        sensor_button = tk.Button(frame, text="Lire depuis un capteur", command=self.list_ports)
        sensor_button.pack(side=tk.LEFT, padx=10)

        self.file_label = tk.Label(self.app, text="")
        self.file_label.pack(pady=10)

        self.port_label = tk.Label(self.app, text="")
        self.port_label.pack(pady=10)

        self.ports_frame = tk.Frame(self.app)
        self.ports_frame.pack(pady=10)


        launch_button = tk.Button(self.app, text="Lancer", command=self.launch_main)
        launch_button.pack(pady=10)

    def select_file(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.data = (False, file_path)
            self.file_label.config(text=f"Fichier sélectionné: {file_path}")
            self.clear_port_selection()

    def list_ports(self):
        ports = serial.tools.list_ports.comports()
        for widget in self.ports_frame.winfo_children():
            widget.destroy()  # Clear previous buttons

        for port in ports:
            button = tk.Button(self.ports_frame, text=f"{port.device}: {port.description}",
                               command=lambda p=port.device: self.select_port(p))
            button.pack(fill=tk.X)

    def select_port(self, port):
        self.data = (True, port)
        self.port_label.config(text=f"Port sélectionné: {port}")
        self.clear_file_selection()

    def clear_file_selection(self):
        self.file_label.config(text="")
        if self.data and not self.data[0]:
            self.data = None

    def clear_port_selection(self):
        for widget in self.ports_frame.winfo_children():
            widget.destroy()
        self.port_label.config(text="")
        if self.data and self.data[0]:
            self.data = None

    def launch_main(self):
        if self.mode and self.data:
            self.clear_window()
            selected_gnss_types = [gnss for gnss, var in self.gnss_types.items() if var.get()]
            NMEAGUI(self.app, self.mode, self.data, selected_gnss_types)
        else:
            print("Mode et source de données doivent être sélectionnés")

    def clear_window(self):
        for widget in self.app.winfo_children():
            widget.destroy()

    def run(self):
        self.app.mainloop()

if __name__ == "__main__":
    SELECTGUI().run()
