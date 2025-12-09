import matplotlib.pyplot as plt
import numpy as np
import tidy3d as td
from tidy3d.plugins.resonance import ResonanceFinder
from dataclasses import asdict, fields
import ConfigurationClass


def update_config_file(dataclass_type, default_args, override_args):
    '''
    Update design parameters via default_args if all keys are valid. If not, 
    raise an error. default_args can be a dictionary or ConfigClass object
    '''

    valid_keys = {f.name for f in fields(dataclass_type)}
    invalid_keys = [k for k in override_args if k not in valid_keys]
    if invalid_keys:
        raise KeyError(f"Invalid config keys: {invalid_keys}")

    try:
        updated_args = default_args.copy()  # if default_args is a dictionary
    except:
        updated_args = asdict(
            default_args)  # if default_args is a ConfigClass object

    updated_args.update(override_args)

    return dataclass_type(**updated_args)


class Utilities:
    def __init__(self):
        pass

    def apodization(self):
        """
        Apodization to exclude the source pulse from the frequency-domain monitors.
        """

        # Time after which fields are recorded
        self.tstart = 5 / self.fwidth
        self.apodization = td.ApodizationSpec(start=self.tstart,
                                              width=self.tstart / 5)

    def parse_wavelength_range(self):
        '''
        Initialize the wavelength, frequency range for the simulation
        '''

        self.wavelength_range = np.linspace(*self.config.wavelength_range)
        self.freq0 = td.C_0 / np.mean(self.wavelength_range)
        self.freq_range = td.C_0 / self.wavelength_range
        self.fwidth = (self.freq_range[0] - self.freq_range[-1]) / 2

    def dipole_source(self,
                      location=(0, 0, 0),
                      polarization="Ey",
                      phase=0,
                      plot_bool=False):
        """
        Return dipole source emitting a gaussian pulse with center frequency freq0
        and bandwidth fwidth.
        """

        source = []
        source.append(
            td.PointDipole(
                center=location,
                source_time=td.GaussianPulse(freq0=self.freq0,
                                             fwidth=self.fwidth,
                                             phase=phase),
                polarization=polarization,
            ))

        if plot_bool:
            # Source pulse is much shorter than the simulation self.run_time,
            # so we only examine the signal up to a shorter time = 10fs
            times = np.linspace(0, 10e-13, 2000)
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            self.apodization.plot(times, ax=ax[0])
            source[0].source_time.plot(times, ax=ax[0])
            source[0].source_time.plot_spectrum(times=times, val="abs", ax=ax[1])

        # Add additional sources if present
        source.extend(self.config.additional_sources)

        return source

    def dipole_cloud(
            self,
            center=(0, 0, 0),
            span=(1, 1, 1),
            num_dipoles=5,
            polarization="Ey",
    ):
        '''
        Make a dipole cloud with dipole sources randomly positioned and with random phases
        '''

        # seed for random numbers
        rng = np.random.default_rng(420)

        dipole_positions = rng.uniform(
            center - np.asarray(span) / 2,
            center + np.asarray(span) / 2,
            [num_dipoles, 3],
        )

        dipole_phases = rng.uniform(0, 2 * np.pi, num_dipoles)

        dipole_cloud = []
        for i in range(num_dipoles):
            dipole_cloud.append(
                td.PointDipole(
                    center=tuple(dipole_positions[i]),
                    source_time=td.GaussianPulse(freq0=self.freq0,
                                                 fwidth=self.fwidth,
                                                 phase=dipole_phases[i]),
                    polarization=polarization,
                ))

        # Add additional sources if present
        dipole_cloud.extend(self.config.additional_sources)

        return dipole_cloud

    def FieldTimeMonitor(self, center=(0, 0, 0), name="time_monitor"):
        """
        Initialize num_monitors field time monitors at a specified location
        Start recording values t_start time after which souce has decayed.
        """

        FieldTimeMonitor = td.FieldTimeMonitor(center=center,
                                               size=[0, 0, 0],
                                               start=self.tstart,
                                               name=name)
        return FieldTimeMonitor

    def FieldTimeMonitor_array(self,
                               center=(0, 0, 0),
                               span=(1, 1, 1),
                               num_monitors=3,
                               randomize=True):
        """
        Initialize num_monitors field time monitors at several locations 
        that are randomized or are equally spaced, within center +- span/2
        
        Start recording field after time t_start so that the souce has decayed.
        """

        if np.sum(span) == 0 and num_monitors > 1:
            raise ValueError(
                "Specify a non-zero span when placing multiple time monitors")

        if randomize is True:
            # seed for random numbers
            rng = np.random.default_rng(420)

            monitor_positions = rng.uniform(center - np.asarray(span) / 2,
                                            center + np.asarray(span) / 2,
                                            [num_monitors, 3])
        else:
            monitor_positions = np.linspace(center - np.asarray(span) / 2,
                                            center + np.asarray(span) / 2,
                                            num_monitors)

        FieldTimeMonitors_array = []
        for i in range(num_monitors):
            FieldTimeMonitors_array.append(
                td.FieldTimeMonitor(center=tuple(monitor_positions[i]),
                                    size=(0, 0, 0),
                                    name="time_monitor_" + str(i),
                                    start=self.tstart))

        return FieldTimeMonitors_array

    def FieldMonitor(self,
                     center=[0, 0, 0],
                     size=[4, 2, 0],
                     colocate="True",
                     name="field"):
        """
        Initialize field monitor
        """
        # near field
        FieldMonitor = td.FieldMonitor(center=center,
                                       size=size,
                                       freqs=self.freq_range,
                                       name=name,
                                       apodization=self.apodization,
                                       colocate=colocate)

        return FieldMonitor

    def FarFieldMonitor(self, size=(td.inf, td.inf, 0), name="far_field monitor"):
        """
        Initialize far field monitor
        """

        # local (x,y)*2*pi/wavelength coordinates over which the far field is defined
        ux = np.linspace(-1, 1, 201)
        uy = np.linspace(-1, 1, 201)

        FarFieldMonitor = td.FieldProjectionKSpaceMonitor(
            center=(0, 0, self.config.slab_thickness / 2 + 0.1),
            size=size,
            freqs=self.freq_range,
            name=name,
            proj_axis=2,
            ux=ux,
            uy=uy,
            apodization=self.apodization,
        )

        return FarFieldMonitor

    def FarFieldAngleMonitor(self,
                             size=(td.inf, td.inf, 0),
                             name="far_field angle monitor"):
        """
        Initialize far field angle monitor
        """

        # define field projection angle ranges
        theta_array = np.linspace(0, np.deg2rad(90), 100)  # polar angle
        phi_array = np.linspace(0, 2 * np.pi, 200)  # azimuthal angle

        # define the top field projection monitor
        FarFieldAngleMonitor = td.FieldProjectionAngleMonitor(
            name=name,
            center=(0, 0, self.config.slab_thickness / 2 + 0.1),
            size=size,
            freqs=self.freq_range,
            theta=theta_array,
            phi=phi_array,
            apodization=self.apodization)

        return FarFieldAngleMonitor

    def EpsMonitor(self, center=[0, 0, 0], size=[4, 2, 0], name="eps_monitor"):
        """
        Permittivity Monitor for mode volume calculation
        """
        EpsMonitor = td.PermittivityMonitor(center=center,
                                            size=size,
                                            freqs=self.freq_range,
                                            name=name,
                                            apodization=self.apodization,
                                            colocate=False)

        return EpsMonitor

    def get_mesh_grid(self, grid_size=10):
        """
        Define mesh grid for FDTD simulation
        """

        # Mesh step in x, y, z, in micron
        grid_spec = td.GridSpec(
            grid_x=td.AutoGrid(min_steps_per_wvl=grid_size),
            grid_y=td.AutoGrid(min_steps_per_wvl=grid_size),
            grid_z=td.AutoGrid(min_steps_per_wvl=grid_size),
        )

        return grid_spec

    def run_simulation(self, simulation):
        """
        Run Tidy3D nanobeam simulation
        """

        sim_data = td.web.run(simulation,
                              task_name=self.config.task_name,
                              verbose=self.config.verbose)
        return sim_data

    def initialize_monitors(self):

        # Check if atleast one monitor is present
        if (self.config.field_monitor + self.config.field_time_monitor +
                self.config.eps_monitor + self.config.far_field_monitor +
                self.config.far_field_angle_monitor) == 0:
            raise ValueError("Add Monitors!")

        monitors = []

        # Add time monitor: either one or a randomly distributed array in specified span
        if self.config.field_time_monitor and not self.config.field_time_monitor_array:
            monitors.append(
                self.FieldTimeMonitor(center=self.config.field_time_monitor_center))
        elif self.config.field_time_monitor_array:
            monitors.extend(
                self.FieldTimeMonitor_array(
                    center=self.config.field_time_monitor_center,
                    span=self.config.field_time_monitor_span,
                    num_monitors=self.config.num_field_time_monitor,
                    randomize=self.config.field_time_monitor_location_randomize))

        # Add field monitor
        if self.config.field_monitor:
            monitors.append(
                self.FieldMonitor(size=self.config.field_monitor_size,
                                  center=self.config.field_monitor_center,
                                  colocate=not self.config.eps_monitor))

        # Add far field monitor
        if self.config.far_field_monitor:
            monitors.append(
                self.FarFieldMonitor(size=self.config.far_field_monitor_size))

        # Add far field angle monitor
        if self.config.far_field_angle_monitor:
            monitors.append(
                self.FarFieldAngleMonitor(
                    size=self.config.far_field_angle_monitor_size))

        # Add permittivity monitor
        if self.config.eps_monitor:
            monitors.append(self.EpsMonitor(size=self.config.field_monitor_size))

        # Add additional monitors if present
        monitors.extend(self.config.additional_monitors)

        return monitors

    def parse_simulation_options(self):
        '''
        Initialize various things for the simulation to run
        '''

        monitors = self.initialize_monitors()  # Add monitors

        # Add a single dipole source or a dipole cloud
        if not self.config.dipole_cloud:
            source = self.dipole_source(location=self.config.dipole_center,
                                        polarization=self.config.source_polarization,
                                        phase=self.config.dipole_phase)
        else:
            source = self.dipole_cloud(center=self.config.dipole_center,
                                       span=self.config.dipole_cloud_span,
                                       num_dipoles=self.config.num_dipoles,
                                       polarization=self.config.source_polarization)

        return source, monitors

    def make_simulation(self, simulation_options):
        """
        Initialize the nanobeam simulation object in Tidy3d and plot the simulation
        """

        # Update config file for simulation options
        self.config = update_config_file(
            getattr(ConfigurationClass, self.config_filename), self.config,
            simulation_options)

        source, monitors = self.parse_simulation_options()

        sim = td.Simulation(
            size=self.config.sim_size,
            center=self.config.sim_center,
            symmetry=self.config.symmetry,
            grid_spec=self.get_mesh_grid(self.config.mesh_grid_size),
            run_time=self.config.run_time * 1e-12,
            medium=self.config.background_medium,
            sources=source,
            monitors=monitors,
            boundary_spec=self.config.boundary_spec,
            structures=self.structures,
        )

        if self.config.plot:
            self.plot_simulation(sim)

        return sim

    def plot_simulation(self, sim):
        '''
        Plot the simulation
        '''

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        sim.plot(z=0, ax=ax[0])
        sim.plot(y=0, ax=ax[1])
        plt.show()

    @staticmethod
    def convert_freq_to_wavelength(freq):
        """Convert frequency to wavelength (nm)"""

        wavelength = td.C_0 / freq
        return np.round(wavelength, 5)

    def find_frequency_index(self, freq_range, frequency):
        """
        Find the index in array freq_range that is closest to value frequency
        """
        return np.argmin(np.abs(freq_range - frequency))

    def load_simulation(self, weights_name):

        sim_data = td.SimulationData.from_file(
            rf"D:\Neelesh\inverse design\Tidy3D_data\{weights_name}.hdf5")

        return sim_data

    def analyze_FieldTimeMonitor(
        self,
        sim_data,
        freq_range,
        monitor_name="time_monitor",
        plot_bool=True,
        print_data_bool=False,
        filter_resonances=False,
    ):
        """
        Analyse data from time monitor(s).
        """

        # Get data from the TimeMonitor
        time_series = None

        if not self.config.field_time_monitor_array:
            time_series = getattr(sim_data[monitor_name],
                                  self.config.source_polarization).squeeze()

        else:  # average data from all time monitors
            for i in range(self.config.num_field_time_monitor):
                monitor_name = f"time_monitor_{1}"
                data = getattr(sim_data[monitor_name],
                               self.config.source_polarization).squeeze()
                time_series = data if time_series is None else time_series + data

            time_series = time_series / self.config.num_field_time_monitor

            print("hehe")

        freq_window = (freq_range[-1], freq_range[0])

        if plot_bool:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))

            # Plot time dependence
            time_series.plot(ax=ax1)

            # Make frequency mesh and plot spectrum
            dt = sim_data.simulation.dt
            fmesh = np.linspace(-1 / dt / 2, 1 / dt / 2, time_series.size)
            spectrum = np.fft.fftshift(np.fft.fft(time_series))

            ax2.plot(fmesh, np.abs(spectrum))
            ax2.set_xlim(freq_window)
            ax2.set_xlabel("Frequency [Hz]")
            ax2.set_ylabel("Electric field [a.u.]")
            ax2.set_title("Spectrum")
            plt.show()

        resonance_finder = ResonanceFinder(freq_window=freq_window)
        resonance_data = resonance_finder.run(sim_data[monitor_name])

        if filter_resonances:
            # Keep modes only in the frequency window; drop the rest
            resonance_data = resonance_data.where(
                resonance_data.freq < freq_range[0], drop=True)
            resonance_data = resonance_data.where(
                resonance_data.freq > freq_range[-1], drop=True)

        freq_index = np.argmin(np.asarray(resonance_data.error))
        self.resonant_frequency = np.asarray(resonance_data.freq)[freq_index]
        self.index = self.find_frequency_index(freq_range, self.resonant_frequency)
        self.quality_factor = np.asarray(resonance_data.Q)[freq_index]
        self.resonant_wavelength = np.round(td.C_0 / self.resonant_frequency * 1000,
                                            2)

        if print_data_bool:
            print(resonance_data.to_dataframe())
            print(f"Cavity Resonance at {self.resonant_wavelength} nm; "
                  f"Q = {np.round(self.quality_factor)}")

    def analyze_FieldMonitor(self,
                             sim_data,
                             freq_range,
                             freq=None,
                             plot_field="E",
                             val="abs^2",
                             eps_alpha=0.3,
                             ax=None):

        if freq is None:
            freq = self.resonant_frequency

        # Check if freq is in freq_range
        if (freq > freq_range[0] or freq < freq_range[-1]):
            raise ValueError(
                "The plotting wavelength of "
                f"{self.convert_freq_to_wavelength(freq)*1e3} nm is outside the "
                "wavelength range")

        if ax is None:
            _, ax = plt.subplots(1)

        sim_data.plot_field(
            "field",
            plot_field,
            val=val,
            ax=ax,
            z=self.config.field_monitor_center[2],
            eps_alpha=eps_alpha,
            f=freq,
        )

        ax.set_title(f"{val}({plot_field}) at {np.round(td.C_0/freq*1e3)} nm")
        plt.show()

    def analyze_FarFieldMonitor(self,
                                sim_data,
                                plot_field="E",
                                val="abs",
                                index=None,
                                plot_bool=True):

        if index is None:
            index = self.index

        if plot_field == "power":
            farfield = sim_data["far_field monitor"].power[:, :, :, index].T
        elif plot_field == "E":
            farfield = getattr(sim_data["far_field monitor"].fields_cartesian,
                               self.config.source_polarization)[:, :, :, index].T
        elif plot_field == "Ex":
            farfield = sim_data["far_field monitor"].fields_cartesian.Ex[:, :, :,
                                                                         index].T
        elif plot_field == "Ey":
            farfield = sim_data["far_field monitor"].fields_cartesian.Ey[:, :, :,
                                                                         index].T
        elif plot_field == "Ez":
            farfield = sim_data["far_field monitor"].fields_cartesian.Ez[:, :, :,
                                                                         index].T
        elif plot_field == "Hx":
            farfield = sim_data["far_field monitor"].fields_cartesian.Hx[:, :, :,
                                                                         index].T
        elif plot_field == "Hy":
            farfield = sim_data["far_field monitor"].fields_cartesian.Hy[:, :, :,
                                                                         index].T
        elif plot_field == "Hz":
            farfield = sim_data["far_field monitor"].fields_cartesian.Hz[:, :, :,
                                                                         index].T
        else:
            raise ValueError("Supported inputs: 'E', 'power'")

        ux = np.array(farfield.ux)
        uy = np.array(farfield.uy)

        if val == "real":
            farfield = np.real(farfield)
            farfield = np.squeeze(np.nan_to_num(farfield))

        elif val == "imag":
            farfield = np.imag(farfield)
            farfield = np.squeeze(np.nan_to_num(farfield))

        elif val == "abs":
            farfield = np.abs(farfield)
            farfield = np.squeeze(farfield)

        elif val == "abs^2":
            farfield = np.abs(farfield)**2
            farfield = np.squeeze(farfield)

        elif val == "complex":
            farfield = np.squeeze(np.nan_to_num(farfield))

        else:
            raise ValueError("Supported values: 'real', 'abs', 'abs^2', complex")

        if plot_bool:
            fig = plt.figure()
            ax = fig.add_subplot()

            if val == "real":
                im = ax.imshow(farfield,
                               extent=[ux.min(),
                                       ux.max(),
                                       uy.min(),
                                       uy.max()])
            else:
                im = ax.imshow(farfield,
                               extent=[ux.min(),
                                       ux.max(),
                                       uy.min(),
                                       uy.max()])
            fig.colorbar(im, ax=ax, label=f"{val}({plot_field})")
            ax.set_xlabel("$k_x/k$")
            ax.set_ylabel("$k_y/k$")

            plt.title(f"Q = {np.round(self.quality_factor)}, "
                      f"$\lambda_0$ = {self.resonant_wavelength} nm")

            if val != "real":
                phis = np.linspace(0, np.pi * 2, 101)
                ax.plot(np.cos(phis), np.sin(phis), lw=8, color="w")
                ax.plot(np.cos(phis) * 0.12, np.sin(phis) * 0.12, lw=2, color="w")

            plt.show()

            return (fig, ax, farfield)

        else:
            return farfield

    def analyze_FarFieldAngleMonitor(self, sim_data, val="abs", save_plot=False):

        if val == "abs":
            angle_farfield = sim_data[
                "far_field angle monitor"].power[:, :, :,
                                                 self.index].squeeze(drop=True)

            angle_farfield = angle_farfield / np.max(angle_farfield)
        else:
            raise NotImplementedError("Only |E|^2 plotting implements")

        # plot the radiation pattern in polar coordinates
        fig = plt.figure()
        ax = fig.add_subplot(111, polar=True, facecolor="white")
        c = ax.pcolor(angle_farfield.phi,
                      np.sin(angle_farfield.theta),
                      angle_farfield,
                      shading="auto",
                      cmap="viridis",
                      vmin=0,
                      vmax=1)

        # Set radial limits and ticks
        ax.set_ylim(0, 1)
        na_ticks = [0.39]
        ax.set_yticks(na_ticks)
        ax.set_yticklabels([f"{val}" for val in na_ticks], color="white")
        # ax.set_ylabel("Numerical Aperture (NA)", labelpad=20)

        # Optional: adjust aesthetics
        ax.tick_params(labelsize=10)
        ax.grid(True, linestyle="--", alpha=1)

        # Remove azimuthal angle labels
        ax.set_thetagrids([])  # no labels on 0°, 45°, etc.

        # Optional: remove radial grid lines too
        ax.grid(True)
        ax.xaxis.grid(False)  # turn off azimuthal (theta) grid
        # cb = fig.colorbar(c, ax=ax)
        # cb.set_label("|E$|^2$")
        # _ = plt.setp(ax.get_yticklabels(), color="white")
        # plt.title(f"Q = {np.round(self.quality_factor)}, "
        #           f"$\lambda_0$ = {self.resonant_wavelength} nm")

        return fig, ax, angle_farfield

    def plot_simulation_3D(self, sim_data):
        """
        Make a 3D render of the simulation.
        """
        sim_data.plot_3d()

    def estimate_cost(self, simulation):
        task_id = td.web.upload(simulation,
                                task_name=self.config.task_name,
                                verbose=False)
        cost = td.web.estimate_cost(task_id)
        return cost

    @staticmethod
    def run_batch_simulation(batch_simulation, verbose=True):
        batch = td.web.Batch(simulations=batch_simulation, verbose=verbose)
        batch_data = batch.run(path_dir="data")
        return batch_data

    def analyze_batch_data(self,
                           batch_data,
                           freq_range,
                           plot_bool=False,
                           field_monitor=False):
        quality_factor_array = []
        resonance_wavelength_array = []

        for idx, (task_name, sim_data) in enumerate(batch_data.items()):
            print(task_name)
            self.analyze_FieldTimeMonitor(sim_data,
                                          freq_range=freq_range,
                                          plot_bool=plot_bool)

            quality_factor_array.append(self.quality_factor)
            resonance_wavelength_array.append(self.resonant_wavelength)
            print(f"Resonance Wavelength: {self.resonant_wavelength}, "
                  f"Quality Factor: {np.round(self.quality_factor)}")

            if field_monitor is True:
                self.analyze_FieldMonitor(sim_data, freq_range)

        quality_factor_array = np.asarray(quality_factor_array)
        resonance_wavelength_array = np.asarray(resonance_wavelength_array)

        return quality_factor_array, resonance_wavelength_array

    def analyze_noisy_batch_data(self, batch_data, freq_range):
        '''
        Function specifically to analyze batch data generated by noisy batch runs
        as in file L3_cavity_tidy3D_520nm_G3.ipynb
        '''
        quality_factor_array = []
        wavelength_array = []
        far_field_array = []
        overlap_array = []

        for idx, (task_name, sim_data) in enumerate(batch_data.items()):
            print(task_name)
            # From time monitor get Q and wavelength
            self.analyze_FieldTimeMonitor(sim_data,
                                          freq_range=freq_range,
                                          plot_bool=False)

            # From far field monitor get Real{Ey} and the overlap
            farfield = self.analyze_FarFieldMonitor(sim_data,
                                                    plot_field="E",
                                                    val="real",
                                                    plot_bool=False)

            overlap, _ = self.farfield_overlap_poynting(sim_data,
                                                        focal_length=4,
                                                        target_width=1.65,
                                                        filter_NA=0.55,
                                                        plot_bool=False)

            print(f"Q: {np.round(self.quality_factor)}, "
                  f"overlap: {np.round(overlap,3)}, "
                  f"wavelength: {self.resonant_wavelength} nm")

            quality_factor_array.append(self.quality_factor)
            wavelength_array.append(self.resonant_wavelength)
            far_field_array.append(farfield)
            overlap_array.append(overlap)

        return quality_factor_array, wavelength_array, far_field_array, overlap_array

    def calculate_bandstructure(
        self,
        unit_cell,
        unit_cell_size,
        boundary_spec,
        sim_size,
        run_time,
        field_monitor=False,
        plot_sim_bool=False,
    ):
        rng = np.random.default_rng(69)

        # Initialize dipole sources at random locations on the unit cell
        num_dipoles = 7

        dipole_positions = rng.uniform(
            [-unit_cell_size[0] / 2, -unit_cell_size[1] / 2, 0],
            [unit_cell_size[0] / 2, unit_cell_size[1] / 2, 0],
            [num_dipoles, 3],
        )

        dipole_phases = rng.uniform(0, 2 * np.pi, num_dipoles)

        # Setting dipole polarization and symmetry to be Hz and the symmetry to be (0,0,1)
        # to excite only modes which are even with respect to the xy mirror plane.
        polarization = "Hz"
        symmetry = (0, 0, 1)

        dipoles = []
        for i in range(num_dipoles):
            dipoles.append(
                self.dipole_source(
                    location=tuple(dipole_positions[i]),
                    phase=dipole_phases[i],
                    polarization=polarization,
                ))

        num_monitors = 2
        # Initialize time monitors at different locations
        monitor_positions = rng.uniform(
            [-unit_cell_size[0] / 2, -unit_cell_size[1] / 2, 0],
            [unit_cell_size[0] / 2, unit_cell_size[1] / 2, 0],
            [num_monitors, 3],
        )

        monitors = []
        for i in range(num_monitors):
            monitors.append(
                self.FieldTimeMonitor(center=tuple(monitor_positions[i]),
                                      name="time_monitor_" + str(i)))

        if (field_monitor):
            FieldMonitor_xy = self.FieldMonitor(center=[0, 0, 0],
                                                size=[sim_size[0], sim_size[1], 0],
                                                name="field_xy")
            monitors.append(FieldMonitor_xy)

            FieldMonitor_yz = self.FieldMonitor(center=[0, 0, 0],
                                                size=[0, sim_size[1], sim_size[2]],
                                                name="field_yz")
            monitors.append(FieldMonitor_yz)

        # Create simulation batch sweeping the Bloch boundary condition
        batch_simulation = {}
        for i in range(len(boundary_spec)):
            batch_simulation[f"sim_{i}"] = td.Simulation(
                size=sim_size,
                grid_spec=td.GridSpec.auto(),
                structures=unit_cell,
                sources=dipoles,
                monitors=monitors,
                run_time=run_time,
                boundary_spec=boundary_spec[i],
                symmetry=symmetry,
            )

        if plot_sim_bool:
            fig, ax = plt.subplots(1, 2, tight_layout=True, figsize=(10, 4))
            batch_simulation["sim_0"].plot(z=0.0, ax=ax[0])
            batch_simulation["sim_0"].plot(x=0, ax=ax[1])
            plt.show()

        batch_data = self.run_batch_simulation(batch_simulation)
        return batch_data

    def mode_volume(self, sim_data):
        """
        See: https://docs.flexcompute.com/projects/tidy3d/en/v2.7.3/notebooks/CavityFOM.html#Cavity-Effective-Mode-Volume
        """

        # Electric field.
        e_x = sim_data["field"].Ex.isel(f=self.index).values
        e_y = sim_data["field"].Ey.isel(f=self.index).values
        e_z = sim_data["field"].Ez.isel(f=self.index).values
        e_2 = np.abs(e_x)**2 + np.abs(e_y)**2 + np.abs(e_z)**2

        # Permittivity distribution.
        eps = abs(sim_data['eps_monitor'].eps_xx).isel(f=self.index)

        # Calculation of effective mode volume.
        e_eps = eps * e_2
        num = e_eps.integrate(coord=("x", "y", "z")).item()
        den = np.amax(e_eps)
        V_eff = (num / den).values
        self.mode_volume = V_eff

        print(f"V_eff = {V_eff / 1e18:.3e} m^3")
        print(
            f"V_eff = {V_eff / (self.resonant_wavelength/1e3/self.config.slab_medium_index)**3:.2f} "
            "(lambda/n)^3")

        return V_eff

    def purcell_factor(self, plot_range="False"):

        purcell_factor = (3 / (4 * np.pi**2)) * (
            (self.resonant_wavelength / 1e3 / self.config.slab_medium_index)**
            3) * (self.quality_factor / self.mode_volume)

        self.purcell_factor = purcell_factor
        print(f"Purcell Factor at {self.resonant_wavelength} nm = {purcell_factor}")

        if (plot_range):

            w_c = 2 * np.pi * self.resonant_frequency
            detuning = 5e9
            del_w = np.linspace(-detuning * 2 * np.pi, detuning * 2 * np.pi, 2000)
            F_p_w = self.purcell_factor * (w_c**2 / (4 * self.quality_factor)**2) / (
                del_w**2 + (w_c**2 / (4 * self.quality_factor)**2))

            fig, ax1 = plt.subplots(1, 1, tight_layout=True, figsize=(7, 4))

            ax1.plot(del_w * 1e-9 / (2 * np.pi), F_p_w)
            ax1.set_xlabel("$(\omega_{c} - \omega)/2 \pi$ (GHz)")
            ax1.set_ylabel("Purcell Factor")
            ax1.set_xlim(-detuning * 1e-9, detuning * 1e-9)
            plt.show()

        return purcell_factor

    def calculate_farfield_gaussian_overlap(self,
                                            sim_data,
                                            focal_length,
                                            target_width=1,
                                            filter_NA=False,
                                            plot_bool=True):

        far_field_Ex = np.nan_to_num(
            self.analyze_FarFieldMonitor(sim_data,
                                         plot_field="Ex",
                                         val="abs^2",
                                         plot_bool=False))
        far_field_Ez = np.nan_to_num(
            self.analyze_FarFieldMonitor(sim_data,
                                         plot_field="Ez",
                                         val="abs^2",
                                         plot_bool=False))
        far_field_Ey = np.nan_to_num(
            self.analyze_FarFieldMonitor(sim_data,
                                         plot_field="Ey",
                                         val="abs^2",
                                         plot_bool=False))

        far_field_abs = far_field_Ex + far_field_Ey + far_field_Ez
        # far_field_abs = np.sqrt(far_field_Ey)

        far_field = self.analyze_FarFieldMonitor(sim_data,
                                                 plot_field="Ey",
                                                 val="complex",
                                                 plot_bool=False)

        limit = 0.99
        (ux, uy) = (np.linspace(-limit, limit, far_field.shape[0]),
                    np.linspace(-limit, limit, far_field.shape[0]))

        ux_, uy_ = np.meshgrid(ux, uy)

        # # Denominator for both x and y
        # denom = np.sqrt(np.maximum(1.0 - ux_**2 - uy_**2, 1e-12))  # clamp min value

        # # Avoid division by zero
        # x = focal_length * ux_ / denom
        # y = focal_length * uy_ / denom

        # # Optional: mask NA >= 1 region
        # mask = (ux_**2 + uy_**2) >= 1.0
        # x[mask] = np.inf
        # y[mask] = np.inf

        x = focal_length * ux_ / np.sqrt(1 - ux_**2)
        y = focal_length * uy_ / np.sqrt(1 - uy_**2)

        # print("Ez= ", np.trapz(np.trapz(far_field_Ez, x[0]), x[0]))
        # temp = np.where(np.sqrt(ux_**2 + uy_**2) <= filter_NA, far_field_Ez, 0)
        # print("Ez filtered= ", np.trapz(np.trapz(temp, x[0]), x[0]))

        width = np.linspace(0.1, 3, 500)
        overlap_integral_array = np.zeros(len(width))

        # Integral of E**2 for normalization
        far_field, far_field_abs = (np.nan_to_num(far_field),
                                    np.nan_to_num(far_field_abs))

        # Consider the portion of the field going into the objective
        if filter_NA is not False:
            far_field = np.where(np.sqrt(ux_**2 + uy_**2) <= filter_NA, far_field, 0)
            far_field_abs = np.where(
                np.sqrt(ux_**2 + uy_**2) <= 0.99, far_field_abs, 0)

        far_field_norm = np.trapz(np.trapz(far_field_abs, x[0]), x[0])
        plt.imshow(far_field_abs / np.max(far_field_abs))
        plt.colorbar()
        # far_field_norm = np.trapz(np.trapz(far_field_abs**2))
        print(np.trapz(np.trapz(far_field_abs / np.max(far_field_abs), x[0]), x[0]))

        for i in range(len(width)):

            # 2-D symmetric normalized gaussian function polarized along x
            gaussian = 1 / (np.pi * width[i]**2) * np.exp(-(x**2 + y**2) /
                                                          (width[i]**2))

            gaussian_norm = np.trapz(np.trapz(gaussian**2, x[0]), x[0])

            integrated_value = np.trapz(np.trapz(far_field * gaussian, x[0]), x[0])

            overlap_integral_array[i] = np.abs(integrated_value)**2 / (
                far_field_norm * gaussian_norm)
        target_overlap = overlap_integral_array[np.argmin(
            np.abs(target_width - width))]

        if plot_bool:
            NA_array = np.arcsin(1 / np.sqrt(1 + (focal_length / width)**2))
            plt.figure()
            plt.plot(NA_array, overlap_integral_array, linewidth=2)
            plt.grid(True)

            plt.title(
                f"Overlap = {np.round(target_overlap*100,2)}% "
                f"for w = {target_width} mm",
                fontsize=15)
            # plt.axvline(x=width_max, color='k', linestyle='--')
            # plt.axvline(x=target_width, color='k', linestyle='-.')
            plt.axhline(y=target_overlap, color='k', linestyle='-.')
            plt.xlabel("NA", fontsize=15)
            plt.ylabel("Overlap Integral", fontsize=15)
            # plt.ylim(0, 1.05)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.show()

        return target_overlap, overlap_integral_array

    def calculate_farfield_gaussian_overlap_poynting(self,
                                                     sim_data,
                                                     focal_length,
                                                     target_width=1,
                                                     filter_NA=False,
                                                     plot_bool=True):

        far_field_Ex = np.nan_to_num(
            self.analyze_FarFieldMonitor(sim_data,
                                         plot_field="Ex",
                                         val="complex",
                                         plot_bool=False))
        far_field_Ez = np.nan_to_num(
            self.analyze_FarFieldMonitor(sim_data,
                                         plot_field="Ez",
                                         val="complex",
                                         plot_bool=False))
        far_field_Ey = np.nan_to_num(
            self.analyze_FarFieldMonitor(sim_data,
                                         plot_field="Ey",
                                         val="complex",
                                         plot_bool=False))
        far_field_Hx = np.nan_to_num(
            self.analyze_FarFieldMonitor(sim_data,
                                         plot_field="Hx",
                                         val="complex",
                                         plot_bool=False))
        far_field_Hz = np.nan_to_num(
            self.analyze_FarFieldMonitor(sim_data,
                                         plot_field="Hz",
                                         val="complex",
                                         plot_bool=False))
        far_field_Hy = np.nan_to_num(
            self.analyze_FarFieldMonitor(sim_data,
                                         plot_field="Hy",
                                         val="complex",
                                         plot_bool=False))

        far_field_E = np.zeros((far_field_Ex.shape[0], far_field_Ex.shape[1], 3),
                               dtype=complex)
        (far_field_E[..., 0], far_field_E[..., 1],
         far_field_E[..., 2]) = (far_field_Ex, far_field_Ey, far_field_Ez)

        far_field_H = np.zeros((far_field_Hx.shape[0], far_field_Hx.shape[1], 3),
                               dtype=complex)
        (far_field_H[..., 0], far_field_H[..., 1],
         far_field_H[..., 2]) = (far_field_Hx, far_field_Hy, far_field_Hz)

        poynting_vector = np.cross(far_field_E, np.conj(far_field_H), axis=-1)

        far_field_abs = poynting_vector[..., 2]  # Pz

        (ux, uy) = (np.linspace(-0.99, 0.99, far_field_Ey.shape[0]),
                    np.linspace(-0.99, 0.99, far_field_Ey.shape[0]))

        ux_, uy_ = np.meshgrid(ux, uy)

        x = focal_length * ux_ / np.sqrt(1 - ux_**2)
        y = focal_length * uy_ / np.sqrt(1 - uy_**2)

        width = np.linspace(0.1, 3, 500)
        overlap_integral_array = np.zeros(len(width))

        far_field_E = np.nan_to_num(far_field_E)
        far_field_abs = np.nan_to_num(far_field_abs)

        # Consider the portion of the field going into the objective
        if filter_NA is not False:
            far_field_E[..., 0] = np.where(
                np.sqrt(ux_**2 + uy_**2) <= filter_NA, far_field_E[..., 0], 0)
            far_field_E[..., 1] = np.where(
                np.sqrt(ux_**2 + uy_**2) <= filter_NA, far_field_E[..., 1], 0)
            far_field_E[..., 2] = np.where(
                np.sqrt(ux_**2 + uy_**2) <= filter_NA, far_field_E[..., 2], 0)
            far_field_abs = np.where(
                np.sqrt(ux_**2 + uy_**2) <= 0.99, far_field_abs, 0)

        far_field_norm = np.real(np.trapz(np.trapz(far_field_abs, x[0]), x[0]))
        # far_field_norm = np.real(np.trapz(np.trapz(far_field_abs)))
        # print(np.abs(np.trapz(np.trapz(far_field_abs/np.max(far_field_abs), x[0]), x[0])))

        for i in range(len(width)):

            # 2-D symmetric normalized gaussian function polarized along x
            # gaussian = np.zeros((x.shape[0], x.shape[1], 3), dtype=complex)
            # gaussian[...,
            #          0] = (np.sqrt(2 / (td.C_0 * td.MU_0 * np.pi * width[i]**2)) *
            #                np.exp(-(x**2 + y**2) / (width[i]**2)))
            gaussian = (np.sqrt(2 / (td.C_0 * td.MU_0 * np.pi * width[i]**2)) *
                        np.exp(-(x**2 + y**2) / (width[i]**2)))

            # integrated_value = np.trapz(
            #     np.trapz(np.cross(far_field_E, gaussian, axis=-1)[..., 2], x[0]),
            #     x[0])
            integrated_value = np.trapz(
                np.trapz(far_field_E[..., 1] * gaussian, x[0]), x[0])

            overlap_integral_array[i] = (np.abs(integrated_value)**2 /
                                         (far_field_norm))

        target_overlap = overlap_integral_array[np.argmin(
            np.abs(target_width - width))]

        if plot_bool:
            NA_array = np.arcsin(1 / np.sqrt(1 + (focal_length / width)**2))
            plt.figure()
            plt.plot(NA_array, overlap_integral_array, linewidth=2)
            plt.grid(True)

            plt.title(
                f"Overlap = {np.round(target_overlap*100,2)}% "
                f"for w = {target_width} mm",
                fontsize=15)
            # plt.axvline(x=width_max, color='k', linestyle='--')
            # plt.axvline(x=target_width, color='k', linestyle='-.')
            plt.axhline(y=target_overlap, color='k', linestyle='-.')
            plt.xlabel("NA", fontsize=15)
            plt.ylabel("Overlap Integral", fontsize=15)
            # plt.ylim(0, 1.05)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.show()

        return target_overlap, overlap_integral_array

    def farfield_overlap_poynting(self,
                                  sim_data,
                                  focal_length,
                                  target_width=1,
                                  filter_NA=False,
                                  index=None,
                                  plot_bool=True):

        far_field_Ex = np.nan_to_num(
            self.analyze_FarFieldMonitor(sim_data,
                                         plot_field="Ex",
                                         val="complex",
                                         index=index,
                                         plot_bool=False))
        far_field_Ez = np.nan_to_num(
            self.analyze_FarFieldMonitor(sim_data,
                                         plot_field="Ez",
                                         val="complex",
                                         index=index,
                                         plot_bool=False))
        far_field_Ey = np.nan_to_num(
            self.analyze_FarFieldMonitor(sim_data,
                                         plot_field="Ey",
                                         val="complex",
                                         index=index,
                                         plot_bool=False))
        far_field_Hx = np.nan_to_num(
            self.analyze_FarFieldMonitor(sim_data,
                                         plot_field="Hx",
                                         val="complex",
                                         index=index,
                                         plot_bool=False))
        far_field_Hz = np.nan_to_num(
            self.analyze_FarFieldMonitor(sim_data,
                                         plot_field="Hz",
                                         val="complex",
                                         index=index,
                                         plot_bool=False))
        far_field_Hy = np.nan_to_num(
            self.analyze_FarFieldMonitor(sim_data,
                                         plot_field="Hy",
                                         val="complex",
                                         index=index,
                                         plot_bool=False))

        far_field_E = np.zeros((far_field_Ex.shape[0], far_field_Ex.shape[1], 3),
                               dtype=complex)
        (far_field_E[..., 0], far_field_E[..., 1],
         far_field_E[..., 2]) = (far_field_Ex, far_field_Ey, far_field_Ez)

        far_field_H = np.zeros((far_field_Hx.shape[0], far_field_Hx.shape[1], 3),
                               dtype=complex)
        (far_field_H[..., 0], far_field_H[..., 1],
         far_field_H[..., 2]) = (far_field_Hx, far_field_Hy, far_field_Hz)

        poynting_vector = np.cross(far_field_E, np.conj(far_field_H), axis=-1)

        far_field_abs = poynting_vector[..., 2]  # Pz

        (ux, uy) = (np.linspace(-0.99, 0.99, far_field_Ey.shape[0]),
                    np.linspace(-0.99, 0.99, far_field_Ey.shape[0]))
        ux_, uy_ = np.meshgrid(ux, uy)

        x = focal_length * ux_ / np.sqrt(1 - ux_**2)
        y = focal_length * uy_ / np.sqrt(1 - uy_**2)

        width = np.linspace(0.1, 3, 500)
        overlap_integral_array = np.zeros(len(width))

        far_field_E = np.nan_to_num(far_field_E)
        far_field_abs = np.nan_to_num(far_field_abs)

        # Consider the portion of the field going into the objective
        if filter_NA is not False:
            far_field_E[..., 0] = np.where(
                np.sqrt(ux_**2 + uy_**2) <= filter_NA, far_field_E[..., 0], 0)
            far_field_E[..., 1] = np.where(
                np.sqrt(ux_**2 + uy_**2) <= filter_NA, far_field_E[..., 1], 0)
            far_field_E[..., 2] = np.where(
                np.sqrt(ux_**2 + uy_**2) <= filter_NA, far_field_E[..., 2], 0)
            far_field_abs = np.where(
                np.sqrt(ux_**2 + uy_**2) <= 0.99, far_field_abs, 0)

        far_field_norm = np.real(np.trapz(np.trapz(far_field_abs, x[0]), x[0]))
        for i in range(len(width)):

            # 2-D symmetric normalized gaussian function polarized along x
            gaussian = (np.sqrt(2 / (td.C_0 * td.MU_0 * np.pi * width[i]**2)) *
                        np.exp(-(x**2 + y**2) / (width[i]**2)))

            integrated_value = np.trapz(
                np.trapz(far_field_E[..., 1] * gaussian, x[0]), x[0])

            overlap_integral_array[i] = (np.abs(integrated_value)**
                                         2) / (far_field_norm)

        target_overlap = overlap_integral_array[np.argmin(
            np.abs(target_width - width))]

        if plot_bool:
            NA_array = np.arcsin(1 / np.sqrt(1 + (focal_length / width)**2))
            plt.figure()
            plt.plot(NA_array, overlap_integral_array, linewidth=2)
            plt.grid(True)

            plt.title(
                f"Overlap = {np.round(target_overlap*100,2)}% "
                f"for w = {target_width} mm",
                fontsize=15)
            # plt.axvline(x=width_max, color='k', linestyle='--')
            # plt.axvline(x=target_width, color='k', linestyle='-.')
            plt.axhline(y=target_overlap, color='k', linestyle='-.')
            plt.xlabel("NA", fontsize=15)
            plt.ylabel("Overlap Integral", fontsize=15)
            # plt.ylim(0, 1.05)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.show()

        return target_overlap, overlap_integral_array, NA_array

    def fourier_transform_2D(self, sim_data, plot_bool=True):

        Ey = (sim_data["field"].Ey[:, :, 0, self.index]).T
        dx = sim_data["field"].Ey[:, :, 0, 0].x[1] - sim_data["field"].Ey[:, :, 0,
                                                                          0].x[0]
        dy = sim_data["field"].Ey[:, :, 0, 0].y[1] - sim_data["field"].Ey[:, :, 0,
                                                                          0].y[0]

        # Calculate the 2D FFT
        fft_data = np.fft.fft2(Ey)
        # Shift zero frequency to the center
        fft_data = np.fft.fftshift(fft_data)

        # Compute frequency coordinates
        ny, nx = Ey.shape
        kx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx.values))
        kx = kx / (2 * np.pi * self.config.alattice * self.config.slab_medium_index)
        ky = np.fft.fftshift(np.fft.fftfreq(ny, d=dy.values))
        ky = ky / (2 * np.pi * self.config.alattice * self.config.slab_medium_index)

        # Calculate the magnitude of the FFT
        magnitude = np.abs(fft_data)

        if (plot_bool):
            # Plot the FFT magnitude
            plt.figure(figsize=(8, 6))
            plt.pcolormesh(kx, ky, magnitude, shading='auto', cmap='viridis')

            # Plot light circle
            phis = np.linspace(0, np.pi * 2, 101)
            radius = (1 / self.config.slab_medium_index)
            plt.plot(radius * np.cos(phis), radius * np.sin(phis), lw=2, color="w")

            plt.colorbar(label='Magnitude')
            plt.xlabel('$k_x/ (2 \pi /a)$')
            plt.ylabel('$k_y/ (2 \pi /a)$')
            plt.title('2D FFT Magnitude')
            plt.xlim(-4, 4)
            plt.ylim(-4, 4)
            # plt.grid(True)
            plt.show()

        return magnitude, (kx, ky)

    def plot_noise_analysis_data(self, quality_factor, overlap, loss_function,
                                 plot_title):

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
        plt.rcParams.update({'font.size': 12})

        cmap = plt.colormaps.get_cmap('Greens')

        ax1.hist(quality_factor,
                 bins=10,
                 facecolor=cmap(0.4),
                 edgecolor=cmap(0.8),
                 rwidth=0.8)
        ax1.set_ylabel('Frequency')
        ax1.set_xlabel('Quality Factor')
        ax1.set_xlim(np.min(quality_factor) * 0.9, np.max(quality_factor) * 1.05)

        ax2.hist(overlap,
                 bins=10,
                 facecolor=cmap(0.4),
                 edgecolor=cmap(0.8),
                 rwidth=0.8)
        ax2.set_xlabel('Directionality')
        ax2.set_xlim(np.min(overlap) * 0.95, np.max(overlap) * 1.05)
        ax2.axvline(x=overlap[0], color='k', linestyle='--')
        ax2.set_title(
            f'Median Q = {np.round(np.median(quality_factor[1:]),1)}, Mean $\eta$ = {np.round(np.median(directionality_array[1:]), 2)}',
            fontsize=11)

        ax3.hist(loss_function,
                 bins=10,
                 facecolor=cmap(0.4),
                 edgecolor=cmap(0.8),
                 rwidth=0.8)
        ax3.axvline(wavelength[0], color='k', linestyle='--')
        ax3.set_xlabel('Wavelength')

        plt.suptitle(
            f'{weights_name}_sigma_r = {sigma_r*1e3}, sigma_xy = {sigma_xy*1e3}, NA=0.55',
            fontsize=12)
        plt.savefig(
            f'./L3_520nm_noise analysis/Tidy3D analysis/Figures/{weights_name}_modified_sigma_r={sigma_r}_sigma_xy={sigma_xy}_histogram.png'
        )
        plt.show()
