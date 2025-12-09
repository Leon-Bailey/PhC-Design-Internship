import copy, yaml
import matplotlib.pyplot as plt
import numpy as np
import tidy3d as td
import tidy3d.web as web
import gdstk as gds

from Optimize_designs import Optimize
from tidy3d_utilities import Utilities, update_config_file
import ConfigurationClass


class NanobeamSimulation(Utilities, Optimize):
    """
    Simulation class for Nanobeam photonic crystal
    """
    def __init__(self, wavelength_in_at, design_args):

        self.wavelength_in_at = wavelength_in_at
        self.config_filename = 'NanobeamConfig'

        if self.wavelength_in_at in "950, 1330":
            name = rf".\ConfigFiles\nanobeam\nanobeam_parameters_{self.wavelength_in_at}nm.yaml"
            with open(name, 'r') as stream:
                default_config = yaml.safe_load(stream)
        else:
            raise ValueError("Supported wavelengths: 950 nm and 1330 nm")

        self.config = update_config_file(
            getattr(ConfigurationClass, self.config_filename), default_config,
            design_args)

        self.define_nanobeam_geometry()

        super().__init__()

    def tapered_lattice_constant(self, index, taper_function="linear"):
        """
        Taper the lattice constant as per a defined tapering scheme
        """

        if (taper_function == "linear"):
            if (index <= self.config.num_taper):
                lattice_constant = (self.config.acenter + (index) *
                                    (self.config.alattice - self.config.acenter) /
                                    self.config.num_taper)
            else:
                lattice_constant = self.config.alattice

        elif taper_function == "parabolic":
            if (index <= self.config.num_taper):
                lattice_constant = (self.config.acenter + (index)**2 *
                                    (self.config.alattice - self.config.acenter) /
                                    self.config.num_taper**2)
            else:
                lattice_constant = self.config.alattice

        else:
            raise ValueError("Current supported taper functions: linear, parabolic")

        return lattice_constant

    def define_nanobeam_geometry(self):
        """
        Define the Tidy3D nanobeam geometry given the design arguments
        """
        def nanobeam_taper_polygon(self):

            if self.config.slab_dimensions[0] == td.inf:
                raise ValueError("Slab length cannot be infinite while adding a "
                                 "taper polygon")

            cell = gds.Cell("taper")

            path = gds.RobustPath((self.config.slab_dimensions[0] / 2 +
                                   self.config.taper_polygon_length, 0),
                                  self.taper_polygon_tip_width)
            path.segment((self.config.slab_dimensions[0] / 2, 0),
                         self.config.slab_dimensions[1])

            cell.add(path)

            taper_geometry = td.PolySlab.from_gds(
                cell,
                axis=2,
                gds_layer=0,
                slab_bounds=(-self.config.slab_dimensions[2] / 2,
                             self.config.slab_dimensions[2] / 2),
                sidewall_angle=0,
            )

            self.taper_polygon = td.Structure(geometry=taper_geometry[0],
                                              medium=self.slab_medium)

            self.structures.append(self.taper_polygon)

        # Initialize list for simulation structures. Add the additional structures.
        self.structures = []
        self.structures.extend(self.config.structures)

        # Define permittivity of the slab and holes
        self.slab_medium = td.Medium(permittivity=self.config.slab_medium_index**2)
        self.hole_medium = td.Medium(permittivity=self.config.hole_medium_index**2)

        # Initialise the slab structure
        self.slab = td.Structure(
            geometry=td.Box(center=[0, 0, 0], size=self.config.slab_dimensions),
            medium=self.slab_medium,
        )
        self.structures.append(self.slab)

        # Unwrap hole optimization parameters
        dx = self.config.hole_optimization_parameters[0]
        dr = self.config.hole_optimization_parameters[1]

        # Define hole geometry
        holes_group = []

        # Design tapering hole geometry
        hole_center_position = (self.config.acenter + self.config.spacer) / 2

        for i in range(self.config.num_holes_right):
            if i < self.config.num_holes_optimize:
                hole_center = [hole_center_position + dx[i], 0, 0]
                hole_radius = self.config.radius_ratio * self.tapered_lattice_constant(
                    i) + dr[i]
            else:
                hole_center = [hole_center_position, 0, 0]
                hole_radius = self.config.radius_ratio * self.tapered_lattice_constant(
                    i)
            holes_group.append(
                td.Cylinder(
                    center=hole_center,
                    radius=hole_radius,
                    length=self.config.slab_dimensions[2],
                ))

            hole_center_position += 0.5 * (self.tapered_lattice_constant(i) +
                                           self.tapered_lattice_constant(i + 1))

        hole_center_position = -(self.config.acenter + self.config.spacer) / 2
        for i in range(self.config.num_holes_left):
            if i < self.config.num_holes_optimize:
                hole_center = [hole_center_position - dx[i], 0, 0]
                hole_radius = self.config.radius_ratio * self.tapered_lattice_constant(
                    i) + dr[i]
            else:
                hole_center = [hole_center_position, 0, 0]
                hole_radius = self.config.radius_ratio * self.tapered_lattice_constant(
                    i)

            holes_group.append(
                td.Cylinder(
                    center=hole_center,
                    radius=hole_radius,
                    length=self.config.slab_dimensions[2],
                ))

            hole_center_position -= 0.5 * (self.tapered_lattice_constant(i) +
                                           self.tapered_lattice_constant(i + 1))
        self.holes = td.Structure(geometry=td.GeometryGroup(geometries=holes_group),
                                  medium=self.hole_medium)
        self.structures.append(self.holes)

        if self.config.taper_polygon is True:
            nanobeam_taper_polygon(self)

        self.parse_wavelength_range()  # Initialize wavelength, freq range for sim
        self.apodization()  # Define apodization

    def calculate_gradient(self,
                           nanobeam_design_options,
                           simulation_options,
                           delta=5e-3,
                           optimize_holes=True,
                           verbose=False):
        # Make batch data for hole position +- delta, hole radius +- delta,
        # and device
        gradient_batch = {}

        for i in range(self.config.num_holes_optimize):
            name = f"hole {i} + dx"
            nanobeam_design_options_temp = copy.deepcopy(nanobeam_design_options)
            nanobeam_design_options_temp["hole_optimization_parameters"][0][
                i] += delta
            self.define_nanobeam_geometry(nanobeam_design_options_temp)
            gradient_batch[name] = self.make_simulation(simulation_options)

            name = f"hole {i} - dx"
            nanobeam_design_options_temp = copy.deepcopy(nanobeam_design_options)
            nanobeam_design_options_temp["hole_optimization_parameters"][0][
                i] -= delta
            self.define_nanobeam_geometry(nanobeam_design_options_temp)
            gradient_batch[name] = self.make_simulation(simulation_options)

        if (optimize_holes):
            for i in range(self.config.num_holes_optimize):
                name = f"hole {i} + dr"
                nanobeam_design_options_temp = copy.deepcopy(nanobeam_design_options)
                nanobeam_design_options_temp["hole_optimization_parameters"][1][
                    i] += delta
                self.define_nanobeam_geometry(nanobeam_design_options_temp)
                gradient_batch[name] = self.make_simulation(simulation_options)

                name = f"hole {i} - dr"
                nanobeam_design_options_temp = copy.deepcopy(nanobeam_design_options)
                nanobeam_design_options_temp["hole_optimization_parameters"][1][
                    i] -= delta
                self.define_nanobeam_geometry(nanobeam_design_options_temp)
                gradient_batch[name] = self.make_simulation(simulation_options)

        name = "device"
        self.define_nanobeam_geometry(nanobeam_design_options)
        gradient_batch[name] = self.make_simulation(simulation_options)

        batch_data = self.run_batch_simulation(gradient_batch, verbose=verbose)

        return self.analyze_gradient_batch_data(batch_data)

    def analyze_gradient_batch_data(
        self,
        batch_data,
    ):
        size = len(batch_data.task_paths) // 2
        quality_factor_array = np.zeros((size, 2))
        resonance_wavelength_array = np.zeros((size, 2))

        for idx, (task_name, sim_data) in enumerate(batch_data.items()):
            self.analyze_FieldTimeMonitor(
                sim_data,
                freq_range=self.freq_range,
                plot_bool=False,
                print_data_bool=False,
            )
            if idx == 2 * size:
                quality_factor = self.quality_factor
                resonant_wavelength = self.resonant_wavelength
                print(f"Cavity Resonance at {self.resonant_wavelength} nm; "
                      f"Q = {np.round(self.quality_factor)}")
            else:
                quality_factor_array[idx // 2, idx % 2] = self.quality_factor
                resonance_wavelength_array[idx // 2,
                                           idx % 2] = self.resonant_wavelength

        return (
            quality_factor,
            resonant_wavelength,
            quality_factor_array,
            resonance_wavelength_array,
        )

    def bandstructure(self, kx, run_time=10, FieldMonitor_bool=False):
        """
        Code similar to:
        https://docs.flexcompute.com/projects/tidy3d/en/v2.7.3/notebooks/Bandstructure.html
        """

        # Define unit cell for nanobeam
        hole = td.Structure(
            geometry=td.Cylinder(
                center=[0, 0, 0],
                radius=self.alattice * self.radius_ratio,
                length=self.slab_dimensions[2],
                axis=2,
            ),
            medium=self.hole_medium,
        )

        unit_cell = [self.slab, hole]
        unit_cell_size = (
            self.alattice,
            self.slab_dimensions[1],
            self.slab_dimensions[2],
        )

        boundary_spec = []

        for k in kx:
            boundary_spec.append(
                td.BoundarySpec(x=td.Boundary.bloch(k),
                                y=td.Boundary.pml(),
                                z=td.Boundary.pml()))

        # Define simulation size
        spacing = 2 * self.wavelength_range[-1]  # space between PhC and PML
        sim_size = (
            self.alattice,
            self.slab_dimensions[1] + spacing,
            self.slab_dimensions[2] + spacing,
        )

        batch_data = self.calculate_bandstructure(
            unit_cell=unit_cell,
            unit_cell_size=unit_cell_size,
            boundary_spec=boundary_spec,
            sim_size=sim_size,
            run_time=run_time * 1e-12,
            FieldMonitor_bool=FieldMonitor_bool)

        return batch_data

    def made_gds(self, save_file_name, log_file_name=None):

        if (log_file_name is not None):
            raise NotImplementedError()

        # The GDSII file is called a library, which contains multiple cells.
        lib = gds.Library()

        # Geometry must be placed in cells.
        cell = lib.new_cell("nanobeam")

        if self.config.slab_dimensions[0] == td.inf:
            raise ValueError(
                "Slab length cannot be infinite while making a GDS file")

        # gdstk.rectangle syntax: gdstk.rectangle(corner1, corner2, layer=0)
        slab = gds.rectangle((-self.config.slab_dimensions[0] / 2,
                              self.config.slab_dimensions[1] / 2),
                             (self.config.slab_dimensions[0] / 2,
                              -self.config.slab_dimensions[1] / 2),
                             layer=0)
        cell.add(slab)

        # Unwrap hole optimization parameters
        dx = self.config.hole_optimization_parameters[0]
        dr = self.config.hole_optimization_parameters[1]

        # Design tapering hole geometry
        hole_center_position = (self.config.acenter + self.config.spacer) / 2
        for i in range(self.config.num_holes_right):
            if i < self.config.num_holes_optimize:
                hole_center = hole_center_position + dx[i]
                hole_radius = self.config.radius_ratio * self.tapered_lattice_constant(
                    self, i) + dr[i]
            else:
                hole_center = hole_center_position
                hole_radius = self.config.radius_ratio * self.tapered_lattice_constant(
                    i)

            ellipse = gds.ellipse((hole_center, 0),
                                  hole_radius,
                                  tolerance=1e-4,
                                  layer=1)

            cell.add(ellipse)

            hole_center_position += 0.5 * (self.tapered_lattice_constant(i) +
                                           self.tapered_lattice_constant(i + 1))

        hole_center_position = -(self.config.acenter + self.config.spacer) / 2
        for i in range(self.config.num_holes_left):
            if i < self.config.num_holes_optimize:
                hole_center = hole_center_position - dx[i]
                hole_radius = self.config.radius_ratio * self.tapered_lattice_constant(
                    self, i) + dr[i]
            else:
                hole_center = hole_center_position
                hole_radius = self.config.radius_ratio * self.tapered_lattice_constant(
                    i)

            ellipse = gds.ellipse((hole_center, 0),
                                  hole_radius,
                                  tolerance=1e-4,
                                  layer=1)

            cell.add(ellipse)

            hole_center_position -= 0.5 * (self.tapered_lattice_constant(i) +
                                           self.tapered_lattice_constant(i + 1))

        if self.config.taper_polygon is True:
            path = gds.RobustPath((self.config.slab_dimensions[0] / 2 +
                                   self.config.taper_polygon_length, 0),
                                  self.config.taper_polygon_tip_width)
            path.segment((self.config.slab_dimensions[0] / 2, 0),
                         self.config.slab_dimensions[1])

            cell.add(path)

        lib.write_gds(save_file_name)

class L3Simulation(Utilities):
    def __init__(self, wavelength_in_at, design_args):
        self.wavelength_in_at = wavelength_in_at

        self.config_filename = 'L3Config'

        if self.wavelength_in_at in "440, 520, 780, 910, 950, 1290, 1320":
            name = rf".\ConfigFiles\L3\L3_parameters_{self.wavelength_in_at}nm.yaml"
            with open(name, 'r') as stream:
                default_config = yaml.safe_load(stream)
        else:
            raise ValueError(
                "Supported wavelengths: 440, 520, 780, 910, 950, 1290, and 1320 nm")

        self.config = update_config_file(
            getattr(ConfigurationClass, self.config_filename), default_config,
            design_args)

        self.define_L3_geometry()

        super().__init__()

    def define_L3_geometry(self):
        def load_file(self):

            param_history = np.load(self.config.param_file_name, allow_pickle=True)
            param_history = param_history.reshape(
                param_history.shape[0] // (2 * len(self.y_pos)), 2 * len(self.y_pos))
            parameters = param_history[-1]

            dx = parameters[0:len(self.x_pos)] * self.config.alattice
            dy = parameters[len(self.x_pos):] * self.config.alattice

            return (dx, dy)

        def L3_hole_positions(self, Nx, Ny):

            x_pos, y_pos = [], []

            for iy in range(Ny // 2 + 1):
                for ix in range(Nx // 2 + 1):
                    if (ix == 0 and iy == 0
                            or ix == 1 and iy == 0):  # skip holes for L3 cavity
                        continue
                    else:
                        x_pos.append((ix + (iy % 2) * 0.5) * self.config.alattice)
                        y_pos.append((iy * np.sqrt(3) / 2) * self.config.alattice)

            return (x_pos, y_pos)

        def resizeL3(self):

            nx_orig, ny_orig = self.config.Nx // 2 + 1, self.config.Ny // 2 + 1
            nx_final, ny_final = self.config.Nx_final // 2 + 1, self.config.Ny_final // 2 + 1

            self.x_pos, self.y_pos = L3_hole_positions(self, self.config.Nx_final,
                                                       self.config.Ny_final)

            dx_temp = np.zeros(nx_orig * ny_orig)
            dy_temp = np.zeros(nx_orig * ny_orig)

            dx_temp[2:] = self.dx
            dy_temp[2:] = self.dy

            dx_temp = np.reshape(dx_temp, (nx_orig, ny_orig))
            dy_temp = np.reshape(dy_temp, (nx_orig, ny_orig))

            dx_temp = np.pad(dx_temp, (0, nx_final - nx_orig))
            dy_temp = np.pad(dy_temp, (0, ny_final - ny_orig))

            dx_temp = np.reshape(dx_temp, (1, nx_final * ny_final))
            dy_temp = np.reshape(dy_temp, (1, nx_final * ny_final))

            self.dx = dx_temp[0, 2:]
            self.dy = dy_temp[0, 2:]

        def noise_arrays(self):

            # The arrays are of size (# of holes, 4) because in each for loop for
            # design_noisy_phc, atmost 1 hole is placed in each quadrant
            # Exceptions are the holes in x,y axis but as long as I use the same
            # code to define the PhC everywhere, things should be consistent
            r_noise_array = np.random.normal(loc=0,
                                             scale=self.config.sigma_r,
                                             size=(len(self.x_pos), 4))
            x_noise_array = np.random.normal(loc=0,
                                             scale=self.config.sigma_xy,
                                             size=(len(self.x_pos), 4))
            y_noise_array = np.random.normal(loc=0,
                                             scale=self.config.sigma_xy,
                                             size=(len(self.x_pos), 4))

            # Append everyhting in one array
            self.config.noise_arrays = [x_noise_array, y_noise_array, r_noise_array]

        # Initialize list for simulation structures
        self.structures = []

        # Define permittivity of the slab
        self.slab_medium = td.Medium(permittivity=self.config.slab_medium_index**2)

        # Get hexagonal grid of holes in one quadrant to initialize L3 cavity
        (self.x_pos, self.y_pos) = L3_hole_positions(self, self.config.Nx,
                                                     self.config.Ny)

        # Add displacements to the holes from param_file_name
        if self.config.param_file_name is None:
            (self.dx, self.dy) = np.zeros(len(self.x_pos)), np.zeros(len(self.y_pos))
        else:
            (self.dx, self.dy) = load_file(self)

        # Resize L3 cavity and hole displacements from (Nx, Ny) -> (Nx_final, Ny_final)
        if (self.config.Nx != self.config.Nx_final
                and self.config.Ny != self.config.Ny_final):
            resizeL3(self)

        self.x_pos[0] += self.config.x_shift * self.config.alattice

        # Get noise arrays for hole positions and radius if not already initialized
        if not self.config.noise_arrays:
            noise_arrays(self)

        # Define permittivity of the slab and holes
        self.slab_medium = td.Medium(permittivity=self.config.slab_medium_index**2)
        self.hole_medium = td.Medium(permittivity=self.config.hole_medium_index**2)

        # Initialize structures
        self.slab = td.Structure(geometry=td.Box(center=[0, 0, 0],
                                                 size=(td.inf, td.inf,
                                                       self.config.slab_thickness)),
                                 medium=self.slab_medium)
        self.structures.append(self.slab)

        holes_group = []

        # Get the position of holes for L3 cavity
        nx, ny = self.config.Nx // 2 + 1, self.config.Ny // 2 + 1

        for ic, x in enumerate(self.x_pos):
            yc = self.y_pos[ic] if self.y_pos[
                ic] == 0 else self.y_pos[ic] + self.dy[ic]
            xc = x if x == 0 else self.x_pos[ic] + self.dx[ic]

            if self.config.bandfolding and ic in self.config.bandfolding_hole_index:
                hole_radius = (self.config.radius_ratio +
                               self.config.bandfolding_dr) * self.config.alattice
            else:
                hole_radius = self.config.radius_ratio * self.config.alattice

            x_noise = self.config.noise_arrays[0][ic]
            y_noise = self.config.noise_arrays[1][ic]
            r_noise = self.config.noise_arrays[2][ic]

            holes_group.append(
                td.Cylinder(center=[xc + x_noise[0], yc + y_noise[0], 0],
                            radius=hole_radius + r_noise[0],
                            length=self.config.slab_thickness,
                            sidewall_angle=self.config.sidewall_angle))
            if (nx - self.config.alattice / 2 + 0.1 > self.x_pos[ic] > 0) and (
                (ny - self.config.alattice + 0.1) * np.sqrt(3) / 2 > self.y_pos[ic] >
                    0):
                holes_group.append(
                    td.Cylinder(center=[-xc + x_noise[1], -yc + y_noise[1], 0],
                                radius=hole_radius + r_noise[1],
                                length=self.config.slab_thickness,
                                sidewall_angle=self.config.sidewall_angle))
            if (nx - 1.5 * self.config.alattice + 0.1 > self.x_pos[ic] > 0):
                holes_group.append(
                    td.Cylinder(center=[-xc + x_noise[2], yc + y_noise[2], 0],
                                radius=hole_radius + r_noise[2],
                                length=self.config.slab_thickness,
                                sidewall_angle=self.config.sidewall_angle))
            if ((ny - self.config.alattice + 0.1) * np.sqrt(3) / 2 > self.y_pos[ic] >
                    0) and (nx - self.config.alattice + 0.1 > self.x_pos[ic]):
                holes_group.append(
                    td.Cylinder(center=[xc + x_noise[3], -yc + y_noise[3], 0],
                                radius=hole_radius + r_noise[3],
                                length=self.config.slab_thickness,
                                sidewall_angle=self.config.sidewall_angle))

        self.holes = td.Structure(geometry=td.GeometryGroup(geometries=holes_group),
                                  medium=self.hole_medium)
        self.structures.append(self.holes)

        self.parse_wavelength_range()  # Initialize wavelength, freq range for sim
        self.apodization()  # Define apodization

    def make_gds(self,
                 filename,
                 hole_diameter_array,
                 alattice_array,
                 design_label="L3",
                 array_size_x=3,
                 array_size_y=3,
                 tolerance=1e-4):

        # The GDSII file is called a library, which contains multiple cells.
        lib = gds.Library()

        # Geometry must be placed in cells.
        cell = lib.new_cell("L3_cavty_design")

        nx, ny = self.config.Nx_final // 2 + 1, self.config.Ny_final // 2 + 1

        for index_hole in range(len(hole_diameter_array)):
            shift_x_main = 50 * (index_hole)
            hole_radius = hole_diameter_array[index_hole] / 2

            for index_alattice in range(len(alattice_array)):
                shift_y_main = 40 * index_alattice
                alattice = alattice_array[index_alattice]
                shift_x = shift_x_main

                for ix in range(array_size_x):
                    shift_x = shift_x + 15 * (ix > 0)
                    shift_y = shift_y_main

                    for iy in range(array_size_y):
                        shift_y = shift_y + 15 * (iy > 0)

                        # Apply holes symmetrically in the four quadrants
                        for ic, x in enumerate(self.x_pos):

                            yc = self.y_pos[ic] if self.y_pos[
                                ic] == 0 else self.y_pos[ic] + self.dy[ic]
                            xc = x if x == 0 else self.x_pos[ic] + self.dx[ic]
                            circle = gds.ellipse((xc + shift_x, yc + shift_y),
                                                 hole_radius,
                                                 tolerance=tolerance,
                                                 layer=0)
                            cell.add(circle)

                            if (nx - alattice / 2 + 0.1 > self.x_pos[ic] > 0) and (
                                (ny - alattice + 0.1) * np.sqrt(3) / 2 >
                                    self.y_pos[ic] > 0):
                                circle = gds.ellipse((-xc + shift_x, -yc + shift_y),
                                                     hole_radius,
                                                     tolerance=tolerance,
                                                     layer=0)
                            cell.add(circle)

                            if (nx - 1.5 * alattice + 0.1 > self.x_pos[ic] > 0):
                                circle = gds.ellipse((-xc + shift_x, yc + shift_y),
                                                     hole_radius,
                                                     tolerance=tolerance,
                                                     layer=0)
                            cell.add(circle)

                            if ((ny - alattice + 0.1) * np.sqrt(3) / 2 >
                                    self.y_pos[ic] > 0) and (nx - alattice + 0.1 >
                                                             self.x_pos[ic]):
                                circle = gds.ellipse((xc + shift_x, -yc + shift_y),
                                                     hole_radius,
                                                     tolerance=tolerance,
                                                     layer=0)
                            cell.add(circle)

                        if (ix == array_size_x % 2 and iy == array_size_y - 1):
                            text = gds.text(
                                f"{design_label} {int(hole_diameter_array[index_hole]*1e3)}",
                                size=5,
                                position=(shift_x - 9, shift_y + 5))
                            for i in range(len(text)):
                                cell.add(text[i])

                        if (ix == 0 and iy == 1):
                            text = gds.text(
                                f"A{int(alattice_array[index_alattice]*1e3)}",
                                size=5,
                                position=(shift_x - 11, shift_y + 7),
                                vertical=True)
                            for i in range(len(text)):
                                cell.add(text[i])

        lib.write_oas(f"./GDS_files/{filename}.gds")


class BullseyeSimulation(Utilities, Optimize):
    def __init__(self, wavelength_in_at, design_args):

        self.wavelength_in_at = wavelength_in_at

        if self.wavelength_in_at in "950":
            name = rf".\ConfigFiles\bullseye\bullseye_parameters_{self.wavelength_in_at}nm.yaml"
            with open(name, 'r') as stream:
                default_config = yaml.safe_load(stream)
        else:
            raise ValueError("Supported wavelengths: 950 nm")

        self.config = update_config_file(BullseyeConfig, default_config, design_args)

        self.define_bullseye_geometry(design_args)

        super().__init__()

    def define_bullseye_geometry(self, design_args):
        def load_file(self):

            param_file_name = rf'.\Bullseye_950nm_weights\{self.config.param_file_name}.npy'
            training_params = np.load(param_file_name, allow_pickle=True)
            epochs = len(training_params[-1])

            if (self.optimize_radius is False):
                param_history = training_params[0].reshape(epochs, -1)
            else:
                param_history = training_params[0].reshape(epochs, 2, -1)

            self.config.num_optimize_gratings = param_history.shape[-1]
            self.config.grating_optimization_params = param_history[-1]
            print(self.config.grating_optimization_params)

        def make_partial_ring(radius, width, start_angle, end_angle):

            cell = gds.Cell("partial_ring")  # define a gds cell

            # define a path
            partial_ring = gds.RobustPath(
                (radius * np.cos(start_angle), radius * np.sin(start_angle)),
                width=width,
                tolerance=1e-5,
                layer=1,
            )

            partial_ring.arc(radius, start_angle, end_angle)
            cell.add(partial_ring)  # add path to the cell

            partial_ring_geo = td.PolySlab.from_gds(
                cell,
                gds_layer=1,
                axis=2,
                slab_bounds=(-self.config.slab_thickness / 2,
                             self.config.slab_thickness / 2),
            )

            return partial_ring_geo

        def calculate_grating_angles(self):
            """
            Function to calculate the beginning and end angles for 
            a bulleye cavity with given number of bridges and bridge angle

            Returns:
                start_angles, end_angles: Angle arrays for the rings in radians
            """

            start_angles = (
                180 / self.config.num_bridges + self.config.bridge_angle +
                np.arange(self.config.num_bridges) * 360 / self.config.num_bridges -
                45)

            end_angles = (start_angles + 360 / self.config.num_bridges -
                          2 * self.config.bridge_angle)

            return start_angles * np.pi / 180, end_angles * np.pi / 180

        if self.wavelength_in_at == "950":
            self.bullseye_parameters_950(**design_args)
        else:
            raise ValueError("Supported wavelengths: 950")

        # Define permittivity of the slab
        self.slab_medium = td.Medium(permittivity=self.config.slab_medium_index**2)

        # Load parameters from saved file
        if self.param_file_name is not None:
            load_file(self)

        # Define permittivity of the slab and holes
        self.slab_medium = td.Medium(permittivity=self.config.slab_medium_index**2)
        self.grating_medium = td.Medium(
            permittivity=self.config.grating_medium_index**2)

        # Initialize structures
        self.slab = td.Structure(geometry=td.Box(center=(0, 0, 0),
                                                 size=(td.inf, td.inf,
                                                       self.config.slab_thickness)),
                                 medium=self.slab_medium)

        # Start and end angles for grating rings given the bridges
        start_angles, end_angles = calculate_grating_angles(self)

        # initialize an empty list to store structures for simulations
        self.structures = []
        self.structures.extend(self.config.structures)

        gratings_geometry = None

        for grating in range(self.config.num_gratings):

            if grating < self.config.num_optimize_gratings:
                ring_radius = (grating * self.config.grating_period +
                               self.config.central_radius +
                               self.config.grating_width / 2 +
                               self.config.grating_optimization_params[1][grating])

                ring_width = (self.config.grating_width +
                              self.config.grating_optimization_params[0][grating])
            else:
                ring_radius = (grating * self.config.grating_period +
                               self.config.central_radius +
                               self.config.grating_width / 2)

                ring_width = self.config.grating_width

            for bridge in range(self.config.num_bridges):
                partial_ring = make_partial_ring(radius=ring_radius,
                                                 width=ring_width,
                                                 start_angle=start_angles[bridge],
                                                 end_angle=end_angles[bridge])

                gratings_geometry = (partial_ring if gratings_geometry is None else
                                     gratings_geometry + partial_ring)

        # Make the geometry for bullseye cavity
        self.structures.append(
            td.Structure(geometry=td.GeometryGroup(geometries=gratings_geometry),
                         medium=self.grating_medium))

    def make_simulation(self, simulation_options):
        """
        Initialize the nanobeam simulation object in Tidy3d and plot the simulation
        """

        # Update config file for simulation options
        self.config = update_config_file(BullseyeSimulation, self.config,
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

    def analyze_gradient_batch_data(self, batch_data):

        size = len(batch_data.task_paths) // 2
        quality_factor_array = np.zeros((size, 2))
        resonance_wavelength_array = np.zeros((size, 2))

        for idx, (task_name, sim_data) in enumerate(batch_data.items()):
            self.analyze_FieldTimeMonitor(
                sim_data,
                freq_range=self.freq_range,
                plot_bool=False,
                print_data_bool=False,
            )
            if idx == 2 * size:
                quality_factor = self.quality_factor
                resonant_wavelength = self.resonant_wavelength
                print(f"Cavity Resonance at {self.resonant_wavelength} nm; "
                      f"Q = {np.round(self.quality_factor)}")
            else:
                quality_factor_array[idx // 2, idx % 2] = self.quality_factor
                resonance_wavelength_array[idx // 2,
                                           idx % 2] = self.resonant_wavelength

        return (quality_factor, resonant_wavelength, quality_factor_array,
                resonance_wavelength_array)

    def calculate_gradient(self,
                           bullseye_design_options,
                           simulation_options,
                           delta=5e-3,
                           optimize_radius=False,
                           verbose=False):

        gradient_batch = {}

        for i in range(self.num_optimize_gratings):
            name = f"grating width {i} + dx"
            bullseye_design_options_temp = copy.deepcopy(bullseye_design_options)
            bullseye_design_options_temp["grating_optimization_params"][0][
                i] += delta
            self.define_bullseye_geometry(bullseye_design_options_temp)
            gradient_batch[name] = self.make_simulation(simulation_options)

            name = f"grating width {i} - dx"
            bullseye_design_options_temp = copy.deepcopy(bullseye_design_options)
            bullseye_design_options_temp["grating_optimization_params"][0][
                i] -= delta
            self.define_bullseye_geometry(bullseye_design_options_temp)
            gradient_batch[name] = self.make_simulation(simulation_options)

        if (optimize_radius):
            for i in range(self.num_optimize_gratings):
                name = f"radius {i} + dr"
                bullseye_design_options_temp = copy.deepcopy(bullseye_design_options)
                bullseye_design_options_temp["grating_optimization_params"][1][
                    i] += delta
                self.define_bullseye_geometry(bullseye_design_options_temp)
                gradient_batch[name] = self.make_simulation(simulation_options)

                name = f"radius {i} - dr"
                bullseye_design_options_temp = copy.deepcopy(bullseye_design_options)
                bullseye_design_options_temp["grating_optimization_params"][1][
                    i] -= delta
                self.define_bullseye_geometry(bullseye_design_options_temp)
                gradient_batch[name] = self.make_simulation(simulation_options)

        name = "device"
        self.define_bullseye_geometry(bullseye_design_options)
        gradient_batch[name] = self.make_simulation(simulation_options)

        batch_data = self.run_batch_simulation(gradient_batch, verbose=verbose)

        return self.analyze_gradient_batch_data(batch_data)


class SawfishSimulation(Utilities):
    def __init__(self, wavelength_in_at, **design_args):
        self.wavelength_in_at = wavelength_in_at

        self.define_sawfish_geometry(**design_args)

    def sawfish_parameters_650(
            self,
            slab_thickness=0.266,
            width=11e-3,
            alattice=0.2385,
            height=130e-3,
            num_unit_cells_right=15,
            num_unit_cells_left=15,
            num_unit_cells_taper=4,
            taper_alattice=[0.2, 0.207, 0.2179, 0.2284],
            slab_medium_index=2.41,
            wavelength_range=np.linspace(0.55, 0.65, 50),
    ):
        # Define dimentions
        self.slab_thickness = slab_thickness
        self.width = width
        self.alattice = alattice
        self.height = height
        self.num_unit_cells_right = num_unit_cells_right
        self.num_unit_cells_left = num_unit_cells_left
        self.num_unit_cells_taper = num_unit_cells_taper

        if max(num_unit_cells_left, num_unit_cells_right) < num_unit_cells_taper:
            raise ValueError("# of tapering unit cells should be smaller or equal "
                             "to the unit cells on atleast one side")

        if len(taper_alattice) != num_unit_cells_taper:
            raise ValueError("# of tapering cells is not equal to the length of "
                             "taper_alattice")
        else:
            self.taper_alattice = taper_alattice

        # Define permittivity of the slab and holes
        self.slab_medium = td.Medium(permittivity=slab_medium_index**2)

        # Define wavelength and frequency range of interest
        self.wavelength_range = wavelength_range

        self.freq0 = td.C_0 / np.mean(self.wavelength_range)
        self.freq_range = td.C_0 / self.wavelength_range
        self.fwidth = (self.freq_range[0] - self.freq_range[-1]) / 2

    def sawfish_unit_cell(self, center, alattice, height, width=None):
        """
        function for creating "sawfish" unit cells
        """
        yArray = []
        numPoints = 5001

        if width is None:
            width = self.width

        # defining the vertices for the PolySlab
        xaxis = np.linspace(center[0] - alattice / 2, center[0] + alattice / 2,
                            numPoints)
        xArray = np.linspace(-alattice / 2, alattice / 2, numPoints)

        yArray = (height * ((np.cos(np.pi / 2 + (np.pi / alattice) * xArray))**6) +
                  width)
        yArray2 = (-height * ((np.cos(np.pi / 2 +
                                      (np.pi / alattice) * np.flip(xArray)))**6) -
                   width)

        X = np.concatenate([xaxis, np.flip(xaxis)])
        Y = np.concatenate([yArray, yArray2])
        arr = np.array([X, Y]).T

        geometry = td.PolySlab(
            vertices=arr,
            slab_bounds=(-self.slab_thickness / 2, self.slab_thickness / 2),
            axis=2,
        )

        return geometry

    def define_sawfish_geometry(self, **design_args):
        if self.wavelength_in_at == "650":
            self.sawfish_parameters_650(**design_args)
        else:
            raise ValueError("Supported Wavelengths: 650")

        sawfish_geometry = None
        x_center = 0

        if self.taper_alattice is not None:
            for idx in range(self.num_unit_cells_right):
                if idx < self.num_unit_cells_taper:
                    new_unit_cell = self.sawfish_unit_cell(
                        center=[x_center + self.taper_alattice[idx] / 2, 0],
                        alattice=self.taper_alattice[idx],
                        height=self.height,
                    )

                    x_center = x_center + self.taper_alattice[idx]
                else:
                    new_unit_cell = self.sawfish_unit_cell(
                        center=[x_center + self.alattice / 2, 0],
                        alattice=self.alattice,
                        height=self.height,
                    )
                    x_center = x_center + self.alattice

                # adding the unit cell to the geometry (or initializing the variable
                # if is the first unit cell)
                sawfish_geometry = (new_unit_cell if sawfish_geometry is None else
                                    sawfish_geometry + new_unit_cell)

            x_center = 0
            for idx in range(self.num_unit_cells_left):
                if idx < self.num_unit_cells_taper:
                    new_unit_cell = self.sawfish_unit_cell(
                        center=[x_center - self.taper_alattice[idx] / 2, 0],
                        alattice=self.taper_alattice[idx],
                        height=self.height,
                    )

                    x_center = x_center - self.taper_alattice[idx]
                else:
                    new_unit_cell = self.sawfish_unit_cell(
                        center=[x_center - self.alattice / 2, 0],
                        alattice=self.alattice,
                        height=self.height,
                    )
                    x_center = x_center - self.alattice

                sawfish_geometry += new_unit_cell

        # TODO: add a way to generate the structure without the need to define a taper
        else:
            raise NotImplementedError()

        sawfish_structure = td.Structure(geometry=sawfish_geometry,
                                         medium=self.slab_medium)

        self.sawfish_structure = sawfish_structure
        return sawfish_geometry

    def simulation_run_options(
        self,
        symmetry=[1, -1, 1],
        background_medium=td.Medium(),
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML()),
        sim_center=[0, 0, 0],
        run_time=1e-12,
        source_location=[0, 0, 0],
        source_polarization="Ey",
        sim_size=[6, 2, 2],
        mesh_grid_size=10,
        FieldTimeMonitor_bool=True,
        FieldMonitor_bool=True,
        FieldMonitorSize=[2, 2, 0],
        FarFieldMonitor_bool=False,
        path_dir="data",
        task_name="sawfish_simulation",
        verbose=True,
        plot_bool=True,
    ):
        """
        Set misc parameters for FDTD simulation
        """

        # Define apodization
        self.apodization

        if FieldTimeMonitor_bool + FieldMonitor_bool + FarFieldMonitor_bool == 0:
            raise ValueError("Add Monitors!")
        monitors = []
        if FieldTimeMonitor_bool:
            monitors.append(self.FieldTimeMonitor())
        if FieldMonitor_bool:
            monitors.append(self.FieldMonitor(size=FieldMonitorSize))
        if FarFieldMonitor_bool:
            monitors.append(self.FarFieldMonitor())

        self._sim_run_options = {
            "symmetry": symmetry,
            "background_medium": background_medium,
            "boundary_spec": boundary_spec,
            "run_time": run_time,
            "source_location": source_location,
            "source_polarization": source_polarization,
            "sim_center": sim_center,
            "sim_size": sim_size,
            "mesh_grid_size": mesh_grid_size,
            "monitors": monitors,
            "path_dir": path_dir,
            "task_name": task_name,
            "verbose": verbose,
            "plot_bool": plot_bool,
        }

        # Also store the options as separate attributes
        for option, value in self._sim_run_options.items():
            # Set all the options as class attributes
            setattr(self, option, value)

    def make_simulation(self, **kwargs):
        """
        Initialize the sawfish simulation object in Tidy3d and plot the simulation
        """

        self.simulation_run_options(**kwargs)

        sim = td.Simulation(
            size=self.sim_size,
            center=self.sim_center,
            symmetry=self.symmetry,
            grid_spec=self.get_mesh_grid(self.mesh_grid_size),
            run_time=self.run_time,
            medium=self.background_medium,
            sources=[
                self.dipole_source(location=self.source_location,
                                   polarization=self.source_polarization)
            ],
            monitors=self.monitors,
            boundary_spec=self.boundary_spec,
            structures=[self.sawfish_structure],
        )

        if self.plot_bool:
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            sim.plot(z=0, ax=ax[0])
            sim.plot(y=0, ax=ax[1])
            plt.show()

        return sim

    def run_simulation(self, simulation):
        """
        Run Tidy3D nanobeam simulation
        """

        sim_data = web.run(simulation,
                           task_name=self.task_name,
                           verbose=self.verbose)
        return sim_data


class L3PMSimulation(Utilities):
    def __init__(self, wavelength_in_at, design_args):
        self.wavelength_in_at = wavelength_in_at
        self.config_name = 'L3PMConfig'

        if self.wavelength_in_at in "1320":
            name = rf".\ConfigFiles\L3PM\L3PM_parameters_{self.wavelength_in_at}nm.yaml"
            with open(name, 'r') as stream:
                default_config = yaml.safe_load(stream)
        else:
            raise ValueError("Supported wavelengths: 1320 nm")

        self.config = update_config_file(
            getattr(ConfigurationClass, self.config_filename), default_config,
            design_args)

        self.define_L3PM_geometry()

        super().__init__()

    def define_L3PM_geometry(self):
        def load_file(self):

            param_history = np.load(self.config.param_file_name, allow_pickle=True)
            param_history = param_history.reshape(
                param_history.shape[0] // (2 * len(self.y_pos)), 2 * len(self.y_pos))
            parameters = param_history[-1]

            dx = parameters[0:len(self.x_pos)] * self.config.alattice
            dy = parameters[len(self.x_pos):] * self.config.alattice

            return (dx, dy)

        def L3PM_hole_positions(self, Nx, Ny):

            x_pos, y_pos = [], []

            if self.config.distance % 2 == 0:
                self.skip_holes = self.config.distance // 2

                for iy in range(Ny // 2 + 1):
                    for ix in range(Nx // 2 + 1):
                        if (iy == 0 and
                            (ix >= self.skip_holes and
                             ix < self.skip_holes + 3)):  # skip holes for L3 cavity
                            continue
                        else:
                            x_pos.append(
                                ((ix + 0.5) - (iy % 2) * 0.5) * self.config.alattice)
                            y_pos.append(
                                (iy * np.sqrt(3) / 2) * self.config.alattice)

            else:
                self.skip_holes = (self.config.distance + 1) // 2

                for iy in range(Ny // 2 + 1):
                    for ix in range(Nx // 2 + 1):
                        if (iy == 0 and
                            (ix >= self.skip_holes and
                             ix < self.skip_holes + 3)):  # skip holes for L3 cavity
                            continue
                        else:
                            x_pos.append(
                                (ix + (iy % 2) * 0.5) * self.config.alattice)
                            y_pos.append(
                                (iy * np.sqrt(3) / 2) * self.config.alattice)

            return (x_pos, y_pos)

        def resizeL3PM(self):

            nx_orig, ny_orig = self.config.Nx // 2 + 1, self.config.Ny // 2 + 1
            nx_final, ny_final = self.config.Nx_final // 2 + 1, self.config.Ny_final // 2 + 1

            self.x_pos, self.y_pos = L3PM_hole_positions(self, self.config.Nx_final,
                                                         self.config.Ny_final)

            dx_temp = np.zeros(nx_orig * ny_orig)
            dy_temp = np.zeros(nx_orig * ny_orig)

            dx_temp[0:2], dx_temp[5:] = self.dx[0:2], self.dx[2:]
            dy_temp[0:2], dy_temp[5:] = self.dy[0:2], self.dy[2:]

            dx_temp = np.reshape(dx_temp, (nx_orig, ny_orig))
            dy_temp = np.reshape(dy_temp, (nx_orig, ny_orig))

            dx_temp = np.pad(dx_temp, (0, nx_final - nx_orig))
            dy_temp = np.pad(dy_temp, (0, ny_final - ny_orig))

            dx_temp = np.reshape(dx_temp, (1, nx_final * ny_final))
            dy_temp = np.reshape(dy_temp, (1, nx_final * ny_final))

            dx, dy = np.zeros_like(self.x_pos), np.zeros_like(self.x_pos)

            dx[0:2], dx[2:] = dx_temp[0, 0:2], dx_temp[0, 5:]
            dy[0:2], dy[2:] = dy_temp[0, 0:2], dy_temp[0, 5:]

            self.dx, self.dy = dx, dy

        def noise_arrays(self):

            # The arrays are of size (# of holes, 4) because in each for loop for
            # design_noisy_phc, atmost 1 hole is placed in each quadrant
            # Exceptions are the holes in x,y axis but as long as I use the same
            # code to define the PhC everywhere, things should be consistent
            r_noise_array = np.random.normal(loc=0,
                                             scale=self.config.sigma_r,
                                             size=(len(self.x_pos), 4))
            x_noise_array = np.random.normal(loc=0,
                                             scale=self.config.sigma_xy,
                                             size=(len(self.x_pos), 4))
            y_noise_array = np.random.normal(loc=0,
                                             scale=self.config.sigma_xy,
                                             size=(len(self.x_pos), 4))

            # Append everyhting in one array
            self.config.noise_arrays = [x_noise_array, y_noise_array, r_noise_array]

        # Initialize list for simulation structures
        self.structures = []

        # Define permittivity of the slab
        self.slab_medium = td.Medium(permittivity=self.config.slab_medium_index**2)

        # Get hexagonal grid of holes in one quadrant to initialize L3 cavity
        (self.x_pos, self.y_pos) = L3PM_hole_positions(self, self.config.Nx,
                                                       self.config.Ny)

        # Add displacements to the holes from param_file_name
        if self.config.param_file_name is None:
            (self.dx, self.dy) = np.zeros(len(self.x_pos)), np.zeros(len(self.y_pos))
        else:
            (self.dx, self.dy) = load_file(self)

        # Resize L3 cavity and hole displacements from (Nx, Ny) -> (Nx_final, Ny_final)
        if (self.config.Nx != self.config.Nx_final
                and self.config.Ny != self.config.Ny_final):
            resizeL3PM(self)

        self.x_pos[self.skip_holes - 1] -= self.config.x_shift * self.config.alattice
        self.x_pos[self.skip_holes] += self.config.x_shift * self.config.alattice

        # Get noise arrays for hole positions and radius if not already initialized
        if not self.config.noise_arrays:
            noise_arrays(self)

        # Define permittivity of the slab and holes
        self.slab_medium = td.Medium(permittivity=self.config.slab_medium_index**2)
        self.hole_medium = td.Medium(permittivity=self.config.hole_medium_index**2)

        # Initialize structures
        self.slab = td.Structure(geometry=td.Box(center=[0, 0, 0],
                                                 size=(td.inf, td.inf,
                                                       self.config.slab_thickness)),
                                 medium=self.slab_medium)
        self.structures.append(self.slab)

        holes_group = []

        # Get the position of holes for L3 cavity
        nx, ny = self.config.Nx // 2 + 1, self.config.Ny // 2 + 1

        for ic, x in enumerate(self.x_pos):
            yc = self.y_pos[ic] if self.y_pos[
                ic] == 0 else self.y_pos[ic] + self.dy[ic]
            xc = x if x == 0 else self.x_pos[ic] + self.dx[ic]

            hole_radius = self.config.radius_ratio * self.config.alattice

            x_noise = self.config.noise_arrays[0][ic]
            y_noise = self.config.noise_arrays[1][ic]
            r_noise = self.config.noise_arrays[2][ic]

            holes_group.append(
                td.Cylinder(center=[xc + x_noise[0], yc + y_noise[0], 0],
                            radius=hole_radius + r_noise[0],
                            length=self.config.slab_thickness,
                            sidewall_angle=self.config.sidewall_angle))
            if (nx - self.config.alattice / 2 + 0.1 > self.x_pos[ic] > 0) and (
                (ny - self.config.alattice + 0.1) * np.sqrt(3) / 2 > self.y_pos[ic] >
                    0):
                holes_group.append(
                    td.Cylinder(center=[-xc + x_noise[1], -yc + y_noise[1], 0],
                                radius=hole_radius + r_noise[1],
                                length=self.config.slab_thickness,
                                sidewall_angle=self.config.sidewall_angle))
            if (nx - 1.5 * self.config.alattice + 0.1 > self.x_pos[ic] > 0):
                holes_group.append(
                    td.Cylinder(center=[-xc + x_noise[2], yc + y_noise[2], 0],
                                radius=hole_radius + r_noise[2],
                                length=self.config.slab_thickness,
                                sidewall_angle=self.config.sidewall_angle))
            if ((ny - self.config.alattice + 0.1) * np.sqrt(3) / 2 > self.y_pos[ic] >
                    0) and (nx - self.config.alattice + 0.1 > self.x_pos[ic]):
                holes_group.append(
                    td.Cylinder(center=[xc + x_noise[3], -yc + y_noise[3], 0],
                                radius=hole_radius + r_noise[3],
                                length=self.config.slab_thickness,
                                sidewall_angle=self.config.sidewall_angle))

        self.holes = td.Structure(geometry=td.GeometryGroup(geometries=holes_group),
                                  medium=self.hole_medium)
        self.structures.append(self.holes)

        self.parse_wavelength_range()  # Initialize wavelength, freq range for sim
        self.apodization()  # Define apodization


class HeteroSimulation(Utilities):
    def __init__(self, wavelength_in_at, design_args):
        self.wavelength_in_at = wavelength_in_at
        self.config_filename = 'HeteroConfig'

        if self.wavelength_in_at in "1550":
            name = rf".\ConfigFiles\Hetero\Hetero_parameters_{self.wavelength_in_at}nm.yaml"
            with open(name, 'r') as stream:
                default_config = yaml.safe_load(stream)
        else:
            raise ValueError("Supported wavelengths: 1550 nm")

        self.config = update_config_file(
            getattr(ConfigurationClass, self.config_filename), default_config,
            design_args)

        self.define_Hetero_geometry()

        super().__init__()

    def define_Hetero_geometry(self):
        def load_file(self):

            param_history = np.load(self.config.param_file_name, allow_pickle=True)
            param_history = param_history.reshape(
                param_history.shape[0] // (2 * len(self.y_pos)), 2 * len(self.y_pos))
            parameters = param_history[-1]

            dx = parameters[0:len(self.x_pos)] * self.config.alattice
            dy = parameters[len(self.x_pos):] * self.config.alattice

            return (dx, dy)

        def Hetero_hole_positions_row(self, Nx, Ny):

            x_row, x_row_shifted = [], []

            for ix in range(Nx // 2 + 1):
                if ix == 0 or ix == 1:
                    x_row.append(ix * self.config.alattice_2)
                else:
                    x_row.append(x_row[-1] + self.config.alattice)

                if ix == 0:
                    x_row_shifted.append(0.5 * self.config.alattice_2)
                elif ix == 1:
                    x_row_shifted.append(
                        ix * (self.config.alattice + self.config.alattice_2) / 2 +
                        self.config.alattice * 0.5)
                else:
                    x_row_shifted.append(x_row_shifted[-1] + self.config.alattice)

        def Hetero_hole_positions(self, Nx, Ny):

            x_pos, y_pos, x_pos_down, y_pos_down = [], [], [], []

            for iy in range(Ny // 2 + 1):

                if iy == 0:
                    continue

                for ix in range(Nx // 2 + 1):
                    if (iy % 2) == 0:
                        if ix <= 1:
                            x_pos.append(ix * self.config.alattice_2)
                        else:
                            x_pos.append(x_pos[-1] + self.config.alattice)
                    else:
                        if ix < 1:
                            x_pos.append((ix + 0.5) * self.config.alattice_2)
                        elif ix == 1:
                            x_pos.append(
                                ix *
                                (self.config.alattice + self.config.alattice_2) / 2 +
                                self.config.alattice * 0.5)
                        else:
                            x_pos.append(x_pos[-1] + self.config.alattice)

                    y_pos.append((iy * np.sqrt(3) / 2) * self.config.alattice)

            # x_pos_down = x_pos.copy()
            # y_pos_down = y_pos.copy()

            for iy in range(Ny // 2 + 1):

                if iy == 0:
                    continue

                for ix in range(Nx // 2 + 1):
                    if ((iy + 1) % 2) == 0:
                        if ix <= 1:
                            x_pos_down.append(ix * self.config.alattice_2)
                        else:
                            x_pos_down.append(x_pos_down[-1] + self.config.alattice)
                    else:
                        if ix < 1:
                            x_pos_down.append((ix + 0.5) * self.config.alattice_2)
                        elif ix == 1:
                            x_pos_down.append(
                                ix *
                                (self.config.alattice + self.config.alattice_2) / 2 +
                                self.config.alattice * 0.5)
                        else:
                            x_pos_down.append(x_pos_down[-1] + self.config.alattice)

                    y_pos_down.append((iy * np.sqrt(3) / 2) * self.config.alattice)

            y_pos = y_pos - 0.125 * np.sqrt(3) * self.config.alattice
            y_pos_down = y_pos_down - 0.125 * np.sqrt(3) * self.config.alattice

            return (x_pos, y_pos, x_pos_down, y_pos_down)

        def resizeHetero(self):

            nx_orig, ny_orig = self.config.Nx // 2 + 1, self.config.Ny // 2 + 1
            nx_final, ny_final = self.config.Nx_final // 2 + 1, self.config.Ny_final // 2 + 1

            self.x_pos, self.y_pos = Hetero_hole_positions(self,
                                                           self.config.Nx_final,
                                                           self.config.Ny_final)

            dx_temp = np.zeros(nx_orig * ny_orig)
            dy_temp = np.zeros(nx_orig * ny_orig)

            dx_temp[2:] = self.dx
            dy_temp[2:] = self.dy

            dx_temp = np.reshape(dx_temp, (nx_orig, ny_orig))
            dy_temp = np.reshape(dy_temp, (nx_orig, ny_orig))

            dx_temp = np.pad(dx_temp, (0, nx_final - nx_orig))
            dy_temp = np.pad(dy_temp, (0, ny_final - ny_orig))

            dx_temp = np.reshape(dx_temp, (1, nx_final * ny_final))
            dy_temp = np.reshape(dy_temp, (1, nx_final * ny_final))

            self.dx = dx_temp[0, 2:]
            self.dy = dy_temp[0, 2:]

        def noise_arrays(self):

            # The arrays are of size (# of holes, 4) because in each for loop for
            # design_noisy_phc, atmost 1 hole is placed in each quadrant
            # Exceptions are the holes in x,y axis but as long as I use the same
            # code to define the PhC everywhere, things should be consistent
            r_noise_array = np.random.normal(loc=0,
                                             scale=self.config.sigma_r,
                                             size=(len(self.x_pos), 4))
            x_noise_array = np.random.normal(loc=0,
                                             scale=self.config.sigma_xy,
                                             size=(len(self.x_pos), 4))
            y_noise_array = np.random.normal(loc=0,
                                             scale=self.config.sigma_xy,
                                             size=(len(self.x_pos), 4))

            # Append everyhting in one array
            self.config.noise_arrays = [x_noise_array, y_noise_array, r_noise_array]

        # Initialize list for simulation structures
        self.structures = []

        # Define permittivity of the slab
        self.slab_medium = td.Medium(permittivity=self.config.slab_medium_index**2)

        # Get hexagonal grid of holes in one quadrant to initialize Hetero cavity
        (self.x_pos, self.y_pos, self.x_pos_down,
         self.y_pos_down) = Hetero_hole_positions(self, self.config.Nx,
                                                  self.config.Ny)

        # Add displacements to the holes from param_file_name
        if self.config.param_file_name is None:
            (self.dx, self.dy) = np.zeros(len(self.x_pos)), np.zeros(len(self.y_pos))
        else:
            (self.dx, self.dy) = load_file(self)

        # Resize Hetero cavity and hole displacements from (Nx, Ny) -> (Nx_final, Ny_final)
        if (self.config.Nx != self.config.Nx_final
                and self.config.Ny != self.config.Ny_final):
            resizeHetero(self)

        # Get noise arrays for hole positions and radius if not already initialized
        if not self.config.noise_arrays:
            noise_arrays(self)

        # Define permittivity of the slab and holes
        self.slab_medium = td.Medium(permittivity=self.config.slab_medium_index**2)
        self.hole_medium = td.Medium(permittivity=self.config.hole_medium_index**2)

        # Initialize structures
        self.slab = td.Structure(geometry=td.Box(center=[0, 0, 0],
                                                 size=(td.inf, td.inf,
                                                       self.config.slab_thickness)),
                                 medium=self.slab_medium)
        self.structures.append(self.slab)

        holes_group = []

        # Get the position of holes for Hetero cavity
        nx, ny = self.config.Nx // 2 + 1, self.config.Ny // 2 + 1

        for ic, x in enumerate(self.x_pos):
            yc = self.y_pos[ic] if self.y_pos[ic] == 0 else (self.y_pos[ic] +
                                                             self.dy[ic])
            xc = x if x == 0 else self.x_pos[ic] + self.dx[ic]

            hole_radius = self.config.radius_ratio * self.config.alattice

            x_noise = self.config.noise_arrays[0][ic]
            y_noise = self.config.noise_arrays[1][ic]
            r_noise = self.config.noise_arrays[2][ic]

            holes_group.append(
                td.Cylinder(center=[xc + x_noise[0], yc + y_noise[0], 0],
                            radius=hole_radius + r_noise[0],
                            length=self.config.slab_thickness,
                            sidewall_angle=self.config.sidewall_angle))

            if (nx - 1.5 * self.config.alattice + 0.1 > self.x_pos[ic] > 0):
                holes_group.append(
                    td.Cylinder(center=[-xc + x_noise[2], yc + y_noise[2], 0],
                                radius=hole_radius + r_noise[2],
                                length=self.config.slab_thickness,
                                sidewall_angle=self.config.sidewall_angle))

        for ic, x in enumerate(self.x_pos_down):
            yc = self.y_pos_down[ic] if self.y_pos_down[ic] == 0 else (
                self.y_pos_down[ic] + self.dy[ic])
            xc = x if x == 0 else self.x_pos_down[ic] + self.dx[ic]

            if (nx - self.config.alattice / 2 + 0.1 > self.x_pos_down[ic] > 0) and (
                (ny - self.config.alattice + 0.1) * np.sqrt(3) / 2 >
                    self.y_pos_down[ic] > 0):
                holes_group.append(
                    td.Cylinder(center=[-xc + x_noise[1], -yc + y_noise[1], 0],
                                radius=hole_radius + r_noise[1],
                                length=self.config.slab_thickness,
                                sidewall_angle=self.config.sidewall_angle))

            if ((ny - self.config.alattice + 0.1) * np.sqrt(3) / 2 >
                    self.y_pos_down[ic] > 0) and (nx - self.config.alattice + 0.1 >
                                                  self.x_pos_down[ic]):
                holes_group.append(
                    td.Cylinder(center=[xc + x_noise[3], -yc + y_noise[3], 0],
                                radius=hole_radius + r_noise[3],
                                length=self.config.slab_thickness,
                                sidewall_angle=self.config.sidewall_angle))

        self.holes = td.Structure(geometry=td.GeometryGroup(geometries=holes_group),
                                  medium=self.hole_medium)
        self.structures.append(self.holes)

        self.parse_wavelength_range()  # Initialize wavelength, freq range for sim
        self.apodization()  # Define apodization