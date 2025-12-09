from dataclasses import dataclass, field
import tidy3d as td


@dataclass
class MonitorConfig:

    field_time_monitor: bool = True
    field_time_monitor_center: tuple = (0, 0, 0)
    field_time_monitor_array: bool = False
    num_field_time_monitor: int = 1
    field_time_monitor_span: tuple = (0, 0, 0)
    field_time_monitor_location_randomize: bool = True

    field_monitor: bool = False
    field_monitor_center: tuple = (0, 0, 0)
    field_monitor_size: tuple = (2, 2, 0)

    # eps_monitor center and size is set to be the same as field monitor
    eps_monitor: bool = False

    far_field_monitor: bool = False
    far_field_monitor_size: tuple = (td.inf, td.inf, 0)

    far_field_angle_monitor: bool = False
    far_field_angle_monitor_size: tuple = (td.inf, td.inf, 0)

    additional_monitors: list = field(default_factory=lambda: [])


@dataclass
class DipoleSourceConfig:

    dipole_center: tuple = (0, 0, 0)
    source_polarization: str = "Ey"
    dipole_phase: float = 0  # in radians

    # Settings for dipole cloud
    dipole_cloud: bool = False
    num_dipoles: int = 1
    dipole_cloud_span: tuple = (0, 0, 0)

    additional_sources: list = field(default_factory=lambda: [])


@dataclass
class SimulationConfig:

    # Simulation options
    symmetry: tuple = (1, -1, 1)
    background_medium: td.Medium = td.Medium()
    boundary_spec: td.BoundarySpec = td.BoundarySpec.all_sides(boundary=td.PML())
    sim_center: tuple = (0, 0, 0)
    run_time: float = 1  # in ps
    sim_size: tuple = (6, 2, 2)
    mesh_grid_size: float = 10

    # Save data and other options
    path_dir: str = "data"
    verbose: bool = True
    plot: bool = True

    # Additional structures for the simulation
    structures: list = field(default_factory=lambda: [])


@dataclass
class NanobeamConfig(MonitorConfig, DipoleSourceConfig, SimulationConfig):
    '''
    Default parameters are for 1350 nm resonance wavelength
    '''

    # Nanobeam slab options
    slab_dimensions: tuple = (10, 0.5, 0.22)
    alattice: float = 0.375
    num_taper: int = 4
    acenter: float = 0.270
    radius_ratio: float = 0.32
    spacer: float = 0
    num_holes_right: int = 15
    num_holes_left: int = 15
    slab_medium_index: float = 3.48
    hole_medium_index: float = 1
    wavelength_range: tuple = (1.3, 1.4, 50)
    task_name: str = "nanobeam_simulation"

    # Optimization params
    num_holes_optimize: int = 0
    hole_optimization_parameters: list = field(default_factory=lambda: [[], []])

    # Taper
    taper_polygon: bool = False
    taper_polygon_length: float = 12
    taper_polygon_tip_width: float = 0.15


@dataclass
class L3Config(MonitorConfig, DipoleSourceConfig, SimulationConfig):
    '''
    Default parameters are for 780 nm resonance wavelength
    '''

    Nx: int = 16
    Ny: int = 16
    Nx_final: int = 32
    Ny_final: int = 32
    alattice: float = 0.2
    radius_ratio: float = 0.3
    slab_thickness: float = 0.1
    x_shift: float = 0.2
    slab_medium_index: float = 3.38
    hole_medium_index: float = 1
    wavelength_range: tuple = (0.72, 0.8, 50)
    sidewall_angle: float = 0
    task_name: str = "L3_simulation"

    # load file containing hole displacements
    param_file_name: str = None

    # Enable bandfolding design
    bandfolding: bool = False
    bandfolding_hole_index: list = field(default_factory=lambda: [])
    bandfolding_dr: float = 0

    # gaussian widths for added random noise to hole position and radii
    sigma_xy: float = 0
    sigma_r: float = 0
    noise_arrays: list = field(default_factory=lambda: [])


@dataclass
class BullseyeConfig(MonitorConfig, DipoleSourceConfig, SimulationConfig):
    '''
    Default parameters are for 780 nm resonance wavelength
    '''

    num_gratings: int = 7
    central_grating: float = 0.37
    grating_period: float = 0.185
    grating_width: float = 0.1
    num_bridges: int = 4
    bridge_angle: float = 2.5  # in degrees
    slab_thickness: float = 0.18
    slab_medium_index: float = 3.55
    grating_medium_index: float = 1
    wavelength_range: tuple = (0.92, 1, 50)
    task_name = "Bullseye_simulation"

    # optimization params
    num_gratings_optimize: int = 0
    grating_optimization_params: list = field(default_factory=lambda: [[], []])
    optimize_radius: bool = False
    param_file_name: str = None

    # Additional structures for the simulation
    structures: list = field(default_factory=lambda: [])


@dataclass
class L3PMConfig(MonitorConfig, DipoleSourceConfig, SimulationConfig):
    '''
    Default parameters are for 1320 nm resonance wavelength
    '''

    Nx: int = 16
    Ny: int = 16
    Nx_final: int = 32
    Ny_final: int = 32
    alattice: float = 0.345
    radius_ratio: float = 0.3
    slab_thickness: float = 0.22
    x_shift: float = 0.17
    distance: int = 4
    slab_medium_index: float = 3.48
    hole_medium_index: float = 1
    wavelength_range: tuple = (1.3, 1.35, 50)
    sidewall_angle: float = 0
    task_name: str = "L3PM_simulation"

    # load file containing hole displacements
    param_file_name: str = None

    # gaussian widths for added random noise to hole position and radii
    sigma_xy: float = 0
    sigma_r: float = 0
    noise_arrays: list = field(default_factory=lambda: [])

@dataclass
class HeteroConfig(MonitorConfig, DipoleSourceConfig, SimulationConfig):
    '''
    Default parameters are for 1550 nm resonance wavelength
    '''

    Nx: int = 16
    Ny: int = 16
    Nx_final: int = 32
    Ny_final: int = 32
    alattice: float = 0.41
    alattice_2: float = 0.42
    radius_ratio: float = 0.3
    slab_thickness: float = 0.172
    slab_medium_index: float = 3.464
    hole_medium_index: float = 1
    wavelength_range: tuple = (0.9, 1, 50)
    sidewall_angle: float = 0
    task_name: str = "Hetero_simulation"

    # load file containing hole displacements
    param_file_name: str = None

    # gaussian widths for added random noise to hole position and radii
    sigma_xy: float = 0
    sigma_r: float = 0
    noise_arrays: list = field(default_factory=lambda: [])