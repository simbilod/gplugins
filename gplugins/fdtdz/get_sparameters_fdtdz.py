from typing import Any, Callable

import gdsfactory as gf
import jax.numpy as jnp
import numpy as np
from gdsfactory.component import Component
from gdsfactory.technology import LayerStack
from pjz._field import SimParams, scatter, field
import matplotlib.pyplot as plt
from gdsfactory.typings import Float2

from gplugins.fdtdz.get_epsilon_fdtdz import (
    component_to_epsilon_pjz,
    material_name_to_fdtdz,
    create_physical_grid,
)
from gplugins.fdtdz.get_ports_fdtdz import get_mode_port



def get_simulation_fdtdz(
    component: Component,
    layerstack: LayerStack,
    nm_per_pixel: int = 20,
    zmin: float = -0.75,
    zz: int = 96,
    tt: int = 10000,
    wavelength_start: float = 1.5,
    wavelength_stop: float = 1.6,
    port_margin: float = 1,
    material_name_to_fdtdz: dict = material_name_to_fdtdz,
    default_index: float = 1.44,
) -> tuple:
    """
    This function returns fdtdz simulations settings from a gdsfactory Component and LayerStack.

    Args:
        component (Component): The component to be simulated.
        layerstack (LayerStack, optional): The layerstack for the simulation.
        nm_per_pixel (int, optional): The number of nanometers per pixel. Defaults to 20.
        zmin (float, optional): The minimum z value. Defaults to -0.75.
        zz (int, optional): The z value. Defaults to 96.
        tt (int, optional): The time value. Defaults to 10000.
        wavelength_start (float, optional): The starting wavelength. Defaults to 1.5.
        wavelength_stop (float, optional): The stopping wavelength. Defaults to 1.6.
            The center wavelength is the average of wavelength_start and wavelength_stop, and the output is only fully accurate there.
        port_margin (float, optional): How much of the cross-sectional epsilon to keep beyond the ports. Defaults to 1.
        material_name_to_fdtdz (dict, optional): The dictionary mapping material names to fdtdz values. Defaults to material_name_to_fdtdz.
        default_index (float, optional): The default index. Defaults to 1.44.

    Returns:
        tuple: A tuple containing information to launch pjz simulation: epsilon, omega, excitations, positions, SimParams(tt=tt, omega_range=omega_range)
    """

    optical_portnames = list(component.get_ports_dict(port_type="optical").keys())

    assert isinstance(
        component, Component
    ), f"component needs to be a gf.Component, got Type {type(component)}"


    # Create epsilon distribution
    epsilon = component_to_epsilon_pjz(
        component=component,
        layerstack=layerstack,
        zmin=zmin,
        material_name_to_index=material_name_to_fdtdz,
        default_index=default_index,
    )

    # Setup modes sources/monitors
    wavelength = (wavelength_start + wavelength_stop) / 2
    omega = jnp.array([2 * jnp.pi / (wavelength*1E3 / nm_per_pixel)])
    print(omega)
    omega_range = (2 * jnp.pi / (wavelength_stop*1E3 / nm_per_pixel),
                   2 * jnp.pi / (wavelength_start*1E3 / nm_per_pixel))
    excitations = []
    positions = []
    for portname in optical_portnames:
        excitation, pos, epsilon_port = get_mode_port(
            omega=omega,
            port=component.ports[portname],
            epsilon=epsilon,
            xmin=component.xmin,
            ymin=component.ymin,
            nm_per_pixel=nm_per_pixel,
            port_extent_xy=port_margin,
        )
        excitations.append(excitation)
        positions.append(pos)

    return epsilon, omega, excitations, positions, SimParams(tt=tt, omega_range=omega_range), component.xmin, component.ymin


def get_sparameters_fdtdz(
    component: Component,
    layerstack: LayerStack = None,
    nm_per_pixel: int = 20,
    zmin: float = -0.75,
    zz: int = 96,
    tt: int = 10000,
    wavelength_start: float = 1.5,
    wavelength_stop: float = 1.6,
    port_margin: float = 1,
    material_name_to_fdtdz: dict = material_name_to_fdtdz,
    default_index: float = 1.44,
    run: bool = True,
):
    """
    This function returns fdtdz S-parameters from a gdsfactory Component and LayerStack.

    Args:
        component (Component): The component to be simulated.
        layerstack (LayerStack, optional): The layerstack for the simulation.
        nm_per_pixel (int, optional): The number of nanometers per pixel. Defaults to 20.
        zmin (float, optional): The minimum z value. Defaults to -0.75.
        zz (int, optional): The z value. Defaults to 96.
        tt (int, optional): The time value. Defaults to 10000.
        wavelength_start (float, optional): The starting wavelength. Defaults to 1.5.
        wavelength_stop (float, optional): The stopping wavelength. Defaults to 1.6.
            The center wavelength is the average of wavelength_start and wavelength_stop, and the output is only fully accurate there.
        port_margin (float, optional): How much of the cross-sectional epsilon to keep beyond the ports. Defaults to 1.
        material_name_to_fdtdz (dict, optional): The dictionary mapping material names to fdtdz values. Defaults to material_name_to_fdtdz.
        default_index (float, optional): The default index. Defaults to 1.44.

    Returns:
        tuple: A tuple containing information to launch pjz simulation: epsilon, omega, excitations, positions, SimParams(tt=tt, omega_range=omega_range)
    """
    epsilon, omega, excitations, positions, sim_params, _, _ = get_simulation_fdtdz(
        component,
        layerstack,
        nm_per_pixel,
        zmin,
        zz,
        tt,
        wavelength_start,
        wavelength_stop,
        port_margin,
        material_name_to_fdtdz,
        default_index,
    )
    sout = scatter(
        epsilon=epsilon,
        omega=jnp.array([omega]),
        modes=tuple(excitations),
        pos=tuple(positions),
        sim_params=sim_params,
    )

    # # Format S-parameters into gdsfactory format
    # for port_name in port_names:
    #     key = f"{port_name}@0,{port_source_name}@0"
    #     sp[key] = monitor_exiting / source_entering

    # if bool(port_symmetries):
    #     for key, symmetries in port_symmetries.items():
    #         for sym in symmetries:
    #             if key in sp:
    #                 sp[sym] = sp[key]

    return sout



def plot_fields(
    component: Component,
    layerstack: LayerStack,
    nm_per_pixel: int = 20,
    zmin: float = -0.75,
    zz: int = 96,
    tt: int = 10000,
    wavelength_start: float = 1.5,
    wavelength_stop: float = 1.6,
    port_margin: float = 1,
    material_name_to_fdtdz: dict = material_name_to_fdtdz,
    default_index: float = 1.44,
    x: int | float = 0.0,
    y: int | float = None,
    z: int | float = None,
    figsize: Float2 = (11, 4),
    field_func: Callable = jnp.real,
    v: float = 3E-1,
):
    
    source_ports = component.get_ports_list(port_type="optical")
    
    epsilon, omega, excitations, positions, sim_params, xmin, ymin = get_simulation_fdtdz(
        component,
        layerstack,
        nm_per_pixel,
        zmin,
        zz,
        tt,
        wavelength_start,
        wavelength_stop,
        port_margin,
        material_name_to_fdtdz,
        default_index,
    )

    # Checks
    if (x and y) or (y and z) or (x and z):
        raise ValueError("Only one of x, y or z must be numeric!")

    # Create physical grid
    xarray, yarray, zarray = create_physical_grid(
        xmin, ymin, zmin, epsilon, nm_per_pixel
    )

    fig, ax = plt.subplots(nrows = len(excitations), ncols = 4, figsize=figsize)

    # Plot
    if x is not None:
        plt.title(f"Fields at x = {x}")
        x_index = int(
            np.where(np.isclose(xarray, x, atol=nm_per_pixel * 1e-3 / 2))[0][0] / 2
        )  # factor of 2 from Yee grid?
        for i, (excitation, position) in enumerate(zip(excitations, positions)):
            # Draw epsilon
            ax[i, 0].imshow(
                epsilon[0, x_index, :, :].transpose(),
                origin="lower",
                extent=[yarray[0], yarray[-1], zarray[0], zarray[-1]],
                vmin=np.min(epsilon),
                vmax=np.max(epsilon),
            )
            # Draw fields
            f = field_func(field(epsilon, excitation, omega, position, sim_params))
            ax[i, 1].imshow(f[0, 0, x_index, :, :].T, cmap="bwr", vmin=-v, vmax=v, extent=[yarray[0], yarray[-1], zarray[0], zarray[-1]],)
            ax[i, 1].set_title("Ex")
            ax[i, 1].set_xlabel("y")
            ax[i, 1].set_ylabel("z")
            ax[i, 2].imshow(f[0, 1, x_index, :, :].T, cmap="bwr", vmin=-v, vmax=v, extent=[yarray[0], yarray[-1], zarray[0], zarray[-1]],)
            ax[i, 2].set_title("Ey")
            ax[i, 2].set_xlabel("y")
            ax[i, 2].set_ylabel("z")
            ax[i, 3].imshow(f[0, 2, x_index, :, :].T, cmap="bwr", vmin=-v, vmax=v, extent=[yarray[0], yarray[-1], zarray[0], zarray[-1]],)
            ax[i, 3].set_title("Ez")
            ax[i, 3].set_xlabel("y")
            ax[i, 3].set_ylabel("z")
    elif y is not None:
        plt.title(f"Fields at y = {y}")
        y_index = int(
            np.where(np.isclose(yarray, y, atol=nm_per_pixel * 1e-3 / 2))[0][0] / 2
        )  # factor of 2 from Yee grid?
        for i, (excitation, position) in enumerate(zip(excitations, positions)):
            # Draw epsilon
            ax[i, 0].imshow(
                epsilon[0, :, y_index, :].transpose(),
                origin="lower",
                extent=[yarray[0], yarray[-1], zarray[0], zarray[-1]],
                vmin=np.min(epsilon),
                vmax=np.max(epsilon),
            )
            f = field_func(field(epsilon, excitation, omega, position, sim_params))
            ax[i, 1].imshow(f[0, 0, :, y_index, :].T, cmap="bwr", vmin=-v, vmax=v, extent=[xarray[0], xarray[-1], zarray[0], zarray[-1]])
            ax[i, 1].set_title("Ex")
            ax[i, 1].set_xlabel("x")
            ax[i, 1].set_ylabel("z")
            ax[i, 2].imshow(f[0, 1, :, y_index, :].T, cmap="bwr", vmin=-v, vmax=v, extent=[xarray[0], xarray[-1], zarray[0], zarray[-1]])
            ax[i, 2].set_title("Ey")
            ax[i, 2].set_xlabel("x")
            ax[i, 2].set_ylabel("z")
            ax[i, 3].imshow(f[0, 2, :, y_index, :].T, cmap="bwr", vmin=-v, vmax=v, extent=[xarray[0], xarray[-1], zarray[0], zarray[-1]])
            ax[i, 3].set_title("Ez")
            ax[i, 3].set_xlabel("x")
            ax[i, 3].set_ylabel("z")
    elif z is not None:
        plt.title(f"Fields at z = {z}")
        z_index = int(
            np.where(np.isclose(zarray, z, atol=nm_per_pixel * 1e-3 / 2))[0][0] / 2
        )  # factor of 2 from Yee grid?
        for i, (excitation, position) in enumerate(zip(excitations, positions)):
            # Draw epsilon
            ax[i, 0].imshow(
                epsilon[0, :, :, z_index].transpose(),
                origin="lower",
                # extent=[yarray[0], yarray[-1], zarray[0], zarray[-1]],
                vmin=np.min(epsilon),
                vmax=np.max(epsilon),
            )
            # Draw source line
            source_port = source_ports[i]
            source_orientation = source_port.orientation
            source_x, source_y = source_port.center
            source_x = int(
                np.where(np.isclose(xarray, source_x, atol=nm_per_pixel * 1e-3 / 2))[0][0] / 2
                )
            source_y = int(
                np.where(np.isclose(yarray, source_y, atol=nm_per_pixel * 1e-3 / 2))[0][0] / 2
                )
            margin_pixels = port_margin * 1E3 / nm_per_pixel
            if source_orientation == 0 or source_orientation == 180:
                ax[i,0].plot([source_x, source_y - margin_pixels], [source_x, source_y + margin_pixels], color='red', linewidth=2)
            else:
                ax[i,0].plot([source_x - margin_pixels, source_y], [source_x + margin_pixels, source_y], color='red', linewidth=2)
            f = field_func(field(epsilon, excitation, omega, position, sim_params))
            ax[i, 1].imshow(f[0, 0, :, :, z_index].T, cmap="bwr", vmin=-v, vmax=v, extent=[xarray[0], xarray[-1], yarray[0], yarray[-1]])
            ax[i, 1].set_title("Ex")
            ax[i, 1].set_xlabel("x")
            ax[i, 1].set_ylabel("y")
            ax[i, 2].imshow(f[0, 1, :, :, z_index].T, cmap="bwr", vmin=-v, vmax=v, extent=[xarray[0], xarray[-1], yarray[0], yarray[-1]])
            ax[i, 2].set_title("Ey")
            ax[i, 2].set_xlabel("x")
            ax[i, 2].set_ylabel("y")
            ax[i, 3].imshow(f[0, 2, :, :, z_index].T, cmap="bwr", vmin=-v, vmax=v, extent=[xarray[0], xarray[-1], yarray[0], yarray[-1]])
            ax[i, 3].set_title("Ez")
            ax[i, 3].set_xlabel("x")
            ax[i, 3].set_ylabel("y")

    plt.show()
    plt.savefig("plot_fields.png")
    return fig


if __name__ == "__main__":
    from gdsfactory.generic_tech import LAYER, LAYER_STACK

    length = 5

    c = gf.Component()
    # waveguide = c << gf.components.straight(length=length, layer=LAYER.WG).extract(
    #     layers=(LAYER.WG,)
    # )
    waveguide = c << gf.components.bend_euler(radius=length, layer=LAYER.WG).extract(
        layers=(LAYER.WG,)
    )
    padding = c << gf.components.bbox(
        waveguide.bbox, top=2, bottom=2, layer=LAYER.WAFER
    )
    c.add_ports(gf.components.straight(length=length).get_ports_list())

    filtered_layerstack = LayerStack(
        layers={k: LAYER_STACK.layers[k] for k in ["clad", "box", "core"]}
    )

    fig = plot_fields(
        component=c,
        layerstack=filtered_layerstack,
        nm_per_pixel=20,
        zmin=-0.75,
        zz=96,
        tt=10000,
        wavelength_start=1.5,
        wavelength_stop=1.6,
        port_margin=1,
        x = None,
        y = None,
        z = 0.11,
        v=0.03
    )
    plt.savefig("plot_fields.png")