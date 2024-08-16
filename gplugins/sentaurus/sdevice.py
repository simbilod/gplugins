import pathlib
from pathlib import Path

from gdsfactory.typings import Floats, Tuple

DEFAULT_OUTPUT_SETTINGS = """Plot{
  *--Density and Currents, etc
  eDensity hDensity
  TotalCurrent/Vector eCurrent/Vector hCurrent/Vector
  eMobility hMobility
  eVelocity hVelocity
  eQuasiFermi hQuasiFermi

  *--Temperature
  eTemperature Temperature * hTemperature

  *--Fields and charges
  ElectricField/Vector Potential SpaceCharge

  *--Doping Profiles
  Doping DonorConcentration AcceptorConcentration

  *--Generation/Recombination
  SRH Band2Band * Auger
  AvalancheGeneration eAvalancheGeneration hAvalancheGeneration eAlphaAvalanche hAlphaAvalanche

  *--Driving forces
  eGradQuasiFermi/Vector hGradQuasiFermi/Vector
  eEparallel hEparallel eENormal hENormal

  *--Band structure/Composition
  BandGap
  BandGapNarrowing
  Affinity
  ConductionBand ValenceBand

  *--Complex Refractive Index (changed by FCD)
  ComplexRefractiveIndex
}
"""

DEFAULT_PHYSICS_SETTINGS = """Physics{
    Mobility ( DopingDependence HighFieldSaturation Enormal )
    EffectiveIntrinsicDensity(BandGapNarrowing (OldSlotboom))
    Recombination( SRH Auger )
}
"""

PHYSICS_SETTINGS_AVALANCHE = """Physics{
    Mobility ( DopingDependence Enormal eHighFieldsaturation(GradQuasiFermi) hHighFieldsaturation(GradQuasiFermi) )
    EffectiveIntrinsicDensity(BandGapNarrowing (OldSlotboom))
Recombination( SRH Auger( WithGeneration ) Avalanche( vanOverstraetendeMan ) )
}
"""

DEFAULT_MATH_SETTINGS = """Math{
  Extrapolate
  RelErrControl
  Digits=5
  ErrReff(electron)= 1.0e7
  ErrReff(hole)    = 1.0e7
  Iterations=20
  Notdamped=100
   NumberOfThreads = 4
}
"""


def write_sdevice_quasistationary_ramp_voltage_dd(
    struct_in: str = "./sprocess/struct_out_fps.tdr",
    contacts: Tuple[str] = ("anode", "cathode", "substrate"),
    ramp_contact_name: str = "cathode",
    ramp_final_voltage: float = 1.0,
    ramp_initial_step: float = 0.01,
    ramp_increment: float = 1.3,
    ramp_max_step: float = 0.2,
    ramp_min_step: float = 1e-6,
    ramp_sample_voltages: Floats = (0.0, 0.3, 0.6, 0.8, 1.0),
    script_path: Path = Path("./sdevice_fps.cmd"),
    output_settings: str = DEFAULT_OUTPUT_SETTINGS,
    physics_settings: str = DEFAULT_PHYSICS_SETTINGS,
    math_settings: str = DEFAULT_MATH_SETTINGS,
    num_threads: int = 4,
    contact_resistance: float = 0,
):
    """Writes a Sentaurus Device TLC file for sweeping DC voltage of one terminal of a Sentaurus Structure (from sprocess or structure editor) using the drift-diffusion equations (Hole + Electrons + Poisson).

    You may need to modify the settings or this function itself for better results.

    Arguments:
        struct: Sentaurus Structure object file to run the simulation on.
        contacts: list of all contact names in the struct.
        ramp_contact_name: name of the contact whose voltage to sweep.
        ramp_final_voltage: final target voltage.
        ramp_initial_step: initial ramp step.
        ramp_increment: multiplying factor to increase ramp rate between iterations.
        ramp_max_step: maximum ramping step.
        ramp_min_step: minimum ramping step.
        ramp_sample_voltages: list of voltages between 0V and ramp_final_voltage to report.
        filepath: str = Path to the TLC file to be written.
        file_settings: "File" field settings to add to the TCL file
        output_settings: "Output" field settings to add to the TCL file
        physics_settings: "Physics" field settings to add to the TCL file
        math_settings: str = "Math" field settings to add to the TCL file
        initialization_commands: in the solver, what to execute before the ramp
        contact_resistance: in series with the ramped contact
    """

    # Setup TCL file
    script_path.parent.mkdir(parents=True, exist_ok=True)
    if script_path.exists():
        script_path.unlink()

    # Initialize electrodes
    with open(script_path, "a") as f:
        f.write("Electrode{\n")
        for boundary_name in contacts:
            f.write(f'{{ name="{boundary_name}"      voltage=0 Resist={contact_resistance}}}\n')
        f.write("}\n")

        f.write(
            f"""
File {{
  Grid = "{struct_in}"
  Plot = "./tdrdat_"
  Output = "./log_"
}}
    """
        )

        # Output settings
        f.write(output_settings)

        # Physics settings
        f.write(physics_settings)

        # Math settings
        f.write(math_settings)

        # Solve settings
        f.write("Solve{\n")

        # Initialization
        initialization_commands = f"""
            NewCurrentPrefix=\"./init\"
            Coupled(Iterations=100){{ Poisson }}
            Coupled{{ Poisson Electron Hole }}
        """

        f.write(initialization_commands)

        ramp_sample_voltages_str = ""
        for i, voltage in enumerate(ramp_sample_voltages):
            if i == 0:
                ramp_sample_voltages_str = f" {voltage:1.3f}"
            else:
                ramp_sample_voltages_str += f"; {voltage:1.3f}"

        f.write(
            f"""
    Quasistationary (
        InitialStep={ramp_initial_step} Increment={ramp_increment}
        MaxStep ={ramp_max_step} MinStep = {ramp_min_step}
        Goal{{ Name=\"{ramp_contact_name}\" Voltage={ramp_final_voltage} }}
    ){{ Coupled {{Poisson Electron Hole }}
        Save(FilePrefix=\"./sweep_save\" Time= ({ramp_sample_voltages_str} ) NoOverWrite )
        Plot(FilePrefix=\"./sweep_plot\" Time= ({ramp_sample_voltages_str} ) NoOverWrite )
    }}
    """
        )
        f.write("}\n")


def write_sdevice_ssac_ramp_voltage_dd(
    ramp_final_voltage: float = 3.0,
    device_name_extra_str="0",
    filename: str = "sdevice_fps.cmd",
    save_directory: Path = None,
    execution_directory: Path = None,
    struct: str = "./sprocess/struct_out_fps.tdr",
    num_threads: int = 4,
):
    save_directory = (
        Path("./sdevice/") if save_directory is None else Path(save_directory)
    )
    execution_directory = (
        Path("./") if execution_directory is None else Path(execution_directory)
    )

    relative_save_directory = save_directory.relative_to(execution_directory)
    struct = struct.relative_to(execution_directory)

    # Setup TCL file
    out_file = pathlib.Path(save_directory / filename)
    save_directory.mkdir(parents=True, exist_ok=True)
    if out_file.exists():
        out_file.unlink()

    Vstring = f"{ramp_final_voltage:1.3f}".replace(".", "p").replace("-", "m")

    text1 = f"""
Device PN_{Vstring}_{device_name_extra_str} {{

  Electrode {{
    {{ Name="anode" Voltage=0.0 }}
    {{ Name="cathode" Voltage=0.0 }}
    {{ Name="substrate" Voltage=0.0 }}
  }}
"""
    text3 = f"""
  Physics {{
    Mobility ( DopingDependence HighFieldSaturation Enormal )
    EffectiveIntrinsicDensity(BandGapNarrowing (OldSlotboom))
    Recombination( SRH Auger Avalanche )
  }}
  Plot {{
    eDensity hDensity eMobility hMobility eCurrent hCurrent
    ElectricField eEparallel hEparallel
    eQuasiFermi hQuasiFermi
    Potential Doping SpaceCharge
    DonorConcentration AcceptorConcentration
  }}
}}

Math {{
  Extrapolate
  RelErrControl
  Notdamped=50
  Iterations=20
  NumberOfThreads=4
}}

System {{
  PN_{Vstring}_{device_name_extra_str} trans (cathode=d anode=g substrate=g)
  Vsource_pset vg (g 0) {{dc=0}}
  Vsource_pset vd (d 0) {{dc=0}}
}}

File {{
  Grid = "{struct}"
  Current = "{str(relative_save_directory)}/plot_{Vstring}"
  Plot = "{str(relative_save_directory)}/tdrdat_{Vstring}"
  Output = "{str(relative_save_directory)}/log_{Vstring}"
  ACExtract = "{str(relative_save_directory)}/acplot_{Vstring}"
}}

Solve {{
  #-a) zero solution
  Poisson
  Coupled {{ Poisson Electron Hole }}
"""

    text4 = f"""#-b) ramp cathode
  Quasistationary (
  InitialStep=0.01 MaxStep=0.04 MinStep=1.e-5
  Goal {{ Parameter=vd.dc Voltage={ramp_final_voltage} }}
  )
"""

    text5 = """
  { ACCoupled (
  StartFrequency=1e3 EndFrequency=1e3
  NumberOfPoints=1 Decade
  Node(d g) Exclude(vd vg)

  )
  { Poisson Electron Hole }
  }
}
"""
    f = open(out_file, "a")
    f.write(text1)
    f.write(text3)
    f.write(text4)
    f.write(text5)
    f.close()


if __name__ == "__main__":
    import numpy as np

    write_sdevice_quasistationary_ramp_voltage_dd(
        struct="./sprocess/test_pn_fps.tdr",
        directory="./sdevice",
        physics_settings=PHYSICS_SETTINGS_AVALANCHE,
        ramp_final_voltage=50,
        ramp_sample_voltages=np.linspace(0, 1, 11),
    )
    # write_sdevice_ssac_ramp_voltage_dd()
