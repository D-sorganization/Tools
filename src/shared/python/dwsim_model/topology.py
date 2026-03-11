from __future__ import annotations

from collections.abc import Callable

Connector = Callable[[object, object, int, int], None]


def _connect(
    connect: Connector, source, target, source_port: int = 0, target_port: int = 0
) -> None:
    connect(source, target, source_port, target_port)


def build_gasifier_stage(
    builder, reactor_type: str, connect: Connector
) -> dict[str, object]:
    """Create the gasifier stage and wire its internal topology."""
    feed_biomass = builder.add_object("MaterialStream", "Gasifier_Biomass_Feed", 0, 300)
    feed_solids = builder.add_object("MaterialStream", "Gasifier_Solids_Feed", 0, 350)
    feed_oxygen = builder.add_object("MaterialStream", "Gasifier_Oxygen_Feed", 0, 400)
    feed_steam = builder.add_object("MaterialStream", "Gasifier_Steam_Feed", 0, 450)

    inlet_mixer = builder.add_object("Mixer", "Gasifier_Inlet_Mixer", 100, 350)
    mixed_feed = builder.add_object("MaterialStream", "Gasifier_Mixed_Feed", 150, 350)

    heat_loss_block = builder.add_object("Cooler", "Gasifier_Heat_Loss_Block", 250, 350)
    post_loss_feed = builder.add_object(
        "MaterialStream", "Gasifier_Feed_PostLoss", 310, 350
    )
    jacket_transfer = builder.add_object("Cooler", "Gasifier_Heat_To_Jacket", 400, 350)
    final_feed = builder.add_object("MaterialStream", "Gasifier_Feed_Final", 460, 350)

    reactor = builder.add_object(reactor_type, "Downdraft_Gasifier", 550, 350)
    syngas_out = builder.add_object("MaterialStream", "Syngas_Pre_PEM", 650, 350)
    glass_out = builder.add_object("MaterialStream", "Gasifier_Glass_Out", 650, 450)

    heat_loss = builder.add_object("EnergyStream", "E_Gasifier_HeatLoss", 250, 250)
    cooling_water_in = builder.add_object(
        "MaterialStream", "Gasifier_Cooling_Water_In", 350, 500
    )
    cooling_steam_out = builder.add_object(
        "MaterialStream", "Gasifier_Cooling_Steam_Out", 500, 500
    )
    cooling_jacket = builder.add_object("Heater", "Gasifier_Cooling_Jacket", 400, 450)
    jacket_flux = builder.add_object("EnergyStream", "E_Gasifier_Flux_to_CW", 400, 400)

    _connect(connect, feed_biomass, inlet_mixer, 0, 0)
    _connect(connect, feed_solids, inlet_mixer, 0, 1)
    _connect(connect, feed_oxygen, inlet_mixer, 0, 2)
    _connect(connect, feed_steam, inlet_mixer, 0, 3)
    _connect(connect, inlet_mixer, mixed_feed, 0, 0)

    _connect(connect, mixed_feed, heat_loss_block, 0, 0)
    _connect(connect, heat_loss_block, post_loss_feed, 0, 0)
    _connect(connect, heat_loss_block, heat_loss, 0, 0)
    _connect(connect, post_loss_feed, jacket_transfer, 0, 0)
    _connect(connect, jacket_transfer, final_feed, 0, 0)
    _connect(connect, jacket_transfer, jacket_flux, 0, 0)

    _connect(connect, cooling_water_in, cooling_jacket, 0, 0)
    _connect(connect, cooling_jacket, cooling_steam_out, 0, 0)
    _connect(connect, cooling_jacket, jacket_flux, 0, 0)

    _connect(connect, final_feed, reactor, 0, 0)
    _connect(connect, reactor, syngas_out, 0, 0)
    _connect(connect, reactor, glass_out, 1, 0)

    return {
        "reactor": reactor,
        "syngas_out": syngas_out,
        "glass_out": glass_out,
    }


def build_pem_stage(
    builder,
    reactor_type: str,
    connect: Connector,
    syngas_inlet=None,
    syngas_inlet_name: str = "PEM_Syngas_Inlet",
) -> dict[str, object]:
    """Create the PEM stage and wire its internal topology."""
    feed_syngas = syngas_inlet or builder.add_object(
        "MaterialStream", syngas_inlet_name, 650, 150
    )
    feed_solids = builder.add_object("MaterialStream", "PEM_Solids_Feed", 650, 200)
    feed_oxygen = builder.add_object("MaterialStream", "PEM_Oxygen_Feed", 650, 250)
    feed_steam = builder.add_object("MaterialStream", "PEM_Steam_Feed", 650, 300)

    inlet_mixer = builder.add_object("Mixer", "PEM_Inlet_Mixer", 750, 350)
    mixed_feed = builder.add_object("MaterialStream", "PEM_Mixed_Feed", 800, 350)

    ac_block = builder.add_object("Heater", "PEM_AC_Block", 900, 350)
    post_ac_feed = builder.add_object("MaterialStream", "PEM_Feed_PostAC", 960, 350)
    dc_block = builder.add_object("Heater", "PEM_DC_Block", 1050, 350)
    post_dc_feed = builder.add_object("MaterialStream", "PEM_Feed_PostDC", 1110, 350)
    heat_loss_block = builder.add_object("Cooler", "PEM_Heat_Loss_Block", 1200, 350)
    final_feed = builder.add_object("MaterialStream", "PEM_Feed_Final", 1260, 350)

    reactor = builder.add_object(reactor_type, "PEM_Reactor", 1350, 350)
    syngas_out = builder.add_object("MaterialStream", "Syngas_Pre_TRC", 1450, 350)
    glass_out = builder.add_object("MaterialStream", "PEM_Glass_Out", 1450, 450)

    ac_power = builder.add_object("EnergyStream", "E_PEM_AC_Power", 900, 250)
    dc_power = builder.add_object("EnergyStream", "E_PEM_DC_Power", 1050, 250)
    heat_loss = builder.add_object("EnergyStream", "E_PEM_HeatLoss", 1200, 250)

    _connect(connect, feed_syngas, inlet_mixer, 0, 0)
    _connect(connect, feed_solids, inlet_mixer, 0, 1)
    _connect(connect, feed_oxygen, inlet_mixer, 0, 2)
    _connect(connect, feed_steam, inlet_mixer, 0, 3)
    _connect(connect, inlet_mixer, mixed_feed, 0, 0)

    _connect(connect, mixed_feed, ac_block, 0, 0)
    _connect(connect, ac_block, post_ac_feed, 0, 0)
    _connect(connect, ac_block, ac_power, 0, 0)

    _connect(connect, post_ac_feed, dc_block, 0, 0)
    _connect(connect, dc_block, post_dc_feed, 0, 0)
    _connect(connect, dc_block, dc_power, 0, 0)

    _connect(connect, post_dc_feed, heat_loss_block, 0, 0)
    _connect(connect, heat_loss_block, final_feed, 0, 0)
    _connect(connect, heat_loss_block, heat_loss, 0, 0)

    _connect(connect, final_feed, reactor, 0, 0)
    _connect(connect, reactor, syngas_out, 0, 0)
    _connect(connect, reactor, glass_out, 1, 0)

    return {
        "syngas_in": feed_syngas,
        "reactor": reactor,
        "syngas_out": syngas_out,
        "glass_out": glass_out,
    }


def build_trc_stage(
    builder,
    reactor_type: str,
    connect: Connector,
    syngas_inlet=None,
    syngas_inlet_name: str = "TRC_Syngas_Inlet",
) -> dict[str, object]:
    """Create the TRC stage and wire its internal topology."""
    feed_syngas = syngas_inlet or builder.add_object(
        "MaterialStream", syngas_inlet_name, 1450, 150
    )
    feed_solids = builder.add_object("MaterialStream", "TRC_Solids_Feed", 1450, 200)
    feed_oxygen = builder.add_object("MaterialStream", "TRC_Oxygen_Feed", 1450, 250)
    feed_steam = builder.add_object("MaterialStream", "TRC_Steam_Feed", 1450, 300)

    inlet_mixer = builder.add_object("Mixer", "TRC_Inlet_Mixer", 1550, 350)
    mixed_feed = builder.add_object("MaterialStream", "TRC_Mixed_Feed", 1600, 350)

    heat_loss_block = builder.add_object("Cooler", "TRC_Heat_Loss_Block", 1700, 350)
    final_feed = builder.add_object("MaterialStream", "TRC_Feed_Final", 1760, 350)

    reactor = builder.add_object(reactor_type, "TRC_Reactor", 1850, 350)
    syngas_out = builder.add_object("MaterialStream", "Syngas_Pre_Quench", 1950, 350)

    heat_loss = builder.add_object("EnergyStream", "E_TRC_HeatLoss", 1700, 250)

    _connect(connect, feed_syngas, inlet_mixer, 0, 0)
    _connect(connect, feed_solids, inlet_mixer, 0, 1)
    _connect(connect, feed_oxygen, inlet_mixer, 0, 2)
    _connect(connect, feed_steam, inlet_mixer, 0, 3)
    _connect(connect, inlet_mixer, mixed_feed, 0, 0)

    _connect(connect, mixed_feed, heat_loss_block, 0, 0)
    _connect(connect, heat_loss_block, final_feed, 0, 0)
    _connect(connect, heat_loss_block, heat_loss, 0, 0)

    _connect(connect, final_feed, reactor, 0, 0)
    _connect(connect, reactor, syngas_out, 0, 0)

    return {
        "syngas_in": feed_syngas,
        "reactor": reactor,
        "syngas_out": syngas_out,
    }
