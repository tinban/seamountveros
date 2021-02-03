"""
Veros from scratch

An idealized model of the Benham Bank Seamount using Veros, a Python-based ocean model

Author: Kristine Bantay
Created on: 01/05/2021
Last edit: 02/03/2021

"""

from veros import VerosSetup, veros_method
from veros.tools import cli
from veros.variables import allocate
from veros.distributed import global_min, global_max
import matplotlib.pyplot as plt


class ch_smnt(VerosSetup):
    """
    A model of the Benham Bank Seamount on idealized channel setting.
"""


    simulation = ch_smnt(backend='bohrium')
    simulation.run()
    plt.imshow(simulation.state.psi[..., 0])
    plt.show()

    @veros_method
    def set_parameter(self, vs):
        vs.nx, vs.ny, vs.nz = (50, 100, 10)                                      #grid size
        vs.coord_degree = True
        vs.enable_cyclic = True
        pass

    @veros_method
    def set_initial_conditions(self, vs):
        vs.u[:, :, :, vs.tau] = np.random.rand(54, 104, 10)

        # wind stress forcing
        yt_min = global_min(vs, vs.yt.min())
        yu_min = global_min(vs, vs.yu.min())
        yt_max = global_max(vs, vs.yt.max())
        yu_max = global_max(vs, vs.yu.max())

        # surface heatflux forcing
        vs._t_star = allocate(vs, ('yt',), fill=15)
        vs._t_star[vs.yt < -20] = 15 * (vs.yt[vs.yt < -20] - yt_min) / (-20 - yt_min)
        vs._t_star[vs.yt > 20] = 15 * (1 - (vs.yt[vs.yt > 20] - 20) / (yt_max - 20))
        vs._t_rest = vs.dzt[None, -1] / (30. * 86400.) * vs.maskT[:, :, -1]
        pass

    @veros_method
    def set_grid(self, vs):
        vs.x_origin, vs.y_origin = 0, 0
        vs.dxt[...] = 1
        vs.dyt[...] = 1
        vs.dzt[...] = 1
        pass

    @veros_method
    def set_coriolis(self, vs):
        vs.coriolis_t[:, :] = 2 * vs.omega * np.sin(vs.yt[np.newaxis, :] / 180. * vs.pi)
        pass

    @veros_method
    def set_topography(self, vs):
        vs.kbot[:, :] = 100
        # add a rectangular island somewhere inside the domain
        vs.kbot[10:20, 10:20] = 0
        pass

    @veros_method
    def set_forcing(self, vs):
        # current_month = (vs.time / (31 * 24 * 60 * 60)) % 12
        # vs.surface_taux[:, :] = vs._windstress_data[:, :, current_month]
        vs.forc_temp_surface[...] = vs._t_rest * (vs._t_star - vs.temp[:, :, -1, vs.tau])
        pass

    @veros_method
    def set_diagnostics(self, vs):
        vs.diagnostics['snapshot'].output_variables += ['dsalt', 'dtemp']
        pass

    def after_timestep(self, vs):
        pass


@cli
def run(*args, **kwargs):
    simulation = ch_smnt(*args, **kwargs)
    simulation.setup()
    simulation.run()


if __name__ == '__main__':
    run()
