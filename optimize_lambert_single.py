#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import astropy.time
import astropy.units as u
import spiceypy
from poliastro.iod.izzo import lambert as lambert_izzo
from poliastro.bodies import Sun

import optuna

dep_time1 = astropy.time.Time('2024-08-01T00:00:00.000000', format='isot', scale='utc').mjd
dep_time2 = astropy.time.Time('2024-12-01T00:00:00.000000', format='isot', scale='utc').mjd
arr_time1 = astropy.time.Time('2025-05-01T00:00:00.000000', format='isot', scale='utc').mjd
arr_time2 = astropy.time.Time('2025-11-01T00:00:00.000000', format='isot', scale='utc').mjd


def objective(trial):
    # suggest API
    x = trial.suggest_float("x", dep_time1, dep_time2)
    y = trial.suggest_float("y", arr_time1, arr_time2)
    # float64をTimeオブジェクトに変換
    start = astropy.time.Time(x, format='mjd')
    end   = astropy.time.Time(y, format='mjd')
    step   = 1000
    etbeg = spiceypy.utc2et( start.fits )
    etend = spiceypy.utc2et( end.fits )
    times = [x*(etend-etbeg)/step + etbeg for x in range(step)]
    # 太陽中心、地球・火星のstate vector
    svear, ltear = spiceypy.spkezr('Earth', times, 'J2000', 'NONE', 'SUN')
    svmar, ltmar = spiceypy.spkezr('Mars' , times, 'J2000', 'NONE', 'SUN')
    pos_a = [svear[0][0], 
             svear[0][1], 
             svear[0][2]] * u.km
    pos_b = [svmar[len(svmar)-1][0], 
             svmar[len(svmar)-1][1], 
             svmar[len(svmar)-1][2]] * u.km
    vel_a = [svear[0][3], 
             svear[0][4], 
             svear[0][5]] * u.km / u.s
    vel_b = [svmar[len(svmar)-1][3], 
             svmar[len(svmar)-1][4], 
             svmar[len(svmar)-1][5]] * u.km / u.s
    # Lambert
    k = Sun.k
    ra = pos_a
    rb = pos_b
    tof = (etend - etbeg) * u.s
    short = 1
    sols = list(lambert_izzo(k, ra, rb, tof, M=0, numiter=35, rtol=1e-8))
    if short:
        va, vb = sols[0]
    else:
        va, vb = sols[-1]
    # Vinf
    vdep = va - vel_a
    varr = vel_b - vb
    # RA, DEC
    vector_dep = [vdep[0].value, vdep[1].value, vdep[2].value]
    RANGE_dep, RA_dep, DEC_dep = spiceypy.recrad( vector_dep )
    # print(RANGE_dep, np.rad2deg(RA_dep), np.rad2deg(DEC_dep))
    vector_arr = [varr[0].value, varr[1].value, varr[2].value]
    RANGE_arr, RA_arr, DEC_arr = spiceypy.recrad( vector_arr )
    # print(RANGE_arr, np.rad2deg(RA_arr), np.rad2deg(DEC_arr))
    return RANGE_dep**2


if __name__ == '__main__':
    # SPICE kernel furnish
    METAKR = './meta_kernel.tm'
    spiceypy.furnsh(METAKR)

    # default, TPE(ベイズ最適化)
    study = optuna.create_study(direction="minimize")

    # ランダムサーチ
    # study = optuna.create_study(
    #     direction="minimize", 
    #     sampler=optuna.samplers.RandomSampler()
    # )

    # グリッドサーチ
    # search_space = {
    #     'x': np.linspace(dep_time1, dep_time2, 10), 
    #     'y': np.linspace(arr_time1, arr_time2, 10)
    # }
    # study = optuna.create_study(
    #     direction="minimize", 
    #     sampler=optuna.samplers.GridSampler(search_space=search_space)
    # )

    #
    # RDB
    #
    # study = optuna.create_study(
    #     direction="minimize", 
    #     storage="sqlite:///optuna.db", 
    #     study_name="ch1-conditional", 
    # )

    #
    # load study
    #
    # study = optuna.load_study(
    #     storage="sqlite:///optuna.db", 
    #     study_name="ch1-conditional", 
    # )

    #
    # optimize
    #
    study.optimize(objective, n_trials=100)

    print(f"Best objective value: {study.best_value}")
    print(f"Best parameter: {study.best_params}")
    dep_opt = astropy.time.Time(study.best_params["x"], format='mjd')
    arr_opt = astropy.time.Time(study.best_params["y"], format='mjd')
    print(dep_opt.datetime)
    print(arr_opt.datetime)

    fig = optuna.visualization.plot_contour(
        study=study, 
        params=["x", "y"]
    )
    fig.write_image('./img/optimize_lambert.png')  # save as png file ($ pip install -U kaleido)
