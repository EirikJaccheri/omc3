"""
Chromatic
----------------

:module: optics_measurements.chromatic
:author: Lukas Malina

Computes various chromatic beam properties
"""
import numpy as np
import pandas as pd
from os.path import join
from omc3.utils import outliers, stats
import tfs

from omc3.definitions.constants import PLANES, PLANE_TO_NUM
from omc3.optics_measurements.constants import DELTA, ERR, MDL
from omc3.optics_measurements.toolbox import df_prod, df_ratio, df_ang_diff, _ang_diff, ang_sum

PLANE_TO_RES2D = dict(X="10", Y="01")
AMP = "AMP"
PHASE = "PHASE"


def calculate_w_and_phi(betas, dpps, input_files, measure_input, plane):
    columns = [f"{pref}{DELTA}{col}{plane}" for pref in ("", ERR) for col in ("BET", "ALF")]
    joined = betas[0].loc[:, columns]
    for i, beta in enumerate(betas[1:]):
        joined = pd.merge(joined, beta.loc[:, columns], how="inner", left_index=True,
                          right_index=True, suffixes=('', '__' + str(i + 1)))
    for column in columns:
        joined.rename(columns={column: column + '__0'}, inplace=True)
    joined = pd.merge(joined,
                      betas[np.argmin(np.abs(dpps))].loc[:, [f"ALF{plane}", f"{ERR}ALF{plane}"]],
                      how="inner",
                      left_index=True, right_index=True)
    for col in ("BET", "ALF"):
        fit = np.polyfit(np.repeat(dpps, 2),
                         np.repeat(input_files.get_data(joined, f"{DELTA}{col}{plane}").T, 2,
                                   axis=0), 1, cov=True)
        joined[f"D{col}{plane}"] = fit[0][-2, :].T
        joined[f"{ERR}D{col}{plane}"] = np.sqrt(fit[1][-2, -2, :].T)
    a = joined.loc[:, f"DBET{plane}"].to_numpy()
    aerr = joined.loc[:, f"{ERR}DBET{plane}"].to_numpy()
    b = joined.loc[:, f"DALF{plane}"].to_numpy() - joined.loc[:, f"ALF{plane}"].to_numpy() * joined.loc[:,
                                                                                     f"DBET{plane}"].to_numpy()
    berr = np.sqrt(df_prod(joined, f"{ERR}DALF{plane}", f"{ERR}DALF{plane}") +
                   np.square(df_prod(joined, f"{ERR}ALF{plane}", f"DBET{plane}")) +
                   np.square(df_prod(joined, f"ALF{plane}", f"{ERR}DBET{plane}")))
    w = np.sqrt(np.square(a) + np.square(b))
    joined[f"W{plane}"] = w
    joined[f"{ERR}W{plane}"] = np.sqrt(np.square(a * aerr / w) + np.square(b * berr / w))
    joined[f"PHI{plane}"] = np.arctan2(b, a) / (2 * np.pi)
    joined[f"{ERR}PHI{plane}"] = 1 / (1 + np.square(a / b)) * np.sqrt(
        np.square(aerr / b) + np.square(berr * a / np.square(b))) / (2 * np.pi)
    output_df = pd.merge(measure_input.accelerator.model.loc[:,
                         ["S", f"MU{plane}", f"BET{plane}", f"ALF{plane}", f"W{plane}",
                          f"PHI{plane}"]],
                         joined.loc[:,
                         [f"{pref}{col}{plane}" for pref in ("", ERR) for col in ("W", "PHI")]],
                         how="inner", left_index=True,
                         right_index=True, suffixes=(MDL, ''))
    output_df.rename(columns={"SMDL": "S"}, inplace=True)
    return output_df


def calculate_chromatic_coupling(couplings, dpps, input_files, measure_input):
    # TODO how to treat the model values?
    columns = [f"{pref}{col}{part}" for pref in ("", ERR) for col in ("F1001", "F1010") for part in ("RE", "IM")]
    joined = couplings[0].loc[:, columns]
    for i, coup in enumerate(couplings[1:]):
        joined = pd.merge(joined, coup.loc[:, columns], how="inner", left_index=True,
                          right_index=True, suffixes=('', '__' + str(i + 1)))
    for column in columns:
        joined.rename(columns={column: column + '__0'}, inplace=True)

    for col in ("F1001", "F1010"):
        for part in ("RE", "IM"):
            fit = np.polyfit(np.repeat(dpps, 2),
                             np.repeat(input_files.get_data(joined, f"{col}{part}").T, 2,
                                       axis=0), 1, cov=True)
            joined[f"D{col}{part}"] = fit[0][-2, :].T
            joined[f"{ERR}D{col}{part}"] = np.sqrt(fit[1][-2, -2, :].T)
        joined[f"D{col}"] = np.sqrt(np.square(joined.loc[:, f"D{col}RE"].to_numpy()) + np.square(joined.loc[:, f"D{col}IM"].to_numpy()))
        joined[f"{ERR}D{col}"] = np.sqrt(np.square(joined.loc[:, f"D{col}RE"].to_numpy() * df_ratio(joined, f"{ERR}D{col}RE", f"D{col}")) +
                                         np.square(joined.loc[:, f"D{col}IM"].to_numpy() * df_ratio(joined, f"{ERR}D{col}IM", f"D{col}")))
    output_df = pd.merge(measure_input.accelerator.model.loc[:, ["S"]], joined.loc[:,
                         [f"{pref}{col}{part}" for pref in ("", ERR) for col in ("F1001", "F1010") for part in ("", "RE", "IM")]],
                         how="inner", left_index=True,
                         right_index=True)
    return output_df


def synchrotron_tunes(input_files):
    headers_with_3d = [df.headers for df in input_files["X"] if df.DPPAMP > 0]
    tune_list = [header[f"Q3"] for header in headers_with_3d]
    tune_rms_list = [header[f"Q3RMS"] for header in headers_with_3d]
    return np.array(tune_list), np.array(tune_rms_list)


def synchrotron_phases(meas_input, input_files):
    df_syn_phases = input_files.joined_frame("X", ["MUZ", ], dpp_amp=True)
    arc_mask = meas_input.accelerator.get_element_types_mask(df_syn_phases.index, ["arc_bpm"])
    arc_synchrotron_phases = input_files.get_data(df_syn_phases, "MUZ")[arc_mask]
    return (stats.circular_mean(arc_synchrotron_phases, period=1, axis=0),
            stats.circular_error(arc_synchrotron_phases, period=1, axis=0))


def chromas(meas_input, input_files, tune_dict, plane, sign=None):
    delta = tune_dict[plane]["Q"] - tune_dict[plane]["QF"]  # signed delta driven - natural tune
    if sign is None:
        sign = int(-np.sign(delta))
    if sign not in (-1, 1):
        raise ValueError("Sign should be either -1 or 1.")

    synch_tunes, synch_tune_errors = synchrotron_tunes(input_files)
    print(synch_tunes.shape)
    synch_phases, synch_phase_errors = synchrotron_phases(meas_input, input_files)
    print(synch_phases.shape)
    dpp_amps = np.array([df.DPPAMP for df in input_files[plane] if df.DPPAMP > 0])

    df = pd.DataFrame(meas_input.accelerator.model).loc[:, ['S', f"MU{plane}", f"W{plane}", f"PHI{plane}"]]
    df.rename(columns={f"MU{plane}": f"MU{plane}{MDL}",f"W{plane}": f"W{plane}{MDL}", f"PHI{plane}": f"PHI{plane}{MDL}"}, inplace=True)
    cols = synch_beta_cols(plane, sign)
    df = pd.merge(df, input_files.joined_frame(plane, cols, dpp_amp=True),
                  how='inner', left_index=True, right_index=True)
    chroma_wo_phase_sign = 2 * input_files.get_data(df, cols[1]) * (delta + sign * synch_tunes) / dpp_amps
    phase_corr = (ang_sum(input_files.get_data(df, cols[2]), synch_phases) if sign < 0 else
                  _ang_diff(input_files.get_data(df, cols[2]), synch_phases))
    cosines = np.cos(2 * np.pi * _ang_diff(input_files.get_data(df, f"MU{plane}"), phase_corr))
    print(f"Average amp(cos): {np.mean(np.abs(cosines), axis=0)}")
    print(f"Average cos: {np.mean(cosines, axis=0)}")
    phase_sign = np.sign(np.cos(2 * np.pi * _ang_diff(input_files.get_data(df, f"MU{plane}"), phase_corr)))
    mask = meas_input.accelerator.get_element_types_mask(df.index, ["arc_bpm"])
    global_phase_signs = np.sign(np.mean(phase_sign[mask], axis=0))
    chromas, chroma_errors = filter_and_average(chroma_wo_phase_sign[mask])
    chromas = chromas * global_phase_signs
    print(f"Average cos(arcs): {np.mean(cosines[mask], axis=0)}")
    print(f"Chroma {plane}{sign}: \n {chromas}+-{chroma_errors}\n")

    #chroma_beating = (all_chromas - chromas) / np.abs(delta)
    #df[f"W{plane}COSPHI"] = np.mean(chroma_beating, axis=1)
    #df[f"W{plane}COSPHI{MDL}"] = df.loc[:,f"W{plane}{MDL}"].to_numpy() * np.cos(2 * np.pi * df.loc[:,f"PHI{plane}{MDL}"].to_numpy())
    #df[f"W{plane}SINPHI{MDL}"] = df.loc[:, f"W{plane}{MDL}"].to_numpy() * np.sin(
    #    2 * np.pi * df.loc[:, f"PHI{plane}{MDL}"].to_numpy())
    #tfs.write(join(meas_input.outputdir, f"chroma{plane.lower()}{'p' if sign > 0 else 'm'}.tfs"), df, headers_dict={f"DQ{plane}": np.mean(chromas)})
    return chromas, chroma_errors


def append_chromas(list_of_tfs, chromas, chroma_errors, plane):
    dpp_amp_inds = np.ravel(np.argwhere(np.array([df.DPPAMP for df in list_of_tfs]) > 0))
    if dpp_amp_inds.shape != chromas.shape:
        print(f"{dpp_amp_inds.shape} {chromas.shape}")
        raise ValueError("dp/p amplitudes are of different length than chromas, something went wrong")

    for i, k in enumerate(dpp_amp_inds):
        list_of_tfs[k].headers[f"DQ{PLANE_TO_NUM[plane]}"] = chromas[i]
        list_of_tfs[k].headers[f"DQ{PLANE_TO_NUM[plane]}RMS"] = chroma_errors[i]
    return list_of_tfs


def synch_beta_cols(plane, sign):
    z_str = str(sign).replace("-", "_")
    return [f"MU{plane}", f"{AMP}{PLANE_TO_RES2D[plane]}{z_str}", f"{PHASE}{PLANE_TO_RES2D[plane]}{z_str}"]


def filter_and_average(arc_chromas):
    if arc_chromas.ndim == 1:
        filtered_amps = arc_chromas[outliers.get_filter_mask(arc_chromas)]
        return np.average(filtered_amps), np.std(filtered_amps)/np.sqrt(len(filtered_amps))
    avs, stds = [], []
    for i in range(arc_chromas.shape[1]):
        filtered_amps = arc_chromas[outliers.get_filter_mask(arc_chromas[:, i]), i]
        avs.append(np.average(filtered_amps))
        stds.append(np.std(filtered_amps) / np.sqrt(len(filtered_amps)))
    return np.array(avs), np.array(stds)



# def get_chromatic_beating(meas_input, input_files, tune_dict, dpp_amp, chromas):
#     synch_beta_col = {"X": ["AMP101", "AMP10_1"], "Y": ["AMP011", "AMP01_1"]}
#     for plane in PLANES:
#         delta = tune_dict[plane]["Q"] - tune_dict[plane]["QF"]
#         model = meas_input.accelerator.get_model_tfs()
#         synch_beta = input_files.joined_frame(plane, synch_beta_col[plane])
#         df = pd.merge(model.loc[:, ['NAME', 'S', 'WX', 'WY', 'PHIX', 'PHIY']], synch_beta,
#                       how='inner', left_index=True, right_index=True)
#         aver_amps = np.sqrt(synch_beta.loc[:, input_files.get_columns(
#             synch_beta, synch_beta_col[plane][0])].values *
#             synch_beta.loc[:, input_files.get_columns(synch_beta, synch_beta_col[plane][1])].values) * 2
#         # 2 comes from sb lines on both sides...+ Qs is correction for driven motion
#
#         # TODO do the sign correctly
#         chroma_contribution = chromas[plane] * dpp_amp / np.abs(delta)
#         chroma_beating_sqrt = aver_amps - chroma_contribution
#         chroma_beating = (np.square(1 + chroma_beating_sqrt) - 1) / (2 * dpp_amp)
#         for i in range(chroma_beating.shape[1]):
#             df[f"CHBEAT__{i}"] = chroma_beating[:, i]
#         df[f"CHROMBEAT"] = np.mean(chroma_beating, axis=1)
#         df[f"STDCHROMBEAT"] = np.std(chroma_beating, axis=1)
#         tfs.write(join(meas_input.outputdir, f"chrombeat{plane.lower()}.out"), df)
#
#     return



def chroma(meas_input, input_files, tune_dict, plane):
    synch_tunes, synch_tune_errors = synchrotron_tunes(input_files)
    synch_phases, synch_phase_errors = synchrotron_phases(meas_input, input_files)
    dpp_amps = np.array([df.DPPAMP for df in input_files[plane] if df.DPPAMP > 0])
    delta = tune_dict[plane]["Q"] - tune_dict[plane]["QF"]  # signed delta driven - natural tune
    # if dpp_amps.shape[0] == 0:
    #     return np.zeros(len(input_files["X"]))
    df = pd.DataFrame(meas_input.accelerator.model).loc[:, ['S', f"MU{plane}"]]
    df.rename(columns={f"MU{plane}": f"MU{plane}{MDL}"}, inplace=True)


    synch_beta_col = {"X": ["AMP101", "AMP10_1", "MUX", "PHASE101", "PHASE10_1",], "Y": ["AMP011", "AMP01_1", "MUY","PHASE011", "PHASE01_1",]}
    df = pd.merge(df, input_files.joined_frame(plane, synch_beta_col[plane], dpp_amp=True),
                  how='inner', left_index=True, right_index=True)
    mask = meas_input.accelerator.get_element_types_mask(df.index, ["arc_bpm"])
    #synch_phases = stats.circular_mean(input_files.get_data(df, "MUZ")[mask], period=1, axis=0)
    print(f"Synchrotron phases: {synch_phases[0]}")

    chroma_amps1 = 2 * input_files.get_data(df, synch_beta_col[plane][0]) * (delta + synch_tunes) / dpp_amps
    chroma_amps_1 = 2 * input_files.get_data(df, synch_beta_col[plane][1]) * (delta - synch_tunes) / dpp_amps
    phase_signp = _ang_diff(input_files.get_data(df, synch_beta_col[plane][2]), _ang_diff(input_files.get_data(df, synch_beta_col[plane][3]), synch_phases - 0.25))
    phase_signm = _ang_diff(input_files.get_data(df, synch_beta_col[plane][2]), ang_sum(input_files.get_data(df, synch_beta_col[plane][4]), synch_phases - 0.25))
    chromasp = chroma_amps1 * np.sign(np.cos(2 * np.pi * phase_signp))
    chromasm = chroma_amps_1 * np.sign(np.cos(2 * np.pi * phase_signm))
    df["CHROM_P"] = chromasp
    df["CHROM_M"] = chromasm

    p1, p1err = filter_and_average(chromasp[mask])
    m1, m1err = filter_and_average(chromasm[mask])
    print(f"Chroma {plane}: \n   p1: {p1}+-{p1err}\n    {m1}+-{m1err}\n")
    tfs.write(join(meas_input.outputdir, f"chroma{plane.lower()}.tfs"), df)
    return