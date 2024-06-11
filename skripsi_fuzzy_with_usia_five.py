# BISMILLAH - TERUJI COCOK

from fastapi import FastAPI
from pydantic import BaseModel

import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

app = FastAPI()


class InputData(BaseModel):
    usia: float
    bb: float
    tb: float
    jk: str


def countBB_U(usia, bb, jk):
    # definiskan domain atau rentang nilai variabel
    # VAR INPUT
    x_usia = np.arange(0, 60, 1)
    x_bb = np.arange(2, 28, 1)
    if jk == "L":
        x_tb = np.arange(44, 130, 1)
    else:
        x_tb = np.arange(43, 130, 1)

    # VAR OUTPUT
    if jk == "L":
        x_bb_u = np.arange(3, 30, 1)
        x_tb_u = np.arange(45, 130, 1)
        x_bb_tb = np.arange(3, 40, 1)
    else:
        x_bb_u = np.arange(4, 30, 1)
        x_tb_u = np.arange(44, 130, 1)
        x_bb_tb = np.arange(3, 38, 1)

    # membership / keanggotaan USIA
    usia_tahap1 = fuzz.trapmf(x_usia, [0, 0, 6, 12])
    usia_tahap2 = fuzz.trimf(x_usia, [6, 12, 24])
    usia_tahap3 = fuzz.trimf(x_usia, [12, 24, 36])
    usia_tahap4 = fuzz.trimf(x_usia, [24, 36, 48])
    usia_tahap5 = fuzz.trapmf(x_usia, [36, 48, 60, 60])

    # membership / keanggotaan BERAT BADAN
    bb_sangat_kurang = fuzz.trapmf(x_bb, [2, 2, 4, 6])
    if jk == "L":
        bb_kurang = fuzz.trimf(x_bb, [4, 8, 12])
        bb_normal = fuzz.trimf(x_bb, [8, 13, 18])
        bb_lebih = fuzz.trimf(x_bb, [15, 18, 23])
        bb_sangat_lebih = fuzz.trapmf(x_bb, [21, 23, 28, 28])
    else:
        bb_kurang = fuzz.trimf(x_bb, [4, 8, 11])
        bb_normal = fuzz.trimf(x_bb, [9, 13, 18])
        bb_lebih = fuzz.trimf(x_bb, [15, 20, 24])
        bb_sangat_lebih = fuzz.trapmf(x_bb, [22, 25, 28, 28])

    # membership / keanggotaan BB-USIA
    if jk == "L":
        bb_u_sangat_kurang = fuzz.trapmf(x_bb_u, [3, 3, 6, 8])
        bb_u_kurang = fuzz.trimf(x_bb_u, [6, 8, 10])
        bb_u_normal = fuzz.trimf(x_bb_u, [8, 12, 18])
        bb_u_resiko_bb_lebih = fuzz.trapmf(x_bb_u, [16, 24, 30, 30])
    else:
        bb_u_sangat_kurang = fuzz.trapmf(x_bb_u, [4, 4, 7, 9])
        bb_u_kurang = fuzz.trimf(x_bb_u, [7, 9, 11])
        bb_u_normal = fuzz.trimf(x_bb_u, [9, 13, 18])
        bb_u_resiko_bb_lebih = fuzz.trapmf(x_bb_u, [16, 23, 30, 30])

    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(8, 9))

    ax0.plot(x_usia, usia_tahap1, 'b', linewidth=1.5, label="Tahap 1")
    ax0.plot(x_usia, usia_tahap2, 'g', linewidth=1.5, label="Tahap 2")
    ax0.plot(x_usia, usia_tahap3, 'r', linewidth=1.5, label="Tahap 3")
    ax0.plot(x_usia, usia_tahap4, 'm', linewidth=1.5, label="Tahap 4")
    ax0.plot(x_usia, usia_tahap5, 'c', linewidth=1.5, label="Tahap 5")
    ax0.set_title("---- USIA ----")
    ax0.legend()

    ax1.plot(x_bb, bb_sangat_kurang, 'b', linewidth=1.5, label="Sangat Kurang")
    ax1.plot(x_bb, bb_kurang, 'g', linewidth=1.5, label="Kurang")
    ax1.plot(x_bb, bb_normal, 'r', linewidth=1.5, label="Normal")
    ax1.plot(x_bb, bb_lebih, 'm', linewidth=1.5, label="Lebih")
    ax1.plot(x_bb, bb_sangat_lebih, 'c', linewidth=1.5, label="Sangat Lebih")
    ax1.set_title("---- Berat Badan ----")
    ax1.legend()

    ax2.plot(x_bb_u, bb_u_sangat_kurang, 'b', linewidth=1.5, label="BB Sangat Kurang")
    ax2.plot(x_bb_u, bb_u_kurang, 'g', linewidth=1.5, label="BB Kurang")
    ax2.plot(x_bb_u, bb_u_normal, 'r', linewidth=1.5, label="BB Normal")
    ax2.plot(x_bb_u, bb_u_resiko_bb_lebih, 'm', linewidth=1.5, label="Resiko BB Lebih")
    ax2.set_title("---- BB / U ----")
    ax2.legend()

    for ax in (ax0, ax1, ax2):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
    plt.tight_layout()
    # plt.show()

    u_tahap1 = fuzz.interp_membership(x_usia, usia_tahap1, usia)
    u_tahap2 = fuzz.interp_membership(x_usia, usia_tahap2, usia)
    u_tahap3 = fuzz.interp_membership(x_usia, usia_tahap3, usia)
    u_tahap4 = fuzz.interp_membership(x_usia, usia_tahap4, usia)
    u_tahap5 = fuzz.interp_membership(x_usia, usia_tahap5, usia)

    bb_sk = fuzz.interp_membership(x_bb, bb_sangat_kurang, bb)
    bb_k = fuzz.interp_membership(x_bb, bb_kurang, bb)
    bb_n = fuzz.interp_membership(x_bb, bb_normal, bb)
    bb_l = fuzz.interp_membership(x_bb, bb_lebih, bb)
    bb_sl = fuzz.interp_membership(x_bb, bb_sangat_lebih, bb)

    print(u_tahap1, u_tahap2, u_tahap3, u_tahap4, u_tahap5)  # 5 FASE
    print(bb_sk, bb_k, bb_n, bb_l, bb_sl)

    drjt_usia = u_tahap1, u_tahap2, u_tahap3, u_tahap4, u_tahap5  # 5 FASE
    drjt_bb = bb_sk, bb_k, bb_n, bb_l, bb_sl

    # Rule      => usia, bb = output aturan     # Rule      => usia, bb = output aturan

    # USIA 5 FASE
    rule1 = np.fmin(drjt_usia[0], drjt_bb[0])  # Rule 1 => tahap1, sangat_kurang        = kurang
    rule2 = np.fmin(drjt_usia[0], drjt_bb[1])  # Rule 2 => tahap1, kurang               = normal
    rule3 = np.fmin(drjt_usia[0], drjt_bb[2])  # Rule 3 => tahap1, normal               = resiko_bb_lebih
    rule4 = np.fmin(drjt_usia[0], drjt_bb[3])  # Rule 4 => tahap1, lebih                = resiko_bb_lebih
    rule5 = np.fmin(drjt_usia[0], drjt_bb[4])  # Rule 5 => tahap1, sangat_lebih         = resiko_bb_lebih

    # USIA 5 FASE
    rule6 = np.fmin(drjt_usia[1], drjt_bb[0])  # Rule 6 => tahap2, sangat_kurang        = kurang
    rule7 = np.fmin(drjt_usia[1], drjt_bb[1])  # Rule 7 => tahap2, kurang               = normal
    rule8 = np.fmin(drjt_usia[1], drjt_bb[2])  # Rule 8 => tahap2, normal               = normal
    rule9 = np.fmin(drjt_usia[1], drjt_bb[3])  # Rule 9 => tahap2, lebih                = resiko_bb_lebih
    rule10 = np.fmin(drjt_usia[1], drjt_bb[4])  # Rule 10 => tahap2, sangat_lebih       = resiko_bb_lebih

    # USIA 5 FASE
    rule11 = np.fmin(drjt_usia[2], drjt_bb[0])  # Rule 11 => tahap3, sangat_kurang      = sangat_kurang
    rule12 = np.fmin(drjt_usia[2], drjt_bb[1])  # Rule 12 => tahap3, kurang             = kurang
    rule13 = np.fmin(drjt_usia[2], drjt_bb[2])  # Rule 13 => tahap3, normal             = normal
    rule14 = np.fmin(drjt_usia[2], drjt_bb[3])  # Rule 14 => tahap3, lebih              = resiko_bb_lebih
    rule15 = np.fmin(drjt_usia[2], drjt_bb[4])  # Rule 15 => tahap3, sangat_lebih       = resiko_bb_lebih

    # USIA 5 FASE
    rule16 = np.fmin(drjt_usia[3], drjt_bb[0])  # Rule 16 => tahap4, sangat_kurang      = sangat_kurang
    rule17 = np.fmin(drjt_usia[3], drjt_bb[1])  # Rule 17 => tahap4, kurang             = kurang
    rule18 = np.fmin(drjt_usia[3], drjt_bb[2])  # Rule 18 => tahap4, normal             = normal
    rule19 = np.fmin(drjt_usia[3], drjt_bb[3])  # Rule 19 => tahap4, lebih              = resiko_bb_lebih
    rule20 = np.fmin(drjt_usia[3], drjt_bb[4])  # Rule 20 => tahap4, sangat_lebih       = resiko_bb_lebih

    # USIA 5 FASE
    rule21 = np.fmin(drjt_usia[4], drjt_bb[0])  # Rule 21 => tahap5, sangat_kurang      = sangat_kurang
    rule22 = np.fmin(drjt_usia[4], drjt_bb[1])  # Rule 22 => tahap5, kurang             = sangat_kurang
    rule23 = np.fmin(drjt_usia[4], drjt_bb[2])  # Rule 23 => tahap5, normal             = kurang
    rule24 = np.fmin(drjt_usia[4], drjt_bb[3])  # Rule 24 => tahap5, lebih              = normal
    rule25 = np.fmin(drjt_usia[4], drjt_bb[4])  # Rule 25 => tahap5, sangat_lebih       = resiko_bb_lebih

    # USIA 5 FASE
    pk_bb_u_sangat_kurang = np.fmax(rule11, np.fmax(rule16, np.fmax(rule21, rule22)))
    pk_bb_u_kurang = np.fmax(rule1, np.fmax(rule6, np.fmax(rule12, np.fmax(rule17, rule23))))
    pk_bb_u_normal = np.fmax(rule2, np.fmax(rule7, np.fmax(rule8, np.fmax(rule13, np.fmax(rule18, rule24)))))
    pk_bb_u_resiko_bb_lebih = np.fmax(rule3,
                                      np.fmax(rule4, np.fmax(rule5, np.fmax(rule9, np.fmax(rule10, np.fmax(rule14,
                                                                                                           np.fmax(
                                                                                                               rule15,
                                                                                                               np.fmax(
                                                                                                                   rule19,
                                                                                                                   np.fmax(
                                                                                                                       rule20,
                                                                                                                       rule25)))))))))

    print(pk_bb_u_sangat_kurang)
    print(pk_bb_u_kurang)
    print(pk_bb_u_normal)
    print(pk_bb_u_resiko_bb_lebih)

    pk_bb_u_sangat_kurang = np.fmin(pk_bb_u_sangat_kurang, bb_u_sangat_kurang)
    pk_bb_u_kurang = np.fmin(pk_bb_u_kurang, bb_u_kurang)
    pk_bb_u_normal = np.fmin(pk_bb_u_normal, bb_u_normal)
    pk_bb_u_resiko_bb_lebih = np.fmin(pk_bb_u_resiko_bb_lebih, bb_u_resiko_bb_lebih)

    pk_bb_u_0 = np.zeros_like(x_bb_u)
    pk_bb_u_sk = np.zeros_like(bb_u_sangat_kurang)
    pk_bb_u_k = np.zeros_like(bb_u_kurang)
    pk_bb_u_n = np.zeros_like(bb_u_normal)
    pk_bb_u_rbbl = np.zeros_like(bb_u_resiko_bb_lebih)

    fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(nrows=5, figsize=(8, 9))

    ax0.fill_between(x_bb_u, pk_bb_u_sk, pk_bb_u_sangat_kurang, facecolor='b', alpha=0.7)
    ax0.plot(x_bb_u, bb_u_sangat_kurang, 'b', linewidth=1.5, linestyle='--', label="BB Sangat Kurang")
    ax0.set_title('BB/U')
    ax0.legend()

    ax1.fill_between(x_bb_u, pk_bb_u_k, pk_bb_u_kurang, facecolor='b', alpha=0.7)
    ax1.plot(x_bb_u, bb_u_kurang, 'b', linewidth=1.5, linestyle='--', label="Kurang")
    ax1.set_title('BB/U')
    ax1.legend()

    ax2.fill_between(x_bb_u, pk_bb_u_n, pk_bb_u_normal, facecolor='b', alpha=0.7)
    ax2.plot(x_bb_u, bb_u_normal, 'b', linewidth=1.5, linestyle='--', label="Normal")
    ax2.set_title('BB/U')
    ax2.legend()

    ax3.fill_between(x_bb_u, pk_bb_u_rbbl, pk_bb_u_resiko_bb_lebih, facecolor='b', alpha=0.7)
    ax3.plot(x_bb_u, bb_u_resiko_bb_lebih, 'b', linewidth=1.5, linestyle='--', label="Resiko BB Lebih")
    ax3.set_title('BB/U')
    ax3.legend()

    ax4.fill_between(x_bb_u, pk_bb_u_0, pk_bb_u_sangat_kurang, facecolor='b', alpha=0.7)
    ax4.plot(x_bb_u, bb_u_sangat_kurang, 'b', linewidth=1.5, linestyle='--', label="BB Sangat Kurang")
    ax4.fill_between(x_bb_u, pk_bb_u_0, pk_bb_u_kurang, facecolor='g', alpha=0.7)
    ax4.plot(x_bb_u, bb_u_kurang, 'g', linewidth=1.5, linestyle='--', label="Kurang")
    ax4.fill_between(x_bb_u, pk_bb_u_0, pk_bb_u_normal, facecolor='r', alpha=0.7)
    ax4.plot(x_bb_u, bb_u_normal, 'r', linewidth=1.5, linestyle='--', label="Normal")
    ax4.fill_between(x_bb_u, pk_bb_u_0, pk_bb_u_resiko_bb_lebih, facecolor='m', alpha=0.7)
    ax4.plot(x_bb_u, bb_u_resiko_bb_lebih, 'r', linewidth=1.5, linestyle='--', label="Resiko BB Lebih")
    ax4.set_title('Berat Badan / Usia')
    ax4.legend()

    for ax in (ax0, ax1, ax2, ax3, ax4):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
    plt.tight_layout()
    # plt.show()

    komposisi = np.fmax(pk_bb_u_sangat_kurang,
                        np.fmax(pk_bb_u_kurang, np.fmax(pk_bb_u_normal, pk_bb_u_resiko_bb_lebih)))

    berat_badan_per_usia = fuzz.defuzz(x_bb_u, komposisi, 'centroid')

    # Calculate membership values for the defuzzified result
    bb_u_sangat_kurang_degree = fuzz.interp_membership(x_bb_u, bb_u_sangat_kurang, berat_badan_per_usia)
    bb_u_kurang_degree = fuzz.interp_membership(x_bb_u, bb_u_kurang, berat_badan_per_usia)
    bb_u_normal_degree = fuzz.interp_membership(x_bb_u, bb_u_normal, berat_badan_per_usia)
    bb_u_resiko_bb_lebih_degree = fuzz.interp_membership(x_bb_u, bb_u_resiko_bb_lebih, berat_badan_per_usia)

    status_gizi = 'Tidak Terdefinisi'
    max_keanggotaan = max(bb_u_sangat_kurang_degree, bb_u_kurang_degree, bb_u_normal_degree,
                          bb_u_resiko_bb_lebih_degree)
    if max_keanggotaan == bb_u_sangat_kurang_degree:
        status_gizi = 'Sangat Kurang'
    elif max_keanggotaan == bb_u_kurang_degree:
        status_gizi = 'Kurang'
    elif max_keanggotaan == bb_u_normal_degree:
        status_gizi = 'Normal'
    elif max_keanggotaan == bb_u_resiko_bb_lebih_degree:
        status_gizi = 'Resiko BB Lebih'

    # Output result with membership values
    results = {
        'defuzzified_value': berat_badan_per_usia,
        'status gizi': status_gizi,
        'bb_u_sangat_kurang_degree': bb_u_sangat_kurang_degree,
        'bb_u_kurang_degree': bb_u_kurang_degree,
        'bb_u_normal_degree': bb_u_normal_degree,
        'bb_u_resiko_bb_lebih_degree': bb_u_resiko_bb_lebih_degree
    }

    print(results)

    plt.plot([berat_badan_per_usia, berat_badan_per_usia],
             [0, fuzz.interp_membership(x_bb_u, komposisi, berat_badan_per_usia)], 'k',
             linewidth=1.5, alpha=0.9)
    # plt.show()

    return results


# BELUM DIUBAH - CATATAN
def countTB_U(usia, tb, jk):
    # VAR INPUT
    x_usia = np.arange(0, 60, 1)
    if jk == "L":
        x_tb = np.arange(44, 130, 1)
    else:
        x_tb = np.arange(43, 130, 1)

    # VAR OUTPUT
    if jk == "L":
        x_tb_u = np.arange(45, 130, 1)
    else:
        x_tb_u = np.arange(44, 130, 1)

    usia_bayi = fuzz.trapmf(x_usia, [0, 0, 6, 11])
    usia_baduta = fuzz.trimf(x_usia, [9, 16, 23])
    usia_balita_1 = fuzz.trimf(x_usia, [21, 31, 41])
    usia_balita_2 = fuzz.trapmf(x_usia, [39, 49, 60, 60])

    # membership / keanggotaan USIA
    usia_tahap1 = fuzz.trapmf(x_usia, [0, 0, 6, 12])
    usia_tahap2 = fuzz.trimf(x_usia, [6, 12, 24])
    usia_tahap3 = fuzz.trimf(x_usia, [12, 24, 36])
    usia_tahap4 = fuzz.trimf(x_usia, [24, 36, 48])
    usia_tahap5 = fuzz.trapmf(x_usia, [36, 48, 60, 60])

    # membership / keanggotaan TINGGI BADAN
    if jk == "L":
        tb_sangat_pendek = fuzz.trapmf(x_tb, [44, 44, 48, 56])
        tb_pendek = fuzz.trimf(x_tb, [48, 65, 80])
        tb_normal = fuzz.trimf(x_tb, [66, 94, 126])
        tb_tinggi = fuzz.trapmf(x_tb, [110, 126, 130, 130])
    else:
        tb_sangat_pendek = fuzz.trapmf(x_tb, [43, 43, 47, 55])
        tb_pendek = fuzz.trimf(x_tb, [47, 64, 78])
        tb_normal = fuzz.trimf(x_tb, [64, 92, 124])
        tb_tinggi = fuzz.trapmf(x_tb, [108, 124, 130, 130])

    # membership / keanggotaan TB/U
    if jk == "L":
        tb_u_sangat_pendek = fuzz.trapmf(x_tb_u, [45, 45, 56, 67])
        tb_u_pendek = fuzz.trimf(x_tb_u, [56, 72, 88])
        tb_u_normal = fuzz.trimf(x_tb_u, [72, 98, 124])
        tb_u_tinggi = fuzz.trapmf(x_tb_u, [112, 124, 130, 130])
    else:
        tb_u_sangat_pendek = fuzz.trapmf(x_tb_u, [44, 44, 56, 66])
        tb_u_pendek = fuzz.trimf(x_tb_u, [56, 72, 86])
        tb_u_normal = fuzz.trimf(x_tb_u, [72, 95, 123])
        tb_u_tinggi = fuzz.trapmf(x_tb_u, [112, 123, 130, 130])

    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(8, 9))

    # ax0.plot(x_usia, usia_bayi, 'b', linewidth=1.5, label="Bayi")
    # ax0.plot(x_usia, usia_baduta, 'g', linewidth=1.5, label="Baduta")
    # ax0.plot(x_usia, usia_balita_1, 'r', linewidth=1.5, label="Balita 1")
    # ax0.plot(x_usia, usia_balita_2, 'm', linewidth=1.5, label="Balita ")
    # ax0.set_title("---- USIA ----")
    # ax0.legend()

    ax0.plot(x_usia, usia_tahap1, 'b', linewidth=1.5, label="Tahap 1")
    ax0.plot(x_usia, usia_tahap2, 'g', linewidth=1.5, label="Tahap 2")
    ax0.plot(x_usia, usia_tahap3, 'r', linewidth=1.5, label="Tahap 3")
    ax0.plot(x_usia, usia_tahap4, 'm', linewidth=1.5, label="Tahap 4")
    ax0.plot(x_usia, usia_tahap5, 'c', linewidth=1.5, label="Tahap 5")
    ax0.set_title("---- USIA ----")
    ax0.legend()

    ax1.plot(x_tb, tb_sangat_pendek, 'b', linewidth=1.5, label="Sangat Pendek")
    ax1.plot(x_tb, tb_pendek, 'g', linewidth=1.5, label="Pendek")
    ax1.plot(x_tb, tb_normal, 'r', linewidth=1.5, label="Normal")
    ax1.plot(x_tb, tb_tinggi, 'm', linewidth=1.5, label="Tinggi")
    ax1.set_title("---- TINGGI BADAN ----")
    ax1.legend()

    ax2.plot(x_tb_u, tb_u_sangat_pendek, 'b', linewidth=1.5, label="TB Sangat Pendek")
    ax2.plot(x_tb_u, tb_u_pendek, 'g', linewidth=1.5, label="TB Pendek")
    ax2.plot(x_tb_u, tb_u_normal, 'r', linewidth=1.5, label="TB Normal")
    ax2.plot(x_tb_u, tb_u_tinggi, 'm', linewidth=1.5, label="TB Tinggi")
    ax2.set_title("---- TB / U ----")
    ax2.legend()

    for ax in (ax0, ax1, ax2):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
    plt.tight_layout()

    # u_bayi = fuzz.interp_membership(x_usia, usia_bayi, usia)
    # u_baduta = fuzz.interp_membership(x_usia, usia_baduta, usia)
    # u_balita_1 = fuzz.interp_membership(x_usia, usia_balita_1, usia)
    # u_balita_2 = fuzz.interp_membership(x_usia, usia_balita_2, usia)

    u_tahap1 = fuzz.interp_membership(x_usia, usia_tahap1, usia)
    u_tahap2 = fuzz.interp_membership(x_usia, usia_tahap2, usia)
    u_tahap3 = fuzz.interp_membership(x_usia, usia_tahap3, usia)
    u_tahap4 = fuzz.interp_membership(x_usia, usia_tahap4, usia)
    u_tahap5 = fuzz.interp_membership(x_usia, usia_tahap5, usia)

    tb_sp = fuzz.interp_membership(x_tb, tb_sangat_pendek, tb)
    tb_p = fuzz.interp_membership(x_tb, tb_pendek, tb)
    tb_n = fuzz.interp_membership(x_tb, tb_normal, tb)
    tb_t = fuzz.interp_membership(x_tb, tb_tinggi, tb)

    # print(u_bayi, u_baduta, u_balita_1, u_balita_2)
    print(u_tahap1, u_tahap2, u_tahap3, u_tahap4, u_tahap5)
    print(tb_sp, tb_p, tb_n, tb_t)

    # drjt_usia = u_bayi, u_baduta, u_balita_1, u_balita_2
    drjt_usia = u_tahap1, u_tahap2, u_tahap3, u_tahap4, u_tahap5
    drjt_tb = tb_sp, tb_p, tb_n, tb_t

    rule1 = np.fmin(drjt_usia[0], drjt_tb[0])  # Rule 1 => tahap1 & sangat_pendek   = PENDEK
    rule2 = np.fmin(drjt_usia[0], drjt_tb[1])  # Rule 2 => tahap1 & pendek          = NORMAL
    rule3 = np.fmin(drjt_usia[0], drjt_tb[2])  # Rule 3 => tahap1 & normal          = TINGGI
    rule4 = np.fmin(drjt_usia[0], drjt_tb[3])  # Rule 4 => tahap1 & tinggi          = TINGGI

    rule5 = np.fmin(drjt_usia[1], drjt_tb[0])  # Rule 5 => tahap2 & sangat_pendek   = PENDEK
    rule6 = np.fmin(drjt_usia[1], drjt_tb[1])  # Rule 6 => tahap2 & pendek          = NORMAL
    rule7 = np.fmin(drjt_usia[1], drjt_tb[2])  # Rule 7 => tahap2 & normal          = TINGGI
    rule8 = np.fmin(drjt_usia[1], drjt_tb[3])  # Rule 8 => tahap2 & tinggi          = TINGGI

    rule9 = np.fmin(drjt_usia[2], drjt_tb[0])  # Rule 9 => tahap3 & sangat_pendek   = SANGAT PENDEK
    rule10 = np.fmin(drjt_usia[2], drjt_tb[1])  # Rule 10 => tahap3 & pendek        = PENDEK
    rule11 = np.fmin(drjt_usia[2], drjt_tb[2])  # Rule 11 => tahap3 & normal        = NORMAL
    rule12 = np.fmin(drjt_usia[2], drjt_tb[3])  # Rule 12 => tahap3 & tinggi        = TINGGI

    rule13 = np.fmin(drjt_usia[3], drjt_tb[0])  # Rule 9 => tahap4 & sangat_pendek   = SANGAT PENDEK
    rule14 = np.fmin(drjt_usia[3], drjt_tb[1])  # Rule 10 => tahap4 & pendek        = PENDEK
    rule15 = np.fmin(drjt_usia[3], drjt_tb[2])  # Rule 11 => tahap4 & normal        = NORMAL
    rule16 = np.fmin(drjt_usia[3], drjt_tb[3])  # Rule 12 => tahap4 & tinggi        = TINGGI

    rule17 = np.fmin(drjt_usia[4], drjt_tb[0])  # Rule 9 => tahap5 & sangat_pendek   = SANGAT PENDEK
    rule18 = np.fmin(drjt_usia[4], drjt_tb[1])  # Rule 10 => tahap5 & pendek        = PENDEK
    rule19 = np.fmin(drjt_usia[4], drjt_tb[2])  # Rule 11 => tahap5 & normal        = NORMAL
    rule20 = np.fmin(drjt_usia[4], drjt_tb[3])  # Rule 12 => tahap5 & tinggi        = TINGGI

    pk_tb_u_sangat_pendek = np.fmax(rule9, np.fmax(rule13, rule17))
    pk_tb_u_pendek = np.fmax(rule1, np.fmax(rule5, np.fmax(rule10, np.fmax(rule14, rule18))))
    pk_tb_u_normal = np.fmax(rule2, np.fmax(rule6, np.fmax(rule11, np.fmax(rule15, rule19))))
    pk_tb_u_tinggi = np.fmax(rule3,
                             np.fmax(rule4, np.fmax(rule7, np.fmax(rule8, np.fmax(rule12, np.fmax(rule16, rule20))))))

    print(pk_tb_u_sangat_pendek)
    print(pk_tb_u_pendek)
    print(pk_tb_u_normal)
    print(pk_tb_u_tinggi)

    pk_tb_u_sangat_pendek = np.fmin(pk_tb_u_sangat_pendek, tb_u_sangat_pendek)
    pk_tb_u_pendek = np.fmin(pk_tb_u_pendek, tb_u_pendek)
    pk_tb_u_normal = np.fmin(pk_tb_u_normal, tb_u_normal)
    pk_tb_u_tinggi = np.fmin(pk_tb_u_tinggi, tb_u_tinggi)

    pk_tb_u_0 = np.zeros_like(x_tb_u)
    pk_tb_u_sp = np.zeros_like(tb_u_sangat_pendek)
    pk_tb_u_p = np.zeros_like(tb_u_pendek)
    pk_tb_u_n = np.zeros_like(tb_u_normal)
    pk_tb_u_t = np.zeros_like(tb_u_tinggi)

    fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(nrows=5, figsize=(8, 9))

    ax0.fill_between(x_tb_u, pk_tb_u_sp, pk_tb_u_sangat_pendek, facecolor='b', alpha=0.7)
    ax0.plot(x_tb_u, tb_u_sangat_pendek, 'b', linewidth=1.5, linestyle='--', label="Sangat Pendek")
    ax0.set_title('TB/U')
    ax0.legend()

    ax1.fill_between(x_tb_u, pk_tb_u_p, pk_tb_u_pendek, facecolor='b', alpha=0.7)
    ax1.plot(x_tb_u, tb_u_pendek, 'b', linewidth=1.5, linestyle='--', label="Pendek")
    ax1.set_title('TB/U')
    ax1.legend()

    ax2.fill_between(x_tb_u, pk_tb_u_n, pk_tb_u_normal, facecolor='b', alpha=0.7)
    ax2.plot(x_tb_u, tb_u_normal, 'b', linewidth=1.5, linestyle='--', label="Normal")
    ax2.set_title('TB/U')
    ax2.legend()

    ax3.fill_between(x_tb_u, pk_tb_u_t, pk_tb_u_normal, facecolor='b', alpha=0.7)
    ax3.plot(x_tb_u, tb_u_tinggi, 'b', linewidth=1.5, linestyle='--', label="Tinggi")
    ax3.set_title('TB/U')
    ax3.legend()

    ax4.fill_between(x_tb_u, pk_tb_u_0, pk_tb_u_sangat_pendek, facecolor='b', alpha=0.7)
    ax4.plot(x_tb_u, tb_u_sangat_pendek, 'b', linewidth=1.5, linestyle='--', label="Sangat Pendek")
    ax4.fill_between(x_tb_u, pk_tb_u_0, pk_tb_u_pendek, facecolor='g', alpha=0.7)
    ax4.plot(x_tb_u, tb_u_pendek, 'g', linewidth=1.5, linestyle='--', label="Pendek")
    ax4.fill_between(x_tb_u, pk_tb_u_0, pk_tb_u_normal, facecolor='r', alpha=0.7)
    ax4.plot(x_tb_u, tb_u_normal, 'r', linewidth=1.5, linestyle='--', label="Normal")
    ax4.fill_between(x_tb_u, pk_tb_u_0, pk_tb_u_normal, facecolor='m', alpha=0.7)
    ax4.plot(x_tb_u, tb_u_tinggi, 'm', linewidth=1.5, linestyle='--', label="Tinggi")
    ax4.set_title('Tinggi Badan / Usia')
    ax4.legend()

    for ax in (ax0, ax1, ax2, ax3, ax4):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
    plt.tight_layout()

    komposisi = np.fmax(pk_tb_u_sangat_pendek,
                        np.fmax(pk_tb_u_pendek, np.fmax(pk_tb_u_normal, pk_tb_u_tinggi)))

    tinggi_badan_per_usia = fuzz.defuzz(x_tb_u, komposisi, 'centroid')

    plt.plot([tinggi_badan_per_usia, tinggi_badan_per_usia],
             [0, fuzz.interp_membership(x_tb_u, komposisi, tinggi_badan_per_usia)], 'k',
             linewidth=1.5, alpha=0.9)

    # Calculate membership values for the defuzzified result
    tb_u_sangat_pendek_degree = fuzz.interp_membership(x_tb_u, tb_u_sangat_pendek, tinggi_badan_per_usia)
    tb_u_pendek_degree = fuzz.interp_membership(x_tb_u, tb_u_pendek, tinggi_badan_per_usia)
    tb_u_normal_degree = fuzz.interp_membership(x_tb_u, tb_u_normal, tinggi_badan_per_usia)
    tb_u_tinggi_degree = fuzz.interp_membership(x_tb_u, tb_u_tinggi, tinggi_badan_per_usia)

    status_gizi = 'Tidak Terdefinisi'
    max_keanggotaan = max(tb_u_sangat_pendek_degree, tb_u_pendek_degree, tb_u_normal_degree,
                          tb_u_tinggi_degree)
    if max_keanggotaan == tb_u_sangat_pendek_degree:
        status_gizi = 'Sangat Pendek'
    elif max_keanggotaan == tb_u_pendek_degree:
        status_gizi = 'Pendek'
    elif max_keanggotaan == tb_u_normal_degree:
        status_gizi = 'Normal'
    elif max_keanggotaan == tb_u_tinggi_degree:
        status_gizi = 'Tinggi'

    results = {
        'defuzzified_value': tinggi_badan_per_usia,
        'status_gizi': status_gizi,
        'tb_u_sangat_pendek_degree': tb_u_sangat_pendek_degree,
        'tb_u_pendek_degree': tb_u_pendek_degree,
        'tb_u_normal_degree': tb_u_normal_degree,
        'tb_u_tinggi_degree': tb_u_tinggi_degree
    }

    print(results)

    # SHOW PLOT OR GRAPH
    # plt.show()

    return results


def countBB_TB(bb, tb, jk):
    # VAR INPUT
    x_bb = np.arange(2, 28, 1)
    if jk == "L":
        x_tb = np.arange(44, 130, 1)
    else:
        x_tb = np.arange(43, 130, 1)

    # VAR OUTPUT
    if jk == "L":
        x_bb_tb = np.arange(3, 40, 1)
    else:
        x_bb_tb = np.arange(3, 38, 1)

    # membership / keanggotaan BERAT BADAN
    bb_sangat_kurang = fuzz.trapmf(x_bb, [2, 2, 4, 6])
    if jk == "L":
        bb_kurang = fuzz.trimf(x_bb, [4, 8, 12])
        bb_normal = fuzz.trimf(x_bb, [8, 13, 18])
        bb_lebih = fuzz.trimf(x_bb, [15, 18, 23])
        bb_sangat_lebih = fuzz.trapmf(x_bb, [21, 23, 28, 28])
    else:
        bb_kurang = fuzz.trimf(x_bb, [4, 8, 11])
        bb_normal = fuzz.trimf(x_bb, [9, 13, 18])
        bb_lebih = fuzz.trimf(x_bb, [15, 20, 24])
        bb_sangat_lebih = fuzz.trapmf(x_bb, [22, 25, 28, 28])

    # membership / keanggotaan TINGGI BADAN
    if jk == "L":
        tb_sangat_pendek = fuzz.trapmf(x_tb, [44, 44, 48, 56])
        tb_pendek = fuzz.trimf(x_tb, [48, 65, 80])
        tb_normal = fuzz.trimf(x_tb, [66, 94, 126])
        tb_tinggi = fuzz.trapmf(x_tb, [110, 126, 130, 130])
    else:
        tb_sangat_pendek = fuzz.trapmf(x_tb, [43, 43, 47, 55])
        tb_pendek = fuzz.trimf(x_tb, [47, 64, 78])
        tb_normal = fuzz.trimf(x_tb, [64, 92, 124])
        tb_tinggi = fuzz.trapmf(x_tb, [108, 124, 130, 130])

    # membership / keanggotaan BB-TB

    # Gizi Buruk	        3 - 10	    3 - 8	    3,7,10	    3,6,8
    # Gizi Kurang	        7 - 14	    6 - 12	    7,12,14	    6,9,12
    # Gizi Baik	            12 - 25	    10 - 24 	12,19,25	10,16,24
    # Beresiko Gizi Lebih	21 - 27 	19 - 25 	21,23,27	19,22,25
    # Gizi Lebih	        25 - 29	    24 - 28 	25,27,29	24,26,28
    # Obesitas	            28 - 34 	28 - 38	    28,34,40	28,32,38

    if jk == "L":
        bb_tb_gizi_buruk = fuzz.trapmf(x_bb_tb, [3, 3, 7, 10])
        bb_tb_gizi_kurang = fuzz.trimf(x_bb_tb, [7, 12, 14])
        bb_tb_gizi_baik = fuzz.trimf(x_bb_tb, [12, 19, 25])
        bb_tb_beresiko_gizi_lebih = fuzz.trimf(x_bb_tb, [21, 23, 27])
        bb_tb_gizi_lebih = fuzz.trimf(x_bb_tb, [25, 27, 29])
        bb_tb_gizi_obesitas = fuzz.trapmf(x_bb_tb, [28, 34, 40, 40])
    else:
        bb_tb_gizi_buruk = fuzz.trapmf(x_bb_tb, [3, 3, 6, 8])
        bb_tb_gizi_kurang = fuzz.trimf(x_bb_tb, [6, 9, 12])
        bb_tb_gizi_baik = fuzz.trimf(x_bb_tb, [10, 16, 24])
        bb_tb_beresiko_gizi_lebih = fuzz.trimf(x_bb_tb, [19, 22, 25])
        bb_tb_gizi_lebih = fuzz.trimf(x_bb_tb, [24, 26, 28])
        bb_tb_gizi_obesitas = fuzz.trapmf(x_bb_tb, [28, 32, 38, 38])

    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(8, 9))

    ax0.plot(x_bb, bb_sangat_kurang, 'b', linewidth=1.5, label="Sangat Kurang")
    ax0.plot(x_bb, bb_kurang, 'g', linewidth=1.5, label="Kurang")
    ax0.plot(x_bb, bb_normal, 'r', linewidth=1.5, label="Normal")
    ax0.plot(x_bb, bb_lebih, 'm', linewidth=1.5, label="Lebih")
    ax0.plot(x_bb, bb_sangat_lebih, 'c', linewidth=1.5, label="Sangat Lebih")
    ax0.set_title("---- Berat Badan ----")
    ax0.legend()

    ax1.plot(x_tb, tb_sangat_pendek, 'b', linewidth=1.5, label="Sangat Pendek")
    ax1.plot(x_tb, tb_pendek, 'g', linewidth=1.5, label="Pendek")
    ax1.plot(x_tb, tb_normal, 'r', linewidth=1.5, label="Normal")
    ax1.plot(x_tb, tb_tinggi, 'm', linewidth=1.5, label="Tinggi")
    ax1.set_title("---- TINGGI BADAN ----")
    ax1.legend()

    ax2.plot(x_bb_tb, bb_tb_gizi_buruk, 'b', linewidth=1.5, label="Gizi Buruk")
    ax2.plot(x_bb_tb, bb_tb_gizi_kurang, 'g', linewidth=1.5, label="Gizi Kurang")
    ax2.plot(x_bb_tb, bb_tb_gizi_baik, 'r', linewidth=1.5, label="Gizi Baik / Normal")
    ax2.plot(x_bb_tb, bb_tb_beresiko_gizi_lebih, 'm', linewidth=1.5, label="Beresiko Gizi Lebih")
    ax2.plot(x_bb_tb, bb_tb_gizi_lebih, 'c', linewidth=1.5, label="Gizi Lebih")
    ax2.plot(x_bb_tb, bb_tb_gizi_obesitas, 'y', linewidth=1.5, label="Obesitas")
    ax2.set_title("---- BB / TB ----")
    ax2.legend()

    for ax in (ax0, ax1, ax2):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
    plt.tight_layout()

    bb_sk = fuzz.interp_membership(x_bb, bb_sangat_kurang, bb)
    bb_k = fuzz.interp_membership(x_bb, bb_kurang, bb)
    bb_n = fuzz.interp_membership(x_bb, bb_normal, bb)
    bb_l = fuzz.interp_membership(x_bb, bb_lebih, bb)
    bb_sl = fuzz.interp_membership(x_bb, bb_sangat_lebih, bb)

    tb_sp = fuzz.interp_membership(x_tb, tb_sangat_pendek, tb)
    tb_p = fuzz.interp_membership(x_tb, tb_pendek, tb)
    tb_n = fuzz.interp_membership(x_tb, tb_normal, tb)
    tb_t = fuzz.interp_membership(x_tb, tb_tinggi, tb)

    print(bb_sk, bb_k, bb_n, bb_l, bb_sl)
    print(tb_sp, tb_p, tb_n, tb_t)

    drjt_bb = bb_sk, bb_k, bb_n, bb_l, bb_sl
    drjt_tb = tb_sp, tb_p, tb_n, tb_t

    # Rule      => usia, bb = output aturan     # Rule   => BERAT BADAN & TINGGI BADAN      = OUTPUT RULE

    rule1 = np.fmin(drjt_bb[0], drjt_tb[0])  # Rule 1 => sangat_kurang & sangat_pendek   = GIZI KURANG
    rule2 = np.fmin(drjt_bb[0], drjt_tb[1])  # Rule 2 => sangat_kurang & pendek          = GIZI KURANG
    rule3 = np.fmin(drjt_bb[0], drjt_tb[2])  # Rule 3 => sangat_kurang & normal          = GIZI KURANG
    rule4 = np.fmin(drjt_bb[0], drjt_tb[3])  # Rule 4 => sangat_kurang & tinggi          = GIZI BURUK

    rule5 = np.fmin(drjt_bb[1], drjt_tb[0])  # Rule 5 => kurang & sangat_pendek          = GIZI KURANG
    rule6 = np.fmin(drjt_bb[1], drjt_tb[1])  # Rule 6 => kurang & pendek                 = NORMAL
    rule7 = np.fmin(drjt_bb[1], drjt_tb[2])  # Rule 7 => kurang & normal                 = GIZI KURANG
    rule8 = np.fmin(drjt_bb[1], drjt_tb[3])  # Rule 8 => kurang & tinggi                 = GIZI KURANG

    rule9 = np.fmin(drjt_bb[2], drjt_tb[0])  # Rule 9  => normal & sangat_pendek         = GIZI LEBIH
    rule10 = np.fmin(drjt_bb[2], drjt_tb[1])  # Rule 10 => normal & pendek                = NORMAL
    rule11 = np.fmin(drjt_bb[2], drjt_tb[2])  # Rule 11 => normal & normal                = NORMAL
    rule12 = np.fmin(drjt_bb[2], drjt_tb[3])  # Rule 12 => normal & tinggi                = GIZI KURANG

    rule13 = np.fmin(drjt_bb[3], drjt_tb[0])  # Rule 13 => lebih & sangat_pendek          = OBESITAS
    rule14 = np.fmin(drjt_bb[3], drjt_tb[1])  # Rule 14 => lebih & pendek                 = GIZI LEBIH
    rule15 = np.fmin(drjt_bb[3], drjt_tb[2])  # Rule 15 => lebih & normal                 = BERESIKO GIZI LEBIH
    rule16 = np.fmin(drjt_bb[3], drjt_tb[3])  # Rule 16 => lebih & tinggi                 = NORMAL

    rule17 = np.fmin(drjt_bb[4], drjt_tb[0])  # Rule 13 => sangat_lebih & sangat_pendek   = OBESITAS
    rule18 = np.fmin(drjt_bb[4], drjt_tb[1])  # Rule 14 => sangat_lebih & pendek          = OBESITAS
    rule19 = np.fmin(drjt_bb[4], drjt_tb[2])  # Rule 15 => sangat_lebih & normal          = BERESIKO GIZI LEBIH
    rule20 = np.fmin(drjt_bb[4], drjt_tb[3])  # Rule 16 => sangat_lebih & tinggi          = BERESIKO GIZI LEBIH

    pk_bb_tb_gz_buruk = rule4
    pk_bb_tb_gz_kurang = np.fmax(rule1,
                                 np.fmax(rule2, np.fmax(rule3, np.fmax(rule5, np.fmax(rule7, np.fmax(rule8, rule12))))))
    pk_bb_tb_gz_baik_or_normal = np.fmax(rule6, np.fmax(rule10, np.fmax(rule11, rule16)))
    pk_bb_tb_gz_beresiko_lebih = np.fmax(rule15, np.fmax(rule19, rule20))
    pk_bb_tb_gz_lebih = np.fmax(rule9, rule14)
    pk_bb_tb_gz_obesitas = np.fmax(rule13, np.fmax(rule17, rule18))

    print(pk_bb_tb_gz_buruk)
    print(pk_bb_tb_gz_kurang)
    print(pk_bb_tb_gz_baik_or_normal)
    print(pk_bb_tb_gz_beresiko_lebih)
    print(pk_bb_tb_gz_lebih)
    print(pk_bb_tb_gz_obesitas)

    pk_bb_tb_gz_buruk = np.fmin(pk_bb_tb_gz_buruk, bb_tb_gizi_buruk)
    pk_bb_tb_gz_kurang = np.fmin(pk_bb_tb_gz_kurang, bb_tb_gizi_kurang)
    pk_bb_tb_gz_baik_or_normal = np.fmin(pk_bb_tb_gz_baik_or_normal, bb_tb_gizi_baik)
    pk_bb_tb_gz_beresiko_lebih = np.fmin(pk_bb_tb_gz_beresiko_lebih, bb_tb_beresiko_gizi_lebih)
    pk_bb_tb_gz_lebih = np.fmin(pk_bb_tb_gz_lebih, bb_tb_gizi_lebih)
    pk_bb_tb_gz_obesitas = np.fmin(pk_bb_tb_gz_obesitas, bb_tb_gizi_obesitas)

    pk_tb_u_0 = np.zeros_like(x_bb_tb)
    pk_bb_tb_gz_b = np.zeros_like(bb_tb_gizi_buruk)
    pk_bb_tb_gz_k = np.zeros_like(bb_tb_gizi_kurang)
    pk_bb_tb_gz_n = np.zeros_like(bb_tb_gizi_baik)
    pk_bb_tb_gz_bl = np.zeros_like(bb_tb_beresiko_gizi_lebih)
    pk_bb_tb_gz_l = np.zeros_like(bb_tb_gizi_lebih)
    pk_bb_tb_gz_ob = np.zeros_like(bb_tb_gizi_obesitas)

    fig, (ax0, ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(nrows=7, figsize=(8, 9))

    ax0.fill_between(x_bb_tb, pk_bb_tb_gz_b, pk_bb_tb_gz_buruk, facecolor='b', alpha=0.7)
    ax0.plot(x_bb_tb, bb_tb_gizi_buruk, 'b', linewidth=1.5, linestyle='--', label="Gizi Buruk")
    ax0.set_title('BB/TB')
    ax0.legend()

    ax1.fill_between(x_bb_tb, pk_bb_tb_gz_k, pk_bb_tb_gz_kurang, facecolor='b', alpha=0.7)
    ax1.plot(x_bb_tb, bb_tb_gizi_kurang, 'b', linewidth=1.5, linestyle='--', label="Gizi Kurang")
    ax1.set_title('BB/TB')
    ax1.legend()

    ax2.fill_between(x_bb_tb, pk_bb_tb_gz_n, pk_bb_tb_gz_baik_or_normal, facecolor='b', alpha=0.7)
    ax2.plot(x_bb_tb, bb_tb_gizi_baik, 'b', linewidth=1.5, linestyle='--', label="Gizi Baik / Normal")
    ax2.set_title('BB/TB')
    ax2.legend()

    ax3.fill_between(x_bb_tb, pk_bb_tb_gz_bl, pk_bb_tb_gz_beresiko_lebih, facecolor='b', alpha=0.7)
    ax3.plot(x_bb_tb, bb_tb_beresiko_gizi_lebih, 'b', linewidth=1.5, linestyle='--', label="Beresiko Gizi Lebih")
    ax3.set_title('BB/TB')
    ax3.legend()

    ax4.fill_between(x_bb_tb, pk_bb_tb_gz_l, pk_bb_tb_gz_lebih, facecolor='b', alpha=0.7)
    ax4.plot(x_bb_tb, bb_tb_gizi_lebih, 'b', linewidth=1.5, linestyle='--', label="Gizi Lebih")
    ax4.set_title('BB/TB')
    ax4.legend()

    ax5.fill_between(x_bb_tb, pk_bb_tb_gz_ob, pk_bb_tb_gz_obesitas, facecolor='b', alpha=0.7)
    ax5.plot(x_bb_tb, bb_tb_gizi_obesitas, 'b', linewidth=1.5, linestyle='--', label="Obesitas")
    ax5.set_title('BB/TB')
    ax5.legend()

    ax6.fill_between(x_bb_tb, pk_bb_tb_gz_b, pk_bb_tb_gz_buruk, facecolor='b', alpha=0.7)
    ax6.plot(x_bb_tb, bb_tb_gizi_buruk, 'b', linewidth=1.5, linestyle='--', label="Gizi Buruk")
    ax6.fill_between(x_bb_tb, pk_bb_tb_gz_k, pk_bb_tb_gz_kurang, facecolor='g', alpha=0.7)
    ax6.plot(x_bb_tb, bb_tb_gizi_kurang, 'g', linewidth=1.5, linestyle='--', label="Gizi Kurang")
    ax6.fill_between(x_bb_tb, pk_bb_tb_gz_n, pk_bb_tb_gz_baik_or_normal, facecolor='r', alpha=0.7)
    ax6.plot(x_bb_tb, bb_tb_gizi_baik, 'r', linewidth=1.5, linestyle='--', label="Gizi Baik / Normal")
    ax6.fill_between(x_bb_tb, pk_bb_tb_gz_bl, pk_bb_tb_gz_beresiko_lebih, facecolor='m', alpha=0.7)
    ax6.plot(x_bb_tb, bb_tb_beresiko_gizi_lebih, 'm', linewidth=1.5, linestyle='--', label="Beresiko Gizi Lebih")
    ax6.fill_between(x_bb_tb, pk_bb_tb_gz_l, pk_bb_tb_gz_lebih, facecolor='c', alpha=0.7)
    ax6.plot(x_bb_tb, bb_tb_gizi_lebih, 'c', linewidth=1.5, linestyle='--', label="Gizi Lebih")
    ax6.fill_between(x_bb_tb, pk_bb_tb_gz_ob, pk_bb_tb_gz_obesitas, facecolor='y', alpha=0.7)
    ax6.plot(x_bb_tb, bb_tb_gizi_obesitas, 'y', linewidth=1.5, linestyle='--', label="Obesitas")
    ax6.set_title('Tinggi Badan / Usia')
    ax6.legend()

    for ax in (ax0, ax1, ax2, ax3, ax4, ax5, ax6):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
    plt.tight_layout()

    komposisi = np.fmax(pk_bb_tb_gz_buruk, np.fmax(pk_bb_tb_gz_kurang, np.fmax(pk_bb_tb_gz_baik_or_normal,
                                                                               np.fmax(pk_bb_tb_gz_beresiko_lebih,
                                                                                       np.fmax(pk_bb_tb_gz_lebih,
                                                                                               pk_bb_tb_gz_obesitas)))))

    berat_badan_per_tinggi_badan = fuzz.defuzz(x_bb_tb, komposisi, 'centroid')

    plt.plot([berat_badan_per_tinggi_badan, berat_badan_per_tinggi_badan],
             [0, fuzz.interp_membership(x_bb_tb, komposisi, berat_badan_per_tinggi_badan)], 'k',
             linewidth=1.5, alpha=0.9)

    # Calculate membership values for the defuzzified result
    bb_tb_gizi_buruk_degree = fuzz.interp_membership(x_bb_tb, bb_tb_gizi_buruk, berat_badan_per_tinggi_badan)
    bb_tb_gizi_kurang_degree = fuzz.interp_membership(x_bb_tb, bb_tb_gizi_kurang, berat_badan_per_tinggi_badan)
    bb_tb_gizi_normal_degree = fuzz.interp_membership(x_bb_tb, bb_tb_gizi_baik, berat_badan_per_tinggi_badan)
    bb_tb_brsk_gizi_lebih_degree = fuzz.interp_membership(x_bb_tb, bb_tb_beresiko_gizi_lebih,
                                                          berat_badan_per_tinggi_badan)
    bb_tb_gizi_lebih_degree = fuzz.interp_membership(x_bb_tb, bb_tb_gizi_lebih, berat_badan_per_tinggi_badan)
    bb_tb_gizi_obesitas_degree = fuzz.interp_membership(x_bb_tb, bb_tb_gizi_obesitas, berat_badan_per_tinggi_badan)

    status_gizi = 'Tidak Terdefinisi'
    max_keanggotaan = max(bb_tb_gizi_buruk_degree, bb_tb_gizi_kurang_degree, bb_tb_gizi_normal_degree,
                          bb_tb_brsk_gizi_lebih_degree, bb_tb_gizi_lebih_degree, bb_tb_gizi_obesitas_degree)
    if max_keanggotaan == bb_tb_gizi_buruk_degree:
        status_gizi = 'Gizi Buruk'
    elif max_keanggotaan == bb_tb_gizi_kurang_degree:
        status_gizi = 'Gizi Kurang'
    elif max_keanggotaan == bb_tb_gizi_normal_degree:
        status_gizi = 'Gizi Baik / Normal'
    elif max_keanggotaan == bb_tb_brsk_gizi_lebih_degree:
        status_gizi = 'Beresiko Gizi Lebih'
    elif max_keanggotaan == bb_tb_gizi_lebih_degree:
        status_gizi = 'Gizi Lebih'
    elif max_keanggotaan == bb_tb_gizi_obesitas_degree:
        status_gizi = 'Gizi Obesitas'

    results = {
        'defuzzified_value': berat_badan_per_tinggi_badan,
        'satatus gizi': status_gizi,
        'bb_tb_gizi_buruk_degree': bb_tb_gizi_buruk_degree,
        'bb_tb_gizi_kurang_degree': bb_tb_gizi_kurang_degree,
        'bb_tb_gizi_normal_degree': bb_tb_gizi_normal_degree,
        'bb_tb_brsk_gizi_lebih_degree': bb_tb_brsk_gizi_lebih_degree,
        'bb_tb_gizi_lebih_degree': bb_tb_gizi_lebih_degree,
        'bb_tb_gizi_obesitas_degree': bb_tb_gizi_obesitas_degree,
    }

    print(results)

    # SHOW PLOT OR GRAPH
    # plt.show()

    return results


# DATA TEST =>  usia    berat   tinggi  jk
# Aurio Jalud   32      15.4      94      L
# Aurio Jalud   29      19      87      L

# TES BB/U
print(countBB_U(32, 15.4, "L"))

# TES TB/U
print(countTB_U(32, 94, "L"))


# TES TB/U
print(countBB_TB(15.4, 94, "L"))


@app.post("/calculate_bb_u")
def calculate_bb_u(input_data: InputData):
    result_bb_u = countBB_U(input_data.usia, input_data.bb, input_data.jk)
    result_tb_u = countTB_U(input_data.usia, input_data.tb, input_data.jk)
    result_bb_tb = countBB_TB(input_data.bb, input_data.tb, input_data.jk)

    result = {
        'usia': input_data.usia,
        'berat_badan': input_data.bb,
        'tinggi_badan': input_data.tb,
        'hasil': {
            'bb_u': result_bb_u,
            'tb_u': result_tb_u,
            'bb_tb': result_bb_tb
        }
    }

    return result
