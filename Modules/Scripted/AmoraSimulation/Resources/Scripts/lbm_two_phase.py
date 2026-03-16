#!/usr/bin/env python
"""
LBM D3Q19 MRT Two-Phase Flow Solver (Color Gradient Model)
============================================================
Based on taichi_LBM3D by yjhp1016.

Two immiscible fluids with interfacial tension via color gradient method.
Supports phase-dependent viscosity, recoloring, and wetting at solid surfaces.

Outputs velocity, phase field, and density as numpy arrays + optional VTK.
Prints JSON progress to stdout for AMORA GUI integration.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

try:
    import taichi as ti
except ImportError:
    print(json.dumps({"error": "taichi not installed. Run: pip install taichi"}))
    sys.exit(1)


# =========================================================================
# D3Q19 MRT Two-Phase LBM Solver (Color Gradient)
# =========================================================================

@ti.data_oriented
class LBM3D_TwoPhase:

    def __init__(self, nx, ny, nz):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.fx, self.fy, self.fz = 0.0, 0.0, 0.0

        # Phase-dependent viscosity
        self.niu_l = 0.1   # liquid (psi > 0)
        self.niu_g = 0.1   # gas (psi < 0)
        self.psi_solid = 0.7  # wetting parameter at solid surfaces
        self.CapA = 0.005     # surface tension parameter

        # Phase-field BCs: 0=periodic, 1=fixed
        self.bc_psi_x_left, self.psi_x_left = 0, -1.0
        self.bc_psi_x_right, self.psi_x_right = 0, 1.0
        self.bc_psi_y_left, self.psi_y_left = 0, 1.0
        self.bc_psi_y_right, self.psi_y_right = 0, 1.0
        self.bc_psi_z_left, self.psi_z_left = 0, 1.0
        self.bc_psi_z_right, self.psi_z_right = 0, 1.0

        # Flow BCs
        self.bc_x_left, self.rho_bcxl = 0, 1.0
        self.bc_x_right, self.rho_bcxr = 0, 1.0
        self.bc_y_left, self.rho_bcyl = 0, 1.0
        self.bc_y_right, self.rho_bcyr = 0, 1.0
        self.bc_z_left, self.rho_bczl = 0, 1.0
        self.bc_z_right, self.rho_bczr = 0, 1.0

        self.max_v = ti.field(ti.f32, shape=())

        # Fields
        self.f = ti.field(ti.f32, shape=(nx, ny, nz, 19))
        self.F = ti.field(ti.f32, shape=(nx, ny, nz, 19))
        self.rho = ti.field(ti.f32, shape=(nx, ny, nz))
        self.v = ti.Vector.field(3, ti.f32, shape=(nx, ny, nz))

        self.psi = ti.field(ti.f32, shape=(nx, ny, nz))
        self.rho_r = ti.field(ti.f32, shape=(nx, ny, nz))
        self.rho_b = ti.field(ti.f32, shape=(nx, ny, nz))
        self.rhor = ti.field(ti.f32, shape=(nx, ny, nz))
        self.rhob = ti.field(ti.f32, shape=(nx, ny, nz))

        self.e = ti.Vector.field(3, ti.i32, shape=(19,))
        self.e_f = ti.Vector.field(3, ti.f32, shape=(19,))
        self.w = ti.field(ti.f32, shape=(19,))
        self.solid = ti.field(ti.i8, shape=(nx, ny, nz))
        self.LR = ti.field(ti.i32, shape=(19,))
        self.ext_f = ti.Vector.field(3, ti.f32, shape=())

        self.M_f = ti.field(ti.f32, shape=(19, 19))
        self.inv_M_f = ti.field(ti.f32, shape=(19, 19))

        M_np = np.array([
            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [-1,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1],
            [1,-2,-2,-2,-2,-2,-2,1,1,1,1,1,1,1,1,1,1,1,1],
            [0,1,-1,0,0,0,0,1,-1,1,-1,1,-1,1,-1,0,0,0,0],
            [0,-2,2,0,0,0,0,1,-1,1,-1,1,-1,1,-1,0,0,0,0],
            [0,0,0,1,-1,0,0,1,-1,-1,1,0,0,0,0,1,-1,1,-1],
            [0,0,0,-2,2,0,0,1,-1,-1,1,0,0,0,0,1,-1,1,-1],
            [0,0,0,0,0,1,-1,0,0,0,0,1,-1,-1,1,1,-1,-1,1],
            [0,0,0,0,0,-2,2,0,0,0,0,1,-1,-1,1,1,-1,-1,1],
            [0,2,2,-1,-1,-1,-1,1,1,1,1,1,1,1,1,-2,-2,-2,-2],
            [0,-2,-2,1,1,1,1,1,1,1,1,1,1,1,1,-2,-2,-2,-2],
            [0,0,0,1,1,-1,-1,1,1,1,1,-1,-1,-1,-1,0,0,0,0],
            [0,0,0,-1,-1,1,1,1,1,1,1,-1,-1,-1,-1,0,0,0,0],
            [0,0,0,0,0,0,0,1,1,-1,-1,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,-1,-1],
            [0,0,0,0,0,0,0,0,0,0,0,1,1,-1,-1,0,0,0,0],
            [0,0,0,0,0,0,0,1,-1,1,-1,-1,1,-1,1,0,0,0,0],
            [0,0,0,0,0,0,0,-1,1,1,-1,0,0,0,0,1,-1,1,-1],
            [0,0,0,0,0,0,0,0,0,0,0,1,-1,-1,1,-1,1,1,-1]
        ], dtype=np.float32)
        inv_M_np = np.linalg.inv(M_np).astype(np.float32)

        LR_np = np.array([0,2,1,4,3,6,5,8,7,10,9,12,11,14,13,16,15,18,17], dtype=np.int32)

        self.M_f.from_numpy(M_np)
        self.inv_M_f.from_numpy(inv_M_np)
        self.LR.from_numpy(LR_np)

    def init_simulation(self):
        self.ext_f[None][0] = self.fx
        self.ext_f[None][1] = self.fy
        self.ext_f[None][2] = self.fz

        # Phase-dependent relaxation parameters
        self.wl = 1.0 / (self.niu_l / (1.0/3.0) + 0.5)
        self.wg = 1.0 / (self.niu_g / (1.0/3.0) + 0.5)
        self.lg0 = 2 * self.wl * self.wg / (self.wl + self.wg)
        self.l1 = 2 * (self.wl - self.lg0) * 10
        self.l2 = -self.l1 / 0.2
        self.g1 = 2 * (self.lg0 - self.wg) * 10
        self.g2 = self.g1 / 0.2

        self.static_init()
        self.init()

    @ti.func
    def feq(self, k, rho_local, u):
        eu = self.e[k].dot(u)
        uv = u.dot(u)
        return self.w[k] * rho_local * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * uv)

    @ti.kernel
    def init(self):
        for i, j, k in self.solid:
            if self.solid[i, j, k] == 0:
                self.rho[i, j, k] = 1.0
                self.v[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
                self.rho_r[i, j, k] = (self.psi[i, j, k] + 1.0) / 2.0
                self.rho_b[i, j, k] = 1.0 - self.rho_r[i, j, k]
                self.rhor[i, j, k] = 0.0
                self.rhob[i, j, k] = 0.0
                for s in ti.static(range(19)):
                    self.f[i, j, k, s] = self.feq(s, 1.0, self.v[i, j, k])
                    self.F[i, j, k, s] = self.feq(s, 1.0, self.v[i, j, k])

    @ti.kernel
    def static_init(self):
        self.e[0] = ti.Vector([0,0,0])
        self.e[1] = ti.Vector([1,0,0]); self.e[2] = ti.Vector([-1,0,0])
        self.e[3] = ti.Vector([0,1,0]); self.e[4] = ti.Vector([0,-1,0])
        self.e[5] = ti.Vector([0,0,1]); self.e[6] = ti.Vector([0,0,-1])
        self.e[7] = ti.Vector([1,1,0]); self.e[8] = ti.Vector([-1,-1,0])
        self.e[9] = ti.Vector([1,-1,0]); self.e[10] = ti.Vector([-1,1,0])
        self.e[11] = ti.Vector([1,0,1]); self.e[12] = ti.Vector([-1,0,-1])
        self.e[13] = ti.Vector([1,0,-1]); self.e[14] = ti.Vector([-1,0,1])
        self.e[15] = ti.Vector([0,1,1]); self.e[16] = ti.Vector([0,-1,-1])
        self.e[17] = ti.Vector([0,1,-1]); self.e[18] = ti.Vector([0,-1,1])

        self.e_f[0] = ti.Vector([0.0,0.0,0.0])
        self.e_f[1] = ti.Vector([1.0,0.0,0.0]); self.e_f[2] = ti.Vector([-1.0,0.0,0.0])
        self.e_f[3] = ti.Vector([0.0,1.0,0.0]); self.e_f[4] = ti.Vector([0.0,-1.0,0.0])
        self.e_f[5] = ti.Vector([0.0,0.0,1.0]); self.e_f[6] = ti.Vector([0.0,0.0,-1.0])
        self.e_f[7] = ti.Vector([1.0,1.0,0.0]); self.e_f[8] = ti.Vector([-1.0,-1.0,0.0])
        self.e_f[9] = ti.Vector([1.0,-1.0,0.0]); self.e_f[10] = ti.Vector([-1.0,1.0,0.0])
        self.e_f[11] = ti.Vector([1.0,0.0,1.0]); self.e_f[12] = ti.Vector([-1.0,0.0,-1.0])
        self.e_f[13] = ti.Vector([1.0,0.0,-1.0]); self.e_f[14] = ti.Vector([-1.0,0.0,1.0])
        self.e_f[15] = ti.Vector([0.0,1.0,1.0]); self.e_f[16] = ti.Vector([0.0,-1.0,-1.0])
        self.e_f[17] = ti.Vector([0.0,1.0,-1.0]); self.e_f[18] = ti.Vector([0.0,-1.0,1.0])

        self.w[0] = 1.0/3.0
        for s in ti.static(range(1, 7)):
            self.w[s] = 1.0/18.0
        for s in ti.static(range(7, 19)):
            self.w[s] = 1.0/36.0

    @ti.func
    def periodic_index(self, i):
        iout = i
        if i[0] < 0: iout[0] = self.nx - 1
        if i[0] > self.nx - 1: iout[0] = 0
        if i[1] < 0: iout[1] = self.ny - 1
        if i[1] > self.ny - 1: iout[1] = 0
        if i[2] < 0: iout[2] = self.nz - 1
        if i[2] > self.nz - 1: iout[2] = 0
        return iout

    @ti.func
    def periodic_index_psi(self, i):
        iout = i
        if i[0] < 0:
            iout[0] = self.nx - 1 if ti.static(self.bc_psi_x_left == 0) else 0
        if i[0] > self.nx - 1:
            iout[0] = 0 if ti.static(self.bc_psi_x_right == 0) else self.nx - 1
        if i[1] < 0:
            iout[1] = self.ny - 1 if ti.static(self.bc_psi_y_left == 0) else 0
        if i[1] > self.ny - 1:
            iout[1] = 0 if ti.static(self.bc_psi_y_right == 0) else self.ny - 1
        if i[2] < 0:
            iout[2] = self.nz - 1 if ti.static(self.bc_psi_z_left == 0) else 0
        if i[2] > self.nz - 1:
            iout[2] = 0 if ti.static(self.bc_psi_z_right == 0) else self.nz - 1
        return iout

    @ti.func
    def meq_vec(self, rho_local, u):
        out = ti.Vector([0.0] * 19)
        out[0] = rho_local
        out[3] = u[0]; out[5] = u[1]; out[7] = u[2]
        out[1] = u.dot(u)
        out[9] = 2*u.x*u.x - u.y*u.y - u.z*u.z
        out[11] = u.y*u.y - u.z*u.z
        out[13] = u.x*u.y; out[14] = u.y*u.z; out[15] = u.x*u.z
        return out

    @ti.func
    def multiply_M(self, i, j, k):
        out = ti.Vector([0.0] * 19)
        for row in range(19):
            s = 0.0
            for col in range(19):
                s += self.M_f[row, col] * self.F[i, j, k, col]
            out[row] = s
        return out

    @ti.func
    def compute_C(self, i, j, k):
        C = ti.Vector([0.0, 0.0, 0.0])
        ind_S = 0
        for s in ti.static(range(19)):
            ip = self.periodic_index_psi(ti.Vector([i, j, k]) + self.e[s])
            if self.solid[ip] == 0:
                C += 3.0 * self.w[s] * self.e_f[s] * self.psi[ip]
            else:
                ind_S = 1
                C += 3.0 * self.w[s] * self.e_f[s] * self.psi_solid
        if (ti.abs(self.rho_r[i, j, k] - self.rho_b[i, j, k]) > 0.9) and (ind_S == 1):
            C = ti.Vector([0.0, 0.0, 0.0])
        return C

    @ti.func
    def compute_S_local(self, i, j, k):
        sv = 0.0
        p = self.psi[i, j, k]
        if p > 0:
            if p > 0.1:
                sv = self.wl
            else:
                sv = self.lg0 + self.l1 * p + self.l2 * p * p
        else:
            if p < -0.1:
                sv = self.wg
            else:
                sv = self.lg0 + self.g1 * p + self.g2 * p * p
        sother = 8.0 * (2.0 - sv) / (8.0 - sv)

        S = ti.Vector([0.0] * 19)
        S[1] = sv; S[2] = sv; S[4] = sother; S[6] = sother; S[8] = sother
        S[9] = sv; S[10] = sv; S[11] = sv; S[12] = sv; S[13] = sv
        S[14] = sv; S[15] = sv; S[16] = sother; S[17] = sother; S[18] = sother
        return S

    @ti.kernel
    def collision(self):
        for i, j, k in self.rho:
            if i < self.nx and j < self.ny and k < self.nz and self.solid[i, j, k] == 0:
                C = self.compute_C(i, j, k)
                cc = C.norm()
                normal = ti.Vector([0.0, 0.0, 0.0])
                if cc > 0:
                    normal = C / cc

                m_temp = self.multiply_M(i, j, k)
                meq = self.meq_vec(self.rho[i, j, k], self.v[i, j, k])

                # Add interfacial tension to equilibrium moments
                meq[1] += self.CapA * cc
                meq[9] += 0.5 * self.CapA * cc * (2*normal.x*normal.x - normal.y*normal.y - normal.z*normal.z)
                meq[11] += 0.5 * self.CapA * cc * (normal.y*normal.y - normal.z*normal.z)
                meq[13] += 0.5 * self.CapA * cc * (normal.x * normal.y)
                meq[14] += 0.5 * self.CapA * cc * (normal.y * normal.z)
                meq[15] += 0.5 * self.CapA * cc * (normal.x * normal.z)

                S_local = self.compute_S_local(i, j, k)

                # MRT collision with Guo forcing
                f_ext = ti.Vector([self.ext_f[None][0], self.ext_f[None][1], self.ext_f[None][2]])
                for s in range(19):
                    m_temp[s] -= S_local[s] * (m_temp[s] - meq[s])
                    f_guo = 0.0
                    for l in range(19):
                        f_guo += self.w[l] * (
                            (self.e_f[l] - self.v[i, j, k]).dot(f_ext) / 3.0 +
                            (self.e_f[l].dot(self.v[i, j, k]) * (self.e_f[l].dot(f_ext))) / 9.0
                        ) * self.M_f[s, l]
                    m_temp[s] += (1 - 0.5 * S_local[s]) * f_guo

                # Compute equilibrium for each phase (for recoloring)
                g_r = ti.Vector([0.0] * 19)
                g_b = ti.Vector([0.0] * 19)

                for s in range(19):
                    self.f[i, j, k, s] = 0.0
                    for l in range(19):
                        self.f[i, j, k, s] += self.inv_M_f[s, l] * m_temp[l]
                    g_r[s] = self.feq(s, self.rho_r[i, j, k], self.v[i, j, k])
                    g_b[s] = self.feq(s, self.rho_b[i, j, k], self.v[i, j, k])

                # Recoloring step
                if cc > 0:
                    for kk in ti.static([1, 3, 5, 7, 9, 11, 13, 15, 17]):
                        ef = self.e[kk].dot(C)
                        cospsi = ti.min(g_r[kk], g_r[kk+1])
                        cospsi = ti.min(cospsi, g_b[kk])
                        cospsi = ti.min(cospsi, g_b[kk+1])
                        cospsi *= ef / cc
                        g_r[kk] += cospsi
                        g_r[kk+1] -= cospsi
                        g_b[kk] -= cospsi
                        g_b[kk+1] += cospsi

                # Streaming of phase densities
                ci = ti.Vector([i, j, k])
                for s in range(19):
                    ip = self.periodic_index(ci + self.e[s])
                    if self.solid[ip] == 0:
                        self.rhor[ip] += g_r[s]
                        self.rhob[ip] += g_b[s]
                    else:
                        self.rhor[i, j, k] += g_r[s]
                        self.rhob[i, j, k] += g_b[s]

    @ti.kernel
    def streaming(self):
        for i, j, k in self.rho:
            if i < self.nx and j < self.ny and k < self.nz and self.solid[i, j, k] == 0:
                ci = ti.Vector([i, j, k])
                for s in ti.static(range(19)):
                    ip = self.periodic_index(ci + self.e[s])
                    if self.solid[ip] == 0:
                        self.F[ip, s] = self.f[ci, s]
                    else:
                        self.F[ci, self.LR[s]] = self.f[ci, s]

    @ti.kernel
    def boundary_condition(self):
        # X boundaries
        if ti.static(self.bc_x_left == 1):
            for j, k in ti.ndrange((0, self.ny), (0, self.nz)):
                if self.solid[0, j, k] == 0:
                    for s in ti.static(range(19)):
                        if self.solid[1, j, k] > 0:
                            self.F[0, j, k, s] = self.feq(s, self.rho_bcxl, self.v[1, j, k])
                        else:
                            self.F[0, j, k, s] = self.feq(s, self.rho_bcxl, self.v[0, j, k])
        if ti.static(self.bc_x_right == 1):
            for j, k in ti.ndrange((0, self.ny), (0, self.nz)):
                if self.solid[self.nx-1, j, k] == 0:
                    for s in ti.static(range(19)):
                        if self.solid[self.nx-2, j, k] > 0:
                            self.F[self.nx-1, j, k, s] = self.feq(s, self.rho_bcxr, self.v[self.nx-2, j, k])
                        else:
                            self.F[self.nx-1, j, k, s] = self.feq(s, self.rho_bcxr, self.v[self.nx-1, j, k])
        # Y boundaries
        if ti.static(self.bc_y_left == 1):
            for i, k in ti.ndrange((0, self.nx), (0, self.nz)):
                if self.solid[i, 0, k] == 0:
                    for s in ti.static(range(19)):
                        if self.solid[i, 1, k] > 0:
                            self.F[i, 0, k, s] = self.feq(s, self.rho_bcyl, self.v[i, 1, k])
                        else:
                            self.F[i, 0, k, s] = self.feq(s, self.rho_bcyl, self.v[i, 0, k])
        if ti.static(self.bc_y_right == 1):
            for i, k in ti.ndrange((0, self.nx), (0, self.nz)):
                if self.solid[i, self.ny-1, k] == 0:
                    for s in ti.static(range(19)):
                        if self.solid[i, self.ny-2, k] > 0:
                            self.F[i, self.ny-1, k, s] = self.feq(s, self.rho_bcyr, self.v[i, self.ny-2, k])
                        else:
                            self.F[i, self.ny-1, k, s] = self.feq(s, self.rho_bcyr, self.v[i, self.ny-1, k])
        # Z boundaries
        if ti.static(self.bc_z_left == 1):
            for i, j in ti.ndrange((0, self.nx), (0, self.ny)):
                if self.solid[i, j, 0] == 0:
                    for s in ti.static(range(19)):
                        if self.solid[i, j, 1] > 0:
                            self.F[i, j, 0, s] = self.feq(s, self.rho_bczl, self.v[i, j, 1])
                        else:
                            self.F[i, j, 0, s] = self.feq(s, self.rho_bczl, self.v[i, j, 0])
        if ti.static(self.bc_z_right == 1):
            for i, j in ti.ndrange((0, self.nx), (0, self.ny)):
                if self.solid[i, j, self.nz-1] == 0:
                    for s in ti.static(range(19)):
                        if self.solid[i, j, self.nz-2] > 0:
                            self.F[i, j, self.nz-1, s] = self.feq(s, self.rho_bczr, self.v[i, j, self.nz-2])
                        else:
                            self.F[i, j, self.nz-1, s] = self.feq(s, self.rho_bczr, self.v[i, j, self.nz-1])

    @ti.kernel
    def boundary_condition_psi(self):
        if ti.static(self.bc_psi_x_left == 1):
            for j, k in ti.ndrange((0, self.ny), (0, self.nz)):
                if self.solid[0, j, k] == 0:
                    self.psi[0, j, k] = self.psi_x_left
                    self.rho_r[0, j, k] = (self.psi_x_left + 1.0) / 2.0
                    self.rho_b[0, j, k] = 1.0 - self.rho_r[0, j, k]
        if ti.static(self.bc_psi_x_right == 1):
            for j, k in ti.ndrange((0, self.ny), (0, self.nz)):
                if self.solid[self.nx-1, j, k] == 0:
                    self.psi[self.nx-1, j, k] = self.psi_x_right
                    self.rho_r[self.nx-1, j, k] = (self.psi_x_right + 1.0) / 2.0
                    self.rho_b[self.nx-1, j, k] = 1.0 - self.rho_r[self.nx-1, j, k]
        if ti.static(self.bc_psi_y_left == 1):
            for i, k in ti.ndrange((0, self.nx), (0, self.nz)):
                if self.solid[i, 0, k] == 0:
                    self.psi[i, 0, k] = self.psi_y_left
                    self.rho_r[i, 0, k] = (self.psi_y_left + 1.0) / 2.0
                    self.rho_b[i, 0, k] = 1.0 - self.rho_r[i, 0, k]
        if ti.static(self.bc_psi_y_right == 1):
            for i, k in ti.ndrange((0, self.nx), (0, self.nz)):
                if self.solid[i, self.ny-1, k] == 0:
                    self.psi[i, self.ny-1, k] = self.psi_y_right
                    self.rho_r[i, self.ny-1, k] = (self.psi_y_right + 1.0) / 2.0
                    self.rho_b[i, self.ny-1, k] = 1.0 - self.rho_r[i, self.ny-1, k]
        if ti.static(self.bc_psi_z_left == 1):
            for i, j in ti.ndrange((0, self.nx), (0, self.ny)):
                if self.solid[i, j, 0] == 0:
                    self.psi[i, j, 0] = self.psi_z_left
                    self.rho_r[i, j, 0] = (self.psi_z_left + 1.0) / 2.0
                    self.rho_b[i, j, 0] = 1.0 - self.rho_r[i, j, 0]
        if ti.static(self.bc_psi_z_right == 1):
            for i, j in ti.ndrange((0, self.nx), (0, self.ny)):
                if self.solid[i, j, self.nz-1] == 0:
                    self.psi[i, j, self.nz-1] = self.psi_z_right
                    self.rho_r[i, j, self.nz-1] = (self.psi_z_right + 1.0) / 2.0
                    self.rho_b[i, j, self.nz-1] = 1.0 - self.rho_r[i, j, self.nz-1]

    @ti.kernel
    def update_macro(self):
        for i, j, k in self.rho:
            if i < self.nx and j < self.ny and k < self.nz and self.solid[i, j, k] == 0:
                self.rho[i, j, k] = 0.0
                self.v[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
                self.rho_r[i, j, k] = self.rhor[i, j, k]
                self.rho_b[i, j, k] = self.rhob[i, j, k]
                self.rhor[i, j, k] = 0.0
                self.rhob[i, j, k] = 0.0

                for s in ti.static(range(19)):
                    self.f[i, j, k, s] = self.F[i, j, k, s]
                    self.rho[i, j, k] += self.f[i, j, k, s]
                    self.v[i, j, k] += self.e_f[s] * self.f[i, j, k, s]

                self.v[i, j, k] /= self.rho[i, j, k]
                f_ext = ti.Vector([self.ext_f[None][0], self.ext_f[None][1], self.ext_f[None][2]])
                self.v[i, j, k] += (f_ext / 2.0) / self.rho[i, j, k]

                rr = self.rho_r[i, j, k]
                rb = self.rho_b[i, j, k]
                total = rr + rb
                if total > 1e-10:
                    self.psi[i, j, k] = (rr - rb) / total
                else:
                    self.psi[i, j, k] = 0.0

    @ti.kernel
    def cal_max_v(self):
        for I in ti.grouped(self.rho):
            ti.atomic_max(self.max_v[None], self.v[I].norm())

    def get_max_v(self):
        self.max_v[None] = -1e10
        self.cal_max_v()
        return self.max_v[None]

    def step(self):
        self.collision()
        self.streaming()
        self.boundary_condition()
        self.update_macro()
        self.boundary_condition_psi()


# =========================================================================
# CLI
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="LBM D3Q19 two-phase flow solver")
    parser.add_argument("--geometry", required=True)
    parser.add_argument("--phase-init", default="half",
                        choices=["half", "random", "file"],
                        help="Phase initialization: half=left/right split, random, file=from .npy")
    parser.add_argument("--phase-file", default="", help="Path to .npy phase field (-1 to 1)")
    parser.add_argument("--niu-liquid", type=float, default=0.1)
    parser.add_argument("--niu-gas", type=float, default=0.1)
    parser.add_argument("--surface-tension", type=float, default=0.005, help="CapA parameter")
    parser.add_argument("--psi-solid", type=float, default=0.7, help="Wetting parameter")
    parser.add_argument("--timesteps", type=int, default=10000)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--backend", default="gpu", choices=["gpu", "cpu", "vulkan"])
    parser.add_argument("--output-dir", default="./lbm_results")
    parser.add_argument("--invert-solid", action="store_true")
    parser.add_argument("--flow-direction", default="x", choices=["x", "y", "z"])
    parser.add_argument("--bc-type", default="force", choices=["pressure", "force"])
    parser.add_argument("--rho-inlet", type=float, default=1.005)
    parser.add_argument("--rho-outlet", type=float, default=0.995)
    parser.add_argument("--body-force", type=float, default=1e-5)
    parser.add_argument("--fx", type=float, default=0.0)
    parser.add_argument("--fy", type=float, default=0.0)
    parser.add_argument("--fz", type=float, default=0.0)
    args = parser.parse_args()

    # Load geometry
    print(f"[LBM-2P] Loading geometry: {args.geometry}")
    geo = np.load(args.geometry)
    if geo.ndim != 3:
        print(json.dumps({"error": f"Expected 3D array, got {geo.ndim}D"}))
        sys.exit(1)

    nz, ny, nx = geo.shape
    geo = (geo > 0).astype(np.int8)
    if args.invert_solid:
        geo = 1 - geo
    print(f"[LBM-2P] Geometry: {nx}x{ny}x{nz} voxels")

    n_pore = int(np.sum(geo == 0))
    porosity = n_pore / (nx * ny * nz)
    print(f"[LBM-2P] Porosity: {porosity:.4f}")

    if porosity < 0.001:
        print(json.dumps({"error": "Porosity too low (<0.1%)."}))
        sys.exit(1)

    # Memory estimate
    n_voxels = nx * ny * nz
    # f(19*4) + F(19*4) + rho + v(3) + psi + rho_r + rho_b + rhor + rhob + solid
    mem_bytes = n_voxels * (19*4*2 + 4 + 3*4 + 4*5 + 1)
    mem_gb = mem_bytes / (1024**3)
    print(f"[LBM-2P] Estimated memory: {mem_gb:.2f} GB")

    # Backend selection with GPU memory check
    backend = args.backend
    if backend == "gpu" and mem_gb > 1.5:
        try:
            import subprocess
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if r.returncode == 0:
                gpu_mem_gb = float(r.stdout.strip().split("\n")[0]) / 1024
                print(f"[LBM-2P] GPU free memory: {gpu_mem_gb:.2f} GB")
                if mem_gb > gpu_mem_gb * 0.85:
                    print("[LBM-2P] Falling back to CPU backend")
                    backend = "cpu"
        except Exception:
            if mem_gb > 4.0:
                backend = "cpu"

    arch_map = {"gpu": ti.gpu, "cpu": ti.cpu, "vulkan": ti.vulkan}
    try:
        ti.init(arch=arch_map[backend])
        print(f"[LBM-2P] Taichi initialized with {backend} backend")
    except Exception:
        ti.init(arch=ti.cpu)
        print("[LBM-2P] Fallback to CPU")

    # Transpose to XYZ
    geo_xyz = np.ascontiguousarray(np.transpose(geo, (2, 1, 0)))

    # Initialize phase field
    if args.phase_init == "file" and args.phase_file and os.path.exists(args.phase_file):
        phase = np.load(args.phase_file).astype(np.float32)
        phase_xyz = np.ascontiguousarray(np.transpose(phase, (2, 1, 0)))
    elif args.phase_init == "random":
        phase_xyz = np.random.choice([-1.0, 1.0], size=(nx, ny, nz)).astype(np.float32)
    else:
        # Half-split along flow direction
        phase_xyz = np.ones((nx, ny, nz), dtype=np.float32)
        fd = args.flow_direction
        if fd == "x":
            phase_xyz[:nx//2, :, :] = -1.0
        elif fd == "y":
            phase_xyz[:, :ny//2, :] = -1.0
        else:
            phase_xyz[:, :, :nz//2] = -1.0
    # Set solid voxels to 0
    phase_xyz[geo_xyz > 0] = 0.0
    print(f"[LBM-2P] Phase field initialized ({args.phase_init})")

    # Create solver
    solver = LBM3D_TwoPhase(nx, ny, nz)
    solver.solid.from_numpy(geo_xyz)
    solver.psi.from_numpy(phase_xyz)
    solver.niu_l = args.niu_liquid
    solver.niu_g = args.niu_gas
    solver.CapA = args.surface_tension
    solver.psi_solid = args.psi_solid

    # Set body forces
    if args.fx != 0 or args.fy != 0 or args.fz != 0:
        solver.fx = args.fx
        solver.fy = args.fy
        solver.fz = args.fz
    elif args.bc_type == "force":
        force_val = args.body_force
        fd = args.flow_direction
        if fd == "x": solver.fx = force_val
        elif fd == "y": solver.fy = force_val
        else: solver.fz = force_val

    # Pressure BCs
    if args.bc_type == "pressure":
        fd = args.flow_direction
        if fd == "x":
            solver.bc_x_left = 1; solver.rho_bcxl = args.rho_inlet
            solver.bc_x_right = 1; solver.rho_bcxr = args.rho_outlet
        elif fd == "y":
            solver.bc_y_left = 1; solver.rho_bcyl = args.rho_inlet
            solver.bc_y_right = 1; solver.rho_bcyr = args.rho_outlet
        else:
            solver.bc_z_left = 1; solver.rho_bczl = args.rho_inlet
            solver.bc_z_right = 1; solver.rho_bczr = args.rho_outlet

    print(f"[LBM-2P] niu_l={args.niu_liquid}, niu_g={args.niu_gas}, CapA={args.surface_tension}")

    solver.init_simulation()

    os.makedirs(args.output_dir, exist_ok=True)

    # Save geometry
    geo_zyx = np.ascontiguousarray(np.transpose(geo_xyz, (2, 1, 0)))
    np.save(os.path.join(args.output_dir, "geometry.npy"), geo_zyx.astype(np.int8))

    print(f"[LBM-2P] Running {args.timesteps} timesteps...")
    t0 = time.time()
    fd = args.flow_direction

    for step in range(args.timesteps):
        solver.step()

        if (step + 1) % args.save_interval == 0 or step == args.timesteps - 1:
            max_v = solver.get_max_v()
            elapsed = time.time() - t0
            mlups = (n_voxels * (step + 1)) / elapsed / 1e6

            v_np = solver.v.to_numpy()
            psi_np = solver.psi.to_numpy()
            pore_mask = (geo_xyz == 0)

            v_mag = np.sqrt(v_np[:,:,:,0]**2 + v_np[:,:,:,1]**2 + v_np[:,:,:,2]**2)
            v_mag_pore = np.where(pore_mask, v_mag, 0.0)

            # Save velocity magnitude (ZYX)
            v_mag_zyx = np.ascontiguousarray(np.transpose(v_mag_pore, (2, 1, 0)))
            out_v = os.path.join(args.output_dir, f"velocity_magnitude_{step+1:06d}.npy")
            np.save(out_v, v_mag_zyx.astype(np.float32))

            # Save phase field (ZYX)
            psi_zyx = np.ascontiguousarray(np.transpose(psi_np, (2, 1, 0)))
            out_p = os.path.join(args.output_dir, f"phase_{step+1:06d}.npy")
            np.save(out_p, psi_zyx.astype(np.float32))

            dir_idx = {"x": 0, "y": 1, "z": 2}[fd]
            mean_v = float(np.mean(v_np[:,:,:,dir_idx][pore_mask])) if pore_mask.any() else 0.0

            progress = {
                "step": step + 1,
                "total": args.timesteps,
                "max_velocity": float(max_v),
                "mean_velocity": float(mean_v),
                "mlups": float(mlups),
                "elapsed_s": float(elapsed),
                "porosity": float(porosity),
                "direction": fd,
                "output": out_v,
                "phase_output": out_p,
            }
            print(json.dumps(progress), flush=True)
            print(
                f"[LBM-2P] Step {step+1}/{args.timesteps} | "
                f"max_v={max_v:.6f} | mean_v={mean_v:.6e} | "
                f"{mlups:.1f} MLUPS | {elapsed:.1f}s",
                flush=True,
            )

    total_time = time.time() - t0
    print(f"\n[LBM-2P] Simulation complete in {total_time:.1f}s")
    print(f"[LBM-2P] Results saved to: {args.output_dir}")

    # VTK export
    try:
        from pyevtk.hl import gridToVTK
        x = np.linspace(0, nx, nx + 1).astype(np.float64)
        y = np.linspace(0, ny, ny + 1).astype(np.float64)
        z = np.linspace(0, nz, nz + 1).astype(np.float64)
        vtk_path = os.path.join(args.output_dir, "lbm_2phase_final")
        gridToVTK(
            vtk_path, x, y, z,
            cellData={
                "velocity_magnitude": np.ascontiguousarray(v_mag, dtype=np.float64),
                "phase": np.ascontiguousarray(psi_np, dtype=np.float64),
                "solid": np.ascontiguousarray(geo_xyz, dtype=np.float64),
            }
        )
        print(f"[LBM-2P] VTK exported: {vtk_path}.vtr")
    except ImportError:
        print("[LBM-2P] pyevtk not installed, skipping VTK export")
    except Exception as e:
        print(f"[LBM-2P] VTK export failed: {e}")


if __name__ == "__main__":
    main()
