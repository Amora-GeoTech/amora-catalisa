#!/usr/bin/env python
"""
LBM D3Q19 MRT Single-Phase Flow Solver
=======================================
Standalone script for porous media flow simulation.
Based on taichi_LBM3D by yjhp1016.

Reads a binarized 3D numpy array (0=pore, 1=solid) and simulates
single-phase fluid flow using the Lattice Boltzmann Method.

Outputs velocity field as numpy arrays and optional VTK files.
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
# D3Q19 MRT LBM Solver
# =========================================================================

@ti.data_oriented
class LBM3D_SinglePhase:

    def __init__(self, nx, ny, nz, grayscale=False):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.fx, self.fy, self.fz = 0.0, 0.0, 0.0
        self.niu = 0.16667
        self.grayscale = grayscale

        self.max_v = ti.field(ti.f32, shape=())

        # Boundary conditions: 0=periodic, 1=pressure, 2=velocity
        self.bc_x_left, self.rho_bcxl, self.vx_bcxl, self.vy_bcxl, self.vz_bcxl = 0, 1.0, 0.0, 0.0, 0.0
        self.bc_x_right, self.rho_bcxr, self.vx_bcxr, self.vy_bcxr, self.vz_bcxr = 0, 1.0, 0.0, 0.0, 0.0
        self.bc_y_left, self.rho_bcyl, self.vx_bcyl, self.vy_bcyl, self.vz_bcyl = 0, 1.0, 0.0, 0.0, 0.0
        self.bc_y_right, self.rho_bcyr, self.vx_bcyr, self.vy_bcyr, self.vz_bcyr = 0, 1.0, 0.0, 0.0, 0.0
        self.bc_z_left, self.rho_bczl, self.vx_bczl, self.vy_bczl, self.vz_bczl = 0, 1.0, 0.0, 0.0, 0.0
        self.bc_z_right, self.rho_bczr, self.vx_bczr, self.vy_bczr, self.vz_bczr = 0, 1.0, 0.0, 0.0, 0.0

        # Fields
        self.f = ti.Vector.field(19, ti.f32, shape=(nx, ny, nz))
        self.F = ti.Vector.field(19, ti.f32, shape=(nx, ny, nz))
        self.rho = ti.field(ti.f32, shape=(nx, ny, nz))
        self.v = ti.Vector.field(3, ti.f32, shape=(nx, ny, nz))

        self.e = ti.Vector.field(3, ti.i32, shape=(19))
        self.S_dig = ti.Vector.field(19, ti.f32, shape=())
        self.e_f = ti.Vector.field(3, ti.f32, shape=(19))
        self.w = ti.field(ti.f32, shape=(19))
        self.solid = ti.field(ti.i8, shape=(nx, ny, nz))
        # Grayscale: solid fraction ns ∈ [0,1], 0=fluid, 1=solid
        self.ns = ti.field(ti.f32, shape=(nx, ny, nz))
        self.ext_f = ti.Vector.field(3, ti.f32, shape=())

        # MRT transformation matrix stored as ti.field (avoids 19x19 Matrix compile warning)
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

        self.LR = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17]

        self.M_f.from_numpy(M_np)
        self.inv_M_f.from_numpy(inv_M_np)

    def init_simulation(self):
        self.bc_vel_x_left = [self.vx_bcxl, self.vy_bcxl, self.vz_bcxl]
        self.bc_vel_x_right = [self.vx_bcxr, self.vy_bcxr, self.vz_bcxr]
        self.bc_vel_y_left = [self.vx_bcyl, self.vy_bcyl, self.vz_bcyl]
        self.bc_vel_y_right = [self.vx_bcyr, self.vy_bcyr, self.vz_bcyr]
        self.bc_vel_z_left = [self.vx_bczl, self.vy_bczl, self.vz_bczl]
        self.bc_vel_z_right = [self.vx_bczr, self.vy_bczr, self.vz_bczr]

        self.tau_f = self.niu / 3.0 + 0.5
        self.s_v = 1.0 / self.tau_f
        self.s_other = 8.0 * (2.0 - self.s_v) / (8.0 - self.s_v)

        self.S_dig[None] = ti.Vector([
            0, self.s_v, self.s_v, 0, self.s_other, 0, self.s_other, 0, self.s_other,
            self.s_v, self.s_v, self.s_v, self.s_v, self.s_v, self.s_v, self.s_v,
            self.s_other, self.s_other, self.s_other
        ])

        self.ext_f[None][0] = self.fx
        self.ext_f[None][1] = self.fy
        self.ext_f[None][2] = self.fz
        self.force_flag = 1 if (abs(self.fx) > 0 or abs(self.fy) > 0 or abs(self.fz) > 0) else 0

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
                for s in ti.static(range(19)):
                    self.f[i, j, k][s] = self.feq(s, 1.0, self.v[i, j, k])
                    self.F[i, j, k][s] = self.feq(s, 1.0, self.v[i, j, k])

    @ti.kernel
    def static_init(self):
        self.e[0] = ti.Vector([0, 0, 0])
        self.e[1] = ti.Vector([1, 0, 0]); self.e[2] = ti.Vector([-1, 0, 0])
        self.e[3] = ti.Vector([0, 1, 0]); self.e[4] = ti.Vector([0, -1, 0])
        self.e[5] = ti.Vector([0, 0, 1]); self.e[6] = ti.Vector([0, 0, -1])
        self.e[7] = ti.Vector([1, 1, 0]); self.e[8] = ti.Vector([-1, -1, 0])
        self.e[9] = ti.Vector([1, -1, 0]); self.e[10] = ti.Vector([-1, 1, 0])
        self.e[11] = ti.Vector([1, 0, 1]); self.e[12] = ti.Vector([-1, 0, -1])
        self.e[13] = ti.Vector([1, 0, -1]); self.e[14] = ti.Vector([-1, 0, 1])
        self.e[15] = ti.Vector([0, 1, 1]); self.e[16] = ti.Vector([0, -1, -1])
        self.e[17] = ti.Vector([0, 1, -1]); self.e[18] = ti.Vector([0, -1, 1])

        self.e_f[0] = ti.Vector([0.0, 0.0, 0.0])
        self.e_f[1] = ti.Vector([1.0, 0.0, 0.0]); self.e_f[2] = ti.Vector([-1.0, 0.0, 0.0])
        self.e_f[3] = ti.Vector([0.0, 1.0, 0.0]); self.e_f[4] = ti.Vector([0.0, -1.0, 0.0])
        self.e_f[5] = ti.Vector([0.0, 0.0, 1.0]); self.e_f[6] = ti.Vector([0.0, 0.0, -1.0])
        self.e_f[7] = ti.Vector([1.0, 1.0, 0.0]); self.e_f[8] = ti.Vector([-1.0, -1.0, 0.0])
        self.e_f[9] = ti.Vector([1.0, -1.0, 0.0]); self.e_f[10] = ti.Vector([-1.0, 1.0, 0.0])
        self.e_f[11] = ti.Vector([1.0, 0.0, 1.0]); self.e_f[12] = ti.Vector([-1.0, 0.0, -1.0])
        self.e_f[13] = ti.Vector([1.0, 0.0, -1.0]); self.e_f[14] = ti.Vector([-1.0, 0.0, 1.0])
        self.e_f[15] = ti.Vector([0.0, 1.0, 1.0]); self.e_f[16] = ti.Vector([0.0, -1.0, -1.0])
        self.e_f[17] = ti.Vector([0.0, 1.0, -1.0]); self.e_f[18] = ti.Vector([0.0, -1.0, 1.0])

        self.w[0] = 1.0/3.0
        self.w[1] = 1.0/18.0; self.w[2] = 1.0/18.0
        self.w[3] = 1.0/18.0; self.w[4] = 1.0/18.0
        self.w[5] = 1.0/18.0; self.w[6] = 1.0/18.0
        self.w[7] = 1.0/36.0; self.w[8] = 1.0/36.0
        self.w[9] = 1.0/36.0; self.w[10] = 1.0/36.0
        self.w[11] = 1.0/36.0; self.w[12] = 1.0/36.0
        self.w[13] = 1.0/36.0; self.w[14] = 1.0/36.0
        self.w[15] = 1.0/36.0; self.w[16] = 1.0/36.0
        self.w[17] = 1.0/36.0; self.w[18] = 1.0/36.0

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
    def multiply_M_vec(self, vec):
        """Multiply M matrix (stored as field) by a 19-vector."""
        result = ti.Vector([0.0] * 19)
        for row in range(19):
            s = 0.0
            for col in range(19):
                s += self.M_f[row, col] * vec[col]
            result[row] = s
        return result

    @ti.func
    def multiply_invM_vec(self, vec):
        """Multiply inv_M matrix (stored as field) by a 19-vector."""
        result = ti.Vector([0.0] * 19)
        for row in range(19):
            s = 0.0
            for col in range(19):
                s += self.inv_M_f[row, col] * vec[col]
            result[row] = s
        return result

    @ti.kernel
    def collision(self):
        for i, j, k in self.rho:
            if self.solid[i, j, k] == 0 and i < self.nx and j < self.ny and k < self.nz:
                m_temp = self.multiply_M_vec(self.F[i, j, k])
                meq = self.meq_vec(self.rho[i, j, k], self.v[i, j, k])
                m_temp -= self.S_dig[None] * (m_temp - meq)

                if ti.static(self.force_flag == 1):
                    f = ti.Vector([self.ext_f[None][0], self.ext_f[None][1], self.ext_f[None][2]])
                    for s in range(19):
                        f_guo = 0.0
                        for l in range(19):
                            f_guo += self.w[l] * (
                                (self.e_f[l] - self.v[i, j, k]).dot(f) / 3.0 +
                                (self.e_f[l].dot(self.v[i, j, k]) * (self.e_f[l].dot(f))) / 9.0
                            ) * self.M_f[s, l]
                        m_temp[s] += (1 - 0.5 * self.S_dig[None][s]) * f_guo

                self.f[i, j, k] = self.multiply_invM_vec(m_temp)

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

    @ti.kernel
    def streaming(self):
        for i in ti.grouped(self.rho):
            if i.x < self.nx and i.y < self.ny and i.z < self.nz:
                if ti.static(self.grayscale):
                    # Grayscale: partial bounce-back (Sukop method)
                    # ns=0 → fully fluid, ns=1 → fully solid
                    for s in ti.static(range(19)):
                        ip = self.periodic_index(i + self.e[s])
                        # Interpolation: f_new = f + ns*(f_bounce - f)
                        self.F[i][s] = self.f[i][s] + self.ns[i] * (
                            self.f[i][self.LR[s]] - self.f[i][s]
                        )
                    # Propagate fluid part to neighbors
                    for s in ti.static(range(19)):
                        ip = self.periodic_index(i + self.e[s])
                        if self.ns[ip] < 1.0:
                            self.F[ip][s] = self.f[i][s] + self.ns[i] * (
                                self.f[i][self.LR[s]] - self.f[i][s]
                            )
                else:
                    # Binary: standard bounce-back
                    if self.solid[i] == 0:
                        for s in ti.static(range(19)):
                            ip = self.periodic_index(i + self.e[s])
                            if self.solid[ip] == 0:
                                self.F[ip][s] = self.f[i][s]
                            else:
                                self.F[i][self.LR[s]] = self.f[i][s]

    @ti.kernel
    def boundary_condition(self):
        # X-left
        if ti.static(self.bc_x_left == 1):
            for j, k in ti.ndrange((0, self.ny), (0, self.nz)):
                if self.solid[0, j, k] == 0:
                    for s in ti.static(range(19)):
                        if self.solid[1, j, k] > 0:
                            self.F[0, j, k][s] = self.feq(s, self.rho_bcxl, self.v[1, j, k])
                        else:
                            self.F[0, j, k][s] = self.feq(s, self.rho_bcxl, self.v[0, j, k])
        if ti.static(self.bc_x_left == 2):
            for j, k in ti.ndrange((0, self.ny), (0, self.nz)):
                if self.solid[0, j, k] == 0:
                    for s in ti.static(range(19)):
                        self.F[0, j, k][s] = self.feq(s, 1.0, ti.Vector(self.bc_vel_x_left))
        # X-right
        if ti.static(self.bc_x_right == 1):
            for j, k in ti.ndrange((0, self.ny), (0, self.nz)):
                if self.solid[self.nx-1, j, k] == 0:
                    for s in ti.static(range(19)):
                        if self.solid[self.nx-2, j, k] > 0:
                            self.F[self.nx-1, j, k][s] = self.feq(s, self.rho_bcxr, self.v[self.nx-2, j, k])
                        else:
                            self.F[self.nx-1, j, k][s] = self.feq(s, self.rho_bcxr, self.v[self.nx-1, j, k])
        if ti.static(self.bc_x_right == 2):
            for j, k in ti.ndrange((0, self.ny), (0, self.nz)):
                if self.solid[self.nx-1, j, k] == 0:
                    for s in ti.static(range(19)):
                        self.F[self.nx-1, j, k][s] = self.feq(s, 1.0, ti.Vector(self.bc_vel_x_right))
        # Y-left
        if ti.static(self.bc_y_left == 1):
            for i, k in ti.ndrange((0, self.nx), (0, self.nz)):
                if self.solid[i, 0, k] == 0:
                    for s in ti.static(range(19)):
                        if self.solid[i, 1, k] > 0:
                            self.F[i, 0, k][s] = self.feq(s, self.rho_bcyl, self.v[i, 1, k])
                        else:
                            self.F[i, 0, k][s] = self.feq(s, self.rho_bcyl, self.v[i, 0, k])
        if ti.static(self.bc_y_left == 2):
            for i, k in ti.ndrange((0, self.nx), (0, self.nz)):
                if self.solid[i, 0, k] == 0:
                    for s in ti.static(range(19)):
                        self.F[i, 0, k][s] = self.feq(s, 1.0, ti.Vector(self.bc_vel_y_left))
        # Y-right
        if ti.static(self.bc_y_right == 1):
            for i, k in ti.ndrange((0, self.nx), (0, self.nz)):
                if self.solid[i, self.ny-1, k] == 0:
                    for s in ti.static(range(19)):
                        if self.solid[i, self.ny-2, k] > 0:
                            self.F[i, self.ny-1, k][s] = self.feq(s, self.rho_bcyr, self.v[i, self.ny-2, k])
                        else:
                            self.F[i, self.ny-1, k][s] = self.feq(s, self.rho_bcyr, self.v[i, self.ny-1, k])
        if ti.static(self.bc_y_right == 2):
            for i, k in ti.ndrange((0, self.nx), (0, self.nz)):
                if self.solid[i, self.ny-1, k] == 0:
                    for s in ti.static(range(19)):
                        self.F[i, self.ny-1, k][s] = self.feq(s, 1.0, ti.Vector(self.bc_vel_y_right))
        # Z-left
        if ti.static(self.bc_z_left == 1):
            for i, j in ti.ndrange((0, self.nx), (0, self.ny)):
                if self.solid[i, j, 0] == 0:
                    for s in ti.static(range(19)):
                        if self.solid[i, j, 1] > 0:
                            self.F[i, j, 0][s] = self.feq(s, self.rho_bczl, self.v[i, j, 1])
                        else:
                            self.F[i, j, 0][s] = self.feq(s, self.rho_bczl, self.v[i, j, 0])
        if ti.static(self.bc_z_left == 2):
            for i, j in ti.ndrange((0, self.nx), (0, self.ny)):
                if self.solid[i, j, 0] == 0:
                    for s in ti.static(range(19)):
                        self.F[i, j, 0][s] = self.feq(s, 1.0, ti.Vector(self.bc_vel_z_left))
        # Z-right
        if ti.static(self.bc_z_right == 1):
            for i, j in ti.ndrange((0, self.nx), (0, self.ny)):
                if self.solid[i, j, self.nz-1] == 0:
                    for s in ti.static(range(19)):
                        if self.solid[i, j, self.nz-2] > 0:
                            self.F[i, j, self.nz-1][s] = self.feq(s, self.rho_bczr, self.v[i, j, self.nz-2])
                        else:
                            self.F[i, j, self.nz-1][s] = self.feq(s, self.rho_bczr, self.v[i, j, self.nz-1])
        if ti.static(self.bc_z_right == 2):
            for i, j in ti.ndrange((0, self.nx), (0, self.ny)):
                if self.solid[i, j, self.nz-1] == 0:
                    for s in ti.static(range(19)):
                        self.F[i, j, self.nz-1][s] = self.feq(s, 1.0, ti.Vector(self.bc_vel_z_right))

    @ti.kernel
    def update_macro(self):
        for i in ti.grouped(self.rho):
            if self.solid[i] == 0 and i.x < self.nx and i.y < self.ny and i.z < self.nz:
                self.rho[i] = 0.0
                self.v[i] = ti.Vector([0.0, 0.0, 0.0])
                self.f[i] = self.F[i]
                self.rho[i] += self.f[i].sum()
                for s in ti.static(range(19)):
                    self.v[i] += self.e_f[s] * self.f[i][s]
                f = ti.Vector([self.ext_f[None][0], self.ext_f[None][1], self.ext_f[None][2]])
                self.v[i] /= self.rho[i]
                self.v[i] += (f / 2.0) / self.rho[i]
            else:
                self.rho[i] = 1.0
                self.v[i] = ti.Vector([0.0, 0.0, 0.0])

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


# =========================================================================
# CLI
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="LBM D3Q19 single-phase flow solver")
    parser.add_argument("--geometry", required=True, help="Path to .npy binary geometry (0=pore, 1=solid)")
    parser.add_argument("--viscosity", type=float, default=0.16667, help="Kinematic viscosity (lattice units)")
    parser.add_argument("--timesteps", type=int, default=5000, help="Number of LBM timesteps")
    parser.add_argument("--save-interval", type=int, default=1000, help="Save results every N steps")
    parser.add_argument("--backend", default="gpu", choices=["gpu", "cpu", "vulkan"], help="Taichi backend")
    parser.add_argument("--output-dir", default="./lbm_results", help="Output directory")
    parser.add_argument("--invert-solid", action="store_true", help="Invert solid convention (0=solid, 1=pore)")
    parser.add_argument("--grayscale", action="store_true", help="Grayscale mode: partial bounce-back (Sukop method)")
    parser.add_argument("--flow-direction", default="x", choices=["x", "y", "z"], help="Flow direction")
    parser.add_argument("--bc-type", default="pressure", choices=["pressure", "velocity", "force"])
    parser.add_argument("--rho-inlet", type=float, default=1.005)
    parser.add_argument("--rho-outlet", type=float, default=0.995)
    parser.add_argument("--velocity-inlet", type=float, default=0.01)
    parser.add_argument("--body-force", type=float, default=1e-5)
    args = parser.parse_args()

    # Load geometry first to estimate memory
    print(f"[LBM] Loading geometry: {args.geometry}")
    geo = np.load(args.geometry)
    if geo.ndim != 3:
        print(json.dumps({"error": f"Expected 3D array, got {geo.ndim}D"}))
        sys.exit(1)

    nz, ny, nx = geo.shape
    print(f"[LBM] Geometry: {nx}x{ny}x{nz} voxels")

    # Grayscale mode: normalize to [0,1] solid fraction
    use_grayscale = args.grayscale
    if use_grayscale:
        geo_float = geo.astype(np.float32)
        vmin, vmax = float(geo_float.min()), float(geo_float.max())
        if vmax > vmin:
            # Map: low intensity → pore (0), high intensity → solid (1)
            # For rock microCT: bright = grain/solid, dark = pore
            ns_array = (geo_float - vmin) / (vmax - vmin)
        else:
            ns_array = np.zeros_like(geo_float)
        if args.invert_solid:
            ns_array = 1.0 - ns_array
        # Binary solid mask: ns > 0.99 is fully solid
        geo = (ns_array > 0.99).astype(np.int8)
        print(f"[LBM] Grayscale mode: ns range [{ns_array.min():.3f}, {ns_array.max():.3f}]")
        print(f"[LBM] Effective porosity (ns<0.5): {float(np.mean(ns_array < 0.5)):.4f}")
    else:
        # Binarize: ensure 0 and 1 only
        geo = (geo > 0).astype(np.int8)
        if args.invert_solid:
            geo = 1 - geo
        ns_array = None

    # Estimate memory: f(19)+F(19) float32 + rho(1) + v(3) + solid(1) + ns(4) per voxel
    n_voxels = nx * ny * nz
    mem_bytes = n_voxels * (19*4 + 19*4 + 4 + 3*4 + 1 + 4)  # ~169 bytes/voxel
    mem_gb = mem_bytes / (1024**3)
    print(f"[LBM] Estimated memory: {mem_gb:.2f} GB for {n_voxels:,} voxels")

    # Check GPU memory if using GPU
    backend = args.backend
    if backend == "gpu" and mem_gb > 1.5:
        # Try to detect GPU memory
        gpu_mem_gb = None
        try:
            import subprocess
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if r.returncode == 0:
                gpu_mem_gb = float(r.stdout.strip().split("\n")[0]) / 1024
                print(f"[LBM] GPU free memory: {gpu_mem_gb:.2f} GB")
        except Exception:
            pass

        if gpu_mem_gb is not None and mem_gb > gpu_mem_gb * 0.85:
            print(f"[LBM] Volume too large for GPU ({mem_gb:.1f} GB > {gpu_mem_gb*0.85:.1f} GB available)")
            print("[LBM] Falling back to CPU backend")
            backend = "cpu"
        elif gpu_mem_gb is None and mem_gb > 4.0:
            print(f"[LBM] Volume very large ({mem_gb:.1f} GB), using CPU to be safe")
            backend = "cpu"

    # Initialize taichi
    arch_map = {"gpu": ti.gpu, "cpu": ti.cpu, "vulkan": ti.vulkan}
    try:
        ti.init(arch=arch_map[backend])
        print(f"[LBM] Taichi initialized with {backend} backend")
    except Exception:
        print(f"[LBM] {backend} not available, falling back to CPU")
        ti.init(arch=ti.cpu)

    # Compute porosity
    if use_grayscale:
        # For grayscale: porosity = mean of (1 - ns)
        porosity = float(np.mean(1.0 - ns_array))
    else:
        n_pore = int(np.sum(geo == 0))
        n_total = nx * ny * nz
        porosity = n_pore / n_total
    print(f"[LBM] Porosity: {porosity:.4f}")

    if porosity < 0.001:
        print(json.dumps({"error": "Porosity too low (<0.1%). Check solid convention."}))
        sys.exit(1)

    # Transpose to (nx, ny, nz) for LBM solver (ZYX → XYZ)
    geo_xyz = np.ascontiguousarray(np.transpose(geo, (2, 1, 0)))

    # Create solver
    solver = LBM3D_SinglePhase(nx, ny, nz, grayscale=use_grayscale)
    solver.solid.from_numpy(geo_xyz)
    if use_grayscale:
        ns_xyz = np.ascontiguousarray(np.transpose(ns_array, (2, 1, 0)).astype(np.float32))
        solver.ns.from_numpy(ns_xyz)
        print(f"[LBM] Grayscale partial bounce-back enabled")
    solver.niu = args.viscosity

    # Set boundary conditions based on flow direction and BC type
    fd = args.flow_direction
    if args.bc_type == "pressure":
        if fd == "x":
            solver.bc_x_left = 1; solver.rho_bcxl = args.rho_inlet
            solver.bc_x_right = 1; solver.rho_bcxr = args.rho_outlet
        elif fd == "y":
            solver.bc_y_left = 1; solver.rho_bcyl = args.rho_inlet
            solver.bc_y_right = 1; solver.rho_bcyr = args.rho_outlet
        else:
            solver.bc_z_left = 1; solver.rho_bczl = args.rho_inlet
            solver.bc_z_right = 1; solver.rho_bczr = args.rho_outlet
        print(f"[LBM] BC: Pressure-driven, rho_in={args.rho_inlet}, rho_out={args.rho_outlet}, dir={fd}")
    elif args.bc_type == "velocity":
        vel = args.velocity_inlet
        if fd == "x":
            solver.bc_x_left = 2
            solver.vx_bcxl = vel
            solver.bc_x_right = 1; solver.rho_bcxr = 1.0
        elif fd == "y":
            solver.bc_y_left = 2
            solver.vy_bcyl = vel
            solver.bc_y_right = 1; solver.rho_bcyr = 1.0
        else:
            solver.bc_z_left = 2
            solver.vz_bczl = vel
            solver.bc_z_right = 1; solver.rho_bczr = 1.0
        print(f"[LBM] BC: Velocity-driven, v_in={vel}, dir={fd}")
    else:  # force
        force_val = args.body_force
        if fd == "x":
            solver.fx = force_val
        elif fd == "y":
            solver.fy = force_val
        else:
            solver.fz = force_val
        print(f"[LBM] BC: Body force = {force_val}, dir={fd}")

    # Initialize
    solver.init_simulation()

    # Output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Run simulation
    print(f"[LBM] Running {args.timesteps} timesteps...")
    t0 = time.time()

    for step in range(args.timesteps):
        solver.step()

        if (step + 1) % args.save_interval == 0 or step == args.timesteps - 1:
            max_v = solver.get_max_v()
            elapsed = time.time() - t0
            mlups = (nx * ny * nz * (step + 1)) / elapsed / 1e6

            # Get velocity field
            v_np = solver.v.to_numpy()  # (nx, ny, nz, 3)
            rho_np = solver.rho.to_numpy()  # (nx, ny, nz)

            # Velocity magnitude
            v_mag = np.sqrt(v_np[:, :, :, 0]**2 + v_np[:, :, :, 1]**2 + v_np[:, :, :, 2]**2)

            # Transpose back to ZYX for Slicer
            v_mag_zyx = np.ascontiguousarray(np.transpose(v_mag, (2, 1, 0)))

            # Save
            out_file = os.path.join(args.output_dir, f"velocity_magnitude_{step+1:06d}.npy")
            np.save(out_file, v_mag_zyx.astype(np.float32))

            # Also save velocity components
            vx_zyx = np.ascontiguousarray(np.transpose(v_np[:, :, :, 0], (2, 1, 0)))
            vy_zyx = np.ascontiguousarray(np.transpose(v_np[:, :, :, 1], (2, 1, 0)))
            vz_zyx = np.ascontiguousarray(np.transpose(v_np[:, :, :, 2], (2, 1, 0)))
            np.save(os.path.join(args.output_dir, f"vx_{step+1:06d}.npy"), vx_zyx.astype(np.float32))
            np.save(os.path.join(args.output_dir, f"vy_{step+1:06d}.npy"), vy_zyx.astype(np.float32))
            np.save(os.path.join(args.output_dir, f"vz_{step+1:06d}.npy"), vz_zyx.astype(np.float32))

            # Compute mean velocity in flow direction (only in pore space)
            dir_idx = {"x": 0, "y": 1, "z": 2}[fd]
            pore_mask = (geo_xyz == 0)
            mean_v = float(np.mean(v_np[:, :, :, dir_idx][pore_mask]))

            # Compute permeability via Darcy's law: k = μ * <v> / (ΔP/L)
            # In lattice units: dP = d_rho * cs^2 = d_rho / 3
            if args.bc_type == "pressure":
                delta_rho = args.rho_inlet - args.rho_outlet
                L = {"x": nx, "y": ny, "z": nz}[fd]
                dp_dx = delta_rho / (3.0 * L)
                if abs(dp_dx) > 1e-15:
                    permeability = args.viscosity * mean_v / dp_dx
                else:
                    permeability = 0.0
            elif args.bc_type == "force":
                force_val = args.body_force
                if abs(force_val) > 1e-15:
                    permeability = args.viscosity * mean_v / force_val
                else:
                    permeability = 0.0
            else:
                permeability = 0.0  # Can't compute from velocity BC easily

            progress = {
                "step": step + 1,
                "total": args.timesteps,
                "max_velocity": float(max_v),
                "mean_velocity": float(mean_v),
                "mlups": float(mlups),
                "elapsed_s": float(elapsed),
                "permeability": float(permeability),
                "porosity": float(porosity),
                "direction": fd,
                "output": out_file,
            }
            print(json.dumps(progress), flush=True)

            # Also print human-readable
            print(
                f"[LBM] Step {step+1}/{args.timesteps} | "
                f"max_v={max_v:.6f} | mean_v={mean_v:.6e} | "
                f"k={permeability:.6e} | {mlups:.1f} MLUPS | "
                f"{elapsed:.1f}s",
                flush=True,
            )

    # Final summary
    total_time = time.time() - t0
    print(f"\n[LBM] Simulation complete in {total_time:.1f}s")
    print(f"[LBM] Results saved to: {args.output_dir}")

    # Try VTK export
    try:
        from pyevtk.hl import gridToVTK
        x = np.linspace(0, nx, nx + 1).astype(np.float64)
        y = np.linspace(0, ny, ny + 1).astype(np.float64)
        z = np.linspace(0, nz, nz + 1).astype(np.float64)
        v_np = solver.v.to_numpy()
        vtk_path = os.path.join(args.output_dir, "lbm_final")
        gridToVTK(
            vtk_path, x, y, z,
            cellData={
                "velocity_magnitude": np.ascontiguousarray(v_mag, dtype=np.float64),
                "vx": np.ascontiguousarray(v_np[:,:,:,0], dtype=np.float64),
                "vy": np.ascontiguousarray(v_np[:,:,:,1], dtype=np.float64),
                "vz": np.ascontiguousarray(v_np[:,:,:,2], dtype=np.float64),
                "solid": np.ascontiguousarray(geo_xyz, dtype=np.float64),
                "density": np.ascontiguousarray(rho_np, dtype=np.float64),
            }
        )
        print(f"[LBM] VTK exported: {vtk_path}.vtr")
    except ImportError:
        print("[LBM] pyevtk not installed, skipping VTK export")
    except Exception as e:
        print(f"[LBM] VTK export failed: {e}")


if __name__ == "__main__":
    main()
