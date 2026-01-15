
import numpy as np
import math
import time
import argparse
import os
from functools import partial


A_MIN, A_MAX = -0.5, 0.5        # linear acceleration a
ALPHA_MIN, ALPHA_MAX = -6.0, 6.0  # angular acceleration α

DX_MAX = 0.02
DY_MAX = 0.02
DA_MAX = 0.2
D_ALPHA_MAX = 0.02


V_HAT_CONST = 0.1
OMEGA_HAT_MIN, OMEGA_HAT_MAX = -1.5, 1.5


DEFAULT_GRID = {
    "nx": 13,  # xr
    "ny": 13,  # yr
    "ntheta": 13,  # theta_r
    "nv": 9,   # v
    "nomega": 9  # omega
}

RANGE = {
    "xr": (-0.6, 0.6),
    "yr": (-0.6, 0.6),
    "theta": (-math.pi, math.pi),
    "v": (-1.5, 1.5),
    "omega": (-2.0, 2.0)
}

DT = 0.02        # pseudo-time step for HJ iteration 
MAX_ITERS = 300  # iteration cap for demo
TOL = 1e-4       # convergence tolerance 


def wrap_angle(a):
    return (a + math.pi) % (2*math.pi) - math.pi


class Grid5D:
    def __init__(self, nx, ny, ntheta, nv, nomega, ranges):
        self.nx, self.ny, self.ntheta, self.nv, self.nomega = nx, ny, ntheta, nv, nomega
        self.xs = np.linspace(ranges["xr"][0], ranges["xr"][1], nx)
        self.ys = np.linspace(ranges["yr"][0], ranges["yr"][1], ny)
        self.thetas = np.linspace(ranges["theta"][0], ranges["theta"][1], ntheta)
        self.vs = np.linspace(ranges["v"][0], ranges["v"][1], nv)
        self.omegas = np.linspace(ranges["omega"][0], ranges["omega"][1], nomega)

        self.dx = self.xs[1] - self.xs[0] if nx>1 else 1.0
        self.dy = self.ys[1] - self.ys[0] if ny>1 else 1.0
        self.dtheta = self.thetas[1] - self.thetas[0] if ntheta>1 else 1.0
        self.dv = self.vs[1] - self.vs[0] if nv>1 else 1.0
        self.domega = self.omegas[1] - self.omegas[0] if nomega>1 else 1.0

        self.shape = (nx, ny, ntheta, nv, nomega)

    def idx_to_state(self, ix, iy, it, iv, io):
        return (self.xs[ix], self.ys[iy], self.thetas[it], self.vs[iv], self.omegas[io])


def evaluate_hamiltonian_at(state, grads, mdl_params):
   
    xr, yr, th, v_s, omega_s = state
    gx, gy, gth, gv, go = grads
    
    a_opt = A_MIN if gv > 0 else A_MAX
    alpha_opt = ALPHA_MIN if go > 0 else ALPHA_MAX


    dx_opt = DX_MAX if gx > 0 else -DX_MAX
    dy_opt = DY_MAX if gy > 0 else -DY_MAX
    da_opt = DA_MAX if gv > 0 else -DA_MAX
    dalpha_opt = D_ALPHA_MAX if go > 0 else -D_ALPHA_MAX

    
    coeff_omega_hat = gx * yr - gy * xr - gth
    omega_hat_opt = OMEGA_HAT_MAX if coeff_omega_hat > 0 else OMEGA_HAT_MIN


    v_hat = V_HAT_CONST

    xdot = -v_hat + v_s * math.cos(th) + omega_hat_opt * yr + dx_opt
    ydot = v_s * math.sin(th) - omega_hat_opt * xr + dy_opt
    thdot = omega_s - omega_hat_opt
    vdot = a_opt + da_opt
    odot = alpha_opt + dalpha_opt 

    odot = alpha_opt + dalpha_opt

    # H = gx*xdot + gy*ydot + gth*thdot + gv*vdot + go*odot
    H = gx*xdot + gy*ydot + gth*thdot + gv*vdot + go*odot
    return H


def hamiltonian_at_state(state, grads):
    xr, yr, th, v_s, omega_s = state
    gx, gy, gth, gv, go = grads

    a_opt = A_MIN if gv > 0 else A_MAX
    alpha_opt = ALPHA_MIN if go > 0 else ALPHA_MAX

    dx_opt = DX_MAX if gx > 0 else -DX_MAX
    dy_opt = DY_MAX if gy > 0 else -DY_MAX
    da_opt = DA_MAX if gv > 0 else -DA_MAX
    dalpha_opt = D_ALPHA_MAX if go > 0 else -D_ALPHA_MAX

    coeff_omega_hat = gx * yr - gy * xr - gth
    omega_hat_opt = OMEGA_HAT_MAX if coeff_omega_hat > 0 else OMEGA_HAT_MIN

    v_hat = V_HAT_CONST

    xdot = -v_hat + v_s * math.cos(th) + omega_hat_opt * yr + dx_opt
    ydot = v_s * math.sin(th) - omega_hat_opt * xr + dy_opt
    thdot = omega_s - omega_hat_opt
    vdot = a_opt + da_opt
    odot = alpha_opt + dalpha_opt

    H = gx*xdot + gy*ydot + gth*thdot + gv*vdot + go*odot
    return H


class HJSolver:
    def __init__(self, grid: Grid5D, dt=DT):
        self.grid = grid
        self.dt = dt
        self.shape = grid.shape
        
        nx, ny, ntheta, nv, nomega = self.shape
        self.V = np.zeros(self.shape, dtype=np.float64)
        for ix in range(nx):
            for iy in range(ny):
                for it in range(ntheta):
                    for iv in range(nv):
                        for io in range(nomega):
                            xr, yr, th, v_s, omega_s = grid.idx_to_state(ix, iy, it, iv, io)
                           
                            self.V[ix,iy,it,iv,io] = xr*xr + yr*yr

    def compute_derivatives(self, V):
        g = self.grid
        nx, ny, nt, nv, no = self.shape
        gx = np.zeros_like(V); gy = np.zeros_like(V)
        gth = np.zeros_like(V); gv = np.zeros_like(V); go = np.zeros_like(V)

        for ix in range(nx):
            ixm = ix-1 if ix>0 else ix
            ixp = ix+1 if ix<nx-1 else ix
            for iy in range(ny):
                iym = iy-1 if iy>0 else iy
                iyp = iy+1 if iy<ny-1 else iy
                for it in range(nt):
                    itm = it-1 if it>0 else nt-1
                    itp = it+1 if it<nt-1 else 0
                    for iv in range(nv):
                        ivm = iv-1 if iv>0 else iv
                        ivp = iv+1 if iv<nv-1 else iv
                        for io in range(no):
                            iom = io-1 if io>0 else io
                            iop = io+1 if io<no-1 else io
                            gx[ix,iy,it,iv,io] = (V[ixp,iy,it,iv,io] - V[ixm,iy,it,iv,io]) / (2.0*g.dx)
                            gy[ix,iy,it,iv,io] = (V[ix, iyp, it, iv, io] - V[ix, iym, it, iv, io]) / (2.0*g.dy)
                            
                            gth[ix,iy,it,iv,io] = (V[ix,iy,itp,iv,io] - V[ix,iy,itm,iv,io]) / (2.0*g.dtheta)
                            gv[ix,iy,it,iv,io] = (V[ix,iy,it,ivp,io] - V[ix,iy,it,ivm,io]) / (2.0*g.dv)
                            go[ix,iy,it,iv,io] = (V[ix,iy,it,iv,iop] - V[ix,iy,it,iv,iom]) / (2.0*g.domega)
        return gx, gy, gth, gv, go

    def step(self):
        V = self.V
        g = self.grid
        nx, ny, nt, nv, no = self.shape
        gx, gy, gth, gv, go = self.compute_derivatives(V)

        alpha_x = np.max(np.abs(g.vs)) + abs(V_HAT_CONST) + 1.0
        alpha_y = np.max(np.abs(g.vs)) + 1.0
        alpha_th = np.max(np.abs(g.omegas)) + abs(OMEGA_HAT_MAX) + 1.0
        alpha_v = max(abs(A_MIN), abs(A_MAX))
        alpha_o = max(abs(ALPHA_MIN), abs(ALPHA_MAX))

        newV = np.empty_like(V)
        for ix in range(nx):
            for iy in range(ny):
                for it in range(nt):
                    for iv in range(nv):
                        for io in range(no):
                            state = g.idx_to_state(ix, iy, it, iv, io)
                            grads = (gx[ix,iy,it,iv,io], gy[ix,iy,it,iv,io],
                                     gth[ix,iy,it,iv,io], gv[ix,iy,it,iv,io], go[ix,iy,it,iv,io])
                            H = hamiltonian_at_state(state, grads)
                           
                            ixm = ix-1 if ix>0 else ix
                            ixp = ix+1 if ix<nx-1 else ix
                            iym = iy-1 if iy>0 else iy
                            iyp = iy+1 if iy<ny-1 else iy
                            itm = it-1 if it>0 else nt-1
                            itp = it+1 if it<nt-1 else 0
                            ivm = iv-1 if iv>0 else iv
                            ivp = iv+1 if iv<nv-1 else iv
                            iom = io-1 if io>0 else io
                            iop = io+1 if io<no-1 else io

                            diss = 0.0
                            diss += 0.5 * alpha_x * (V[ixp, iy, it, iv, io] - V[ixm, iy, it, iv, io])
                            diss += 0.5 * alpha_y * (V[ix, iyp, it, iv, io] - V[ix, iym, it, iv, io])
                            diss += 0.5 * alpha_th * (V[ix, iy, itp, iv, io] - V[ix, iy, itm, iv, io])
                            diss += 0.5 * alpha_v * (V[ix, iy, it, ivp, io] - V[ix, iy, it, ivm, io])
                            diss += 0.5 * alpha_o * (V[ix, iy, it, iv, iop] - V[ix, iy, it, iv, iom])

                            Vold = V[ix,iy,it,iv,io]

                            H = np.clip(H, -1e3, 1e3)
                            Vnew = Vold + self.dt * (H - diss)
               
                            xr, yr, th0, v_s0, omega_s0 = state
                            l_r = xr*xr + yr*yr  # paper running example l(r)
                            if Vnew > l_r:
                                
                                pass
                            newV[ix,iy,it,iv,io] = Vnew
        self.V = newV

    def iterate(self, max_iters=MAX_ITERS, tol=TOL, verbose=True):
        t0 = time.time()
        for k in range(max_iters):
            oldV = self.V.copy()
            self.step()
            diff = np.max(np.abs(self.V - oldV))
            if verbose:
                print(f"[HJ] iter {k+1:4d}: max|dV| = {diff:.6e}")
            if diff < tol:
                print("[HJ] converged at iter", k+1, "elapsed", time.time()-t0)
                break
        else:
            print("[HJ] max iters reached; final diff=", diff)
        gx, gy, gth, gv, go = self.compute_derivatives(self.V)
        return self.V, (gx, gy, gth, gv, go)


def save_value_table(filename, grid: Grid5D, V, grads):
    np.savez_compressed(filename,
                        xs=grid.xs, ys=grid.ys, thetas=grid.thetas,
                        vs=grid.vs, omegas=grid.omegas,
                        V=V, gx=grads[0], gy=grads[1], gth=grads[2], gv=grads[3], go=grads[4])
    print("Saved:", filename)


def main(args):
    cfg = DEFAULT_GRID.copy()
    if args.scale:
        scale = float(args.scale)
        cfg["nx"] = int(cfg["nx"] * scale)
        cfg["ny"] = int(cfg["ny"] * scale)
        cfg["ntheta"] = int(cfg["ntheta"] * scale)
        cfg["nv"] = int(cfg["nv"] * scale)
        cfg["nomega"] = int(cfg["nomega"] * scale)

    grid = Grid5D(cfg["nx"], cfg["ny"], cfg["ntheta"], cfg["nv"], cfg["nomega"], RANGE)
    print("Grid shape", grid.shape)
    solver = HJSolver(grid, dt=args.dt)
    print("Starting iterations...")
    V, grads = solver.iterate(max_iters=args.max_iters, tol=args.tol, verbose=True)

    outname = args.out if args.out else "value_function_5d3d_demo_2k.npz"
    save_value_table(outname, grid, V, grads)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt", type=float, default=DT, help="pseudo-time step")
    parser.add_argument("--max_iters", type=int, default=MAX_ITERS)
    parser.add_argument("--tol", type=float, default=TOL)
    parser.add_argument("--out", type=str, default="value_function_5d3d_demo_2k.npz")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="scale grid sizes by this factor (float). caution: large -> very slow/huge memory")
    args = parser.parse_args()
    main(args)
