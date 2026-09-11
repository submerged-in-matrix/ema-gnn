"""
Parse a Quantum Espresso vc-relax .out file into extxyz training frames.

Pairing logic (QE vc-relax convention):
  structure_1        = initial header block (celldm + crystal axes + tau alat-units)
  structure_i (i>=2)  = the CELL_PARAMETERS(angstrom)+ATOMIC_POSITIONS(crystal) block
                        printed after step (i-1)'s forces (BFGS's proposed next geometry;
                        for the last step this is the final scf at the converged geometry)
  energy_i, forces_i = the i-th "! total energy" / "Forces acting on atoms" block,
                        i.e. the SCF result computed AT structure_i.

Dedup rule: keep frame i unless BOTH max atomic displacement (Angstrom) AND
max |force component change| (eV/Angstrom) vs. the last KEPT frame are below `tol`.
"""
import re
import numpy as np

BOHR_TO_ANG = 0.529177210903
RY_TO_EV = 13.605693009
RY_PER_BOHR_TO_EV_PER_ANG = RY_TO_EV / BOHR_TO_ANG  # = 25.71104...


def parse_qe_vcrelax(path):
    with open(path) as f:
        text = f.read()

    # --- species + initial cell/positions (structure_1) ---
    site_block = re.search(
        r"site n\.\s+atom\s+positions \(alat units\)\n((?:\s*\d+\s+\S+\s+tau\(.*?\n)+)",
        text,
    ).group(1)
    site_rows = re.findall(r"\d+\s+(\S+)\s+tau\(\s*\d+\)\s*=\s*\(\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\)", site_block)
    species = [r[0] for r in site_rows]
    tau_alat = np.array([[float(r[1]), float(r[2]), float(r[3])] for r in site_rows])

    celldm1 = float(re.search(r"celldm\(1\)=\s*([\d.]+)", text).group(1))  # bohr
    axes_block = re.search(
        r"crystal axes: \(cart\. coord\. in units of alat\)\n"
        r"\s*a\(1\) = \(\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\)\s*\n"
        r"\s*a\(2\) = \(\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\)\s*\n"
        r"\s*a\(3\) = \(\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\)",
        text,
    )
    axes = np.array(axes_block.groups(), dtype=float).reshape(3, 3)  # rows a1,a2,a3, units of alat
    alat_ang = celldm1 * BOHR_TO_ANG
    cell_1 = axes * alat_ang  # Angstrom, Cartesian row vectors
    pos_1 = tau_alat * alat_ang  # tau (alat units) are Cartesian coords in units of alat -> Angstrom

    n_atoms = len(species)

    # --- subsequent structures (CELL_PARAMETERS + ATOMIC_POSITIONS blocks) ---
    struct_blocks = re.findall(
        r"CELL_PARAMETERS \(angstrom\)\n"
        r"((?:\s*[-\d.]+\s+[-\d.]+\s+[-\d.]+\n){3})"
        r"\n*ATOMIC_POSITIONS \(crystal\)\n"
        r"((?:\S+\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\n?){" + str(n_atoms) + r"})",
        text,
    )
    later_cells, later_frac_positions = [], []
    for cell_txt, pos_txt in struct_blocks:
        cell = np.array(re.findall(r"([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", cell_txt), dtype=float)
        frac = np.array(
            [row[1:] for row in re.findall(r"(\S+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", pos_txt)],
            dtype=float,
        )
        later_cells.append(cell)
        later_frac_positions.append(frac)

    # --- energies ---
    energies_ry = [float(x) for x in re.findall(r"!\s*total energy\s*=\s*([-\d.]+)\s*Ry", text)]

    # --- forces ---
    force_blocks = []
    for block in re.findall(r"Forces acting on atoms.*?\n\n(.*?)\n\n\s*Total force", text, re.S):
        rows = re.findall(r"force\s*=\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)", block)
        force_blocks.append(np.array(rows, dtype=float))

    n_steps = len(energies_ry)
    assert len(force_blocks) == n_steps, f"{len(force_blocks)} force blocks vs {n_steps} energies"

    note = None
    if len(later_cells) == n_steps:
        # Known QE pattern: BFGS converged, but the subsequent "final scf calculation
        # at the relaxed structure" (sanity check) failed to converge, so it produced
        # no valid "!"-marked energy/forces. Its structure block is dangling (no data
        # to pair with) -- drop it and proceed with the n_steps-1 real pairs.
        if "convergence NOT achieved" in text:
            later_cells = later_cells[:-1]
            later_frac_positions = later_frac_positions[:-1]
            note = "dropped trailing structure: final sanity SCF did not converge"
        else:
            raise AssertionError(
                f"{len(later_cells)} later structures vs {n_steps} energies "
                f"(equal counts but no 'convergence NOT achieved' found -- unexplained, not auto-fixing)"
            )
    assert len(later_cells) == n_steps - 1, f"{len(later_cells)} later structures vs {n_steps} energies"

    frames = []
    for i in range(n_steps):
        if i == 0:
            cell, pos_cart = cell_1, pos_1
        else:
            cell = later_cells[i - 1]
            pos_cart = later_frac_positions[i - 1] @ cell
        energy_ev = energies_ry[i] * RY_TO_EV
        forces_ev_ang = force_blocks[i] * RY_PER_BOHR_TO_EV_PER_ANG
        frames.append(dict(species=species, cell=cell, positions=pos_cart,
                            energy=energy_ev, forces=forces_ev_ang))
    return frames, note


def dedup_frames(frames, tol=1e-5):
    kept = [frames[0]]
    for f in frames[1:]:
        prev = kept[-1]
        dpos = np.abs(f["positions"] - prev["positions"]).max()
        dforce = np.abs(f["forces"] - prev["forces"]).max()
        if dpos < tol and dforce < tol:
            # Same geometry as the last kept frame (QE's "final sanity SCF" pattern:
            # BFGS's own energy estimate followed by a more electronically-converged
            # recompute at the identical, frozen geometry). Energy can differ
            # meaningfully even though position/forces don't -- keep the LATER,
            # more accurate energy rather than silently discarding it.
            kept[-1] = f
            continue
        kept.append(f)
    return kept


def write_extxyz(frames, out_path):
    with open(out_path, "w") as f:
        for fr in frames:
            n = len(fr["species"])
            f.write(f"{n}\n")
            lattice = " ".join(f"{x:.8f}" for x in fr["cell"].flatten())
            f.write(
                f'Lattice="{lattice}" Properties=species:S:1:pos:R:3:forces:R:3 '
                f'energy={fr["energy"]:.8f} pbc="T T T"\n'
            )
            for sp, p, fo in zip(fr["species"], fr["positions"], fr["forces"]):
                f.write(f"{sp} {p[0]:.8f} {p[1]:.8f} {p[2]:.8f} {fo[0]:.8f} {fo[1]:.8f} {fo[2]:.8f}\n")


if __name__ == "__main__":
    import sys
    import os

    if len(sys.argv) < 2:
        print("Usage: python qe_vcrelax_to_extxyz.py <input.out> [tol] [output.xyz]")
        sys.exit(1)

    in_path = sys.argv[1]
    tol = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-5
    out_path = sys.argv[3] if len(sys.argv) > 3 else os.path.splitext(in_path)[0] + ".xyz"

    frames, note = parse_qe_vcrelax(in_path)
    print(f"Parsed {len(frames)} raw frames")
    if note:
        print(f"Note: {note}")
    kept = dedup_frames(frames, tol=tol)
    print(f"Kept {len(kept)} frames after dedup (tol={tol})")
    write_extxyz(kept, out_path)
    print(f"Wrote {out_path}")