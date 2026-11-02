"""Generate and compile an acados OCP solver once, then reuse it."""
import fcntl
import hashlib
import os
import pathlib
import shutil

from acados_template import AcadosOcpSolver


def cached_ocp_solver(ocp, code_export_dir, json_name, sources, extra_stamp=''):
    """Return an AcadosOcpSolver for `ocp`, built in `code_export_dir` only when needed.

    A stamp of the OCP source files, `extra_stamp` (e.g. the model XML) and the acados install
    forces a rebuild when any of them changes. Config-only OCP changes (horizon, constraints, ...)
    still need `code_export_dir` deleted. A file lock next to the directory serializes the build
    between processes started together (parallel runs of a fresh checkout).
    """
    code_export_dir = pathlib.Path(code_export_dir)
    code_export_dir.parent.mkdir(parents=True, exist_ok=True)
    ocp.code_export_directory = str(code_export_dir)
    sources_sha = hashlib.sha256(b''.join(pathlib.Path(f).read_bytes() for f in sources)).hexdigest()
    stamp = (extra_stamp
             + f"ocp_sources_sha256={sources_sha}\n"
             + f"acados={pathlib.Path(os.environ.get('ACADOS_SOURCE_DIR', '')).resolve()}\n")
    stamp_file = code_export_dir / 'build_stamp.txt'
    with open(code_export_dir.parent / f'.{code_export_dir.name}.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        needs_build = (not stamp_file.is_file()) or stamp_file.read_text() != stamp
        if needs_build and code_export_dir.is_dir():
            shutil.rmtree(code_export_dir)
        solver = AcadosOcpSolver(ocp, json_file=str(code_export_dir / json_name),
                                 build=needs_build, generate=needs_build)
        if needs_build:
            stamp_file.write_text(stamp)
    return solver
