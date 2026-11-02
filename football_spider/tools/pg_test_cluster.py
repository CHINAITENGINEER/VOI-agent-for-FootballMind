"""Reuse installed PostgreSQL binaries; never modify the existing server.

Starts a loopback-only throwaway cluster on an OS-assigned unused port. Files
are retained for inspection, and pg_ctl stops it in finally. No new install.
"""
from contextlib import contextmanager
from pathlib import Path
import os
import shutil
import socket
import subprocess
import tempfile


@contextmanager
def local_cluster(root=None):
    binary = Path(os.environ.get('PG_BIN', 'D:/PostgreSQL/bin'))
    if not (binary / 'initdb.exe').exists() and not (binary / 'initdb').exists():
        raise FileNotFoundError('Existing PostgreSQL binaries not found; set PG_BIN')
    root = Path(root or Path(__file__).resolve().parents[1] / '.runtime')
    root.mkdir(parents=True, exist_ok=True)
    cluster = Path(tempfile.mkdtemp(prefix='pg-validation-', dir=root)).resolve()
    with socket.socket() as port_socket:
        port_socket.bind(('127.0.0.1', 0))
        port = port_socket.getsockname()[1]
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0

    def run(name, *args):
        # On Windows postgres inherits pipe handles from pg_ctl even with -l.
        # Use a real log file, not PIPE, so communicate() never waits on server EOF.
        output_path = cluster.parent / (cluster.name + '-control.log')
        with output_path.open('ab') as output:
            result = subprocess.run([str(binary / name), *map(str, args)], stdout=output, stderr=output,
                                    creationflags=flags, timeout=60, stdin=subprocess.DEVNULL)
        if result.returncode:
            raise RuntimeError(output_path.read_text(errors='replace')[-3000:])
        return result

    run('initdb', '-D', cluster, '-U', 'footballmind_test', '--auth-local=trust', '--auth-host=trust', '--encoding=UTF8', '--locale=C')
    started = False
    try:
        run('pg_ctl', '-D', cluster, '-l', cluster / 'server.log', '-o', f'-h 127.0.0.1 -p {port}', '-w', 'start')
        started = True
        yield f'postgresql://footballmind_test@127.0.0.1:{port}/postgres', cluster
    finally:
        if started:
            run('pg_ctl', '-D', cluster, '-m', 'fast', '-w', 'stop')
