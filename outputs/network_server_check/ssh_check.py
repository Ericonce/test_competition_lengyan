import argparse
import hashlib
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(os.environ['TEMP']) / 'codex-ssh-test-deps'))
import paramiko

p = argparse.ArgumentParser()
p.add_argument('--command', required=True)
p.add_argument('--stdin-file')
p.add_argument('--model-dir', default='/root')
args = p.parse_args()
password = os.environ['NETWORK_TEST_SSH_PASSWORD']

def client():
    c = paramiko.SSHClient()
    c.load_system_host_keys()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    return c

jump = client()
target = client()
try:
    jump.connect('112.6.203.30', port=6001, username='ubuntu', password=password,
                 allow_agent=False, look_for_keys=False, timeout=20, auth_timeout=20)
    channel = jump.get_transport().open_channel('direct-tcpip', ('127.0.0.1', 2201), ('127.0.0.1', 0))
    target.connect('127.0.0.1', port=2201, username='root', password=password, sock=channel,
                   allow_agent=False, look_for_keys=False, timeout=20, auth_timeout=20)
    print('SSH_CONNECTED', flush=True)
    sftp = target.open_sftp()
    for name in ('resnet.py', 'cnn.py', 'efficientnet.py'):
        path = args.model_dir.rstrip('/') + '/' + name
        try:
            with sftp.open(path, 'rb') as f:
                content = f.read()
            local = Path('C:/Users/admin/Desktop') / name
            print('FILE', path, 'bytes', len(content), 'matches_desktop',
                  hashlib.sha256(content).digest() == hashlib.sha256(local.read_bytes()).digest(), flush=True)
        except OSError as e:
            print('FILE_ERROR', path, str(e), flush=True)
    sftp.close()
    stdin, stdout, stderr = target.exec_command(args.command, timeout=30)
    if args.stdin_file:
        stdin.write(Path(args.stdin_file).read_text(encoding='utf-8'))
        stdin.flush()
    stdin.channel.shutdown_write()
    ch = stdout.channel
    deadline = time.monotonic() + 300
    while True:
        if ch.recv_ready():
            sys.stdout.write(ch.recv(65536).decode('utf-8', errors='replace'))
            sys.stdout.flush()
        if ch.recv_stderr_ready():
            sys.stdout.write(ch.recv_stderr(65536).decode('utf-8', errors='replace'))
            sys.stdout.flush()
        if ch.exit_status_ready() and not ch.recv_ready() and not ch.recv_stderr_ready():
            break
        if time.monotonic() > deadline:
            raise TimeoutError('Remote command exceeded 300 seconds')
        time.sleep(0.05)
    status = ch.recv_exit_status()
    print('REMOTE_EXIT', status, flush=True)
    sys.exit(status)
finally:
    target.close()
    jump.close()
