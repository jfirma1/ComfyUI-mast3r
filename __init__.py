import os
import subprocess
import threading
import sys
import locale
import traceback


def handle_stream(stream, prefix):
    stream.reconfigure(encoding=locale.getpreferredencoding(), errors='replace')
    for msg in stream:
        if prefix == '[!]' and ('it/s]' in msg or 's/it]' in msg) and ('%|' in msg or 'it [' in msg):
            if msg.startswith('100%'):
                print('\r' + msg, end="", file=sys.stderr),
            else:
                print('\r' + msg[:-1], end="", file=sys.stderr),
        else:
            if prefix == '[!]':
                print(prefix, msg, end="", file=sys.stderr)
            else:
                print(prefix, msg, end="")


def run_script(cmd, cwd='.'):
    if len(cmd) > 0 and cmd[0].startswith("#"):
        print(f"[ComfyUI-Manager] Unexpected behavior: `{cmd}`")
        return 0

    process = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

    stdout_thread = threading.Thread(target=handle_stream, args=(process.stdout, ""))
    stderr_thread = threading.Thread(target=handle_stream, args=(process.stderr, "[!]"))

    stdout_thread.start()
    stderr_thread.start()

    stdout_thread.join()
    stderr_thread.join()

    return process.wait()


# Try to import nodes
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    print("## MASt3R: Nodes loaded successfully")
except Exception as e:
    print(f"## MASt3R: First import attempt failed: {e}")
    traceback.print_exc()
    
    my_path = os.path.dirname(__file__)
    requirements_path = os.path.join(my_path, "requirements.txt")

    print(f"## MASt3R: Installing dependencies from {requirements_path}")

    run_script([sys.executable, '-s', '-m', 'pip', 'install', '-r', requirements_path])

    try:
        from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
        print("## MASt3R: Nodes loaded successfully after installing dependencies")
    except Exception as e2:
        print(f"## [ERROR] MASt3R: Second import attempt failed: {e2}")
        traceback.print_exc()
        
        print(f"## [ERROR] MASt3R: Attempting to reinstall dependencies using --user flag")
        run_script([sys.executable, '-s', '-m', 'pip', 'install', '--user', '-r', requirements_path])

        try:
            from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
            print("## MASt3R: Nodes loaded successfully after --user install")
        except Exception as e3:
            print(f"## [ERROR] MASt3R: All import attempts failed")
            print(f"## [ERROR] MASt3R: Final error: {e3}")
            traceback.print_exc()
            print("## [ERROR] MASt3R: Please check:")
            print("##   1. That all dependencies are installed (torch, scipy, trimesh, roma, etc.)")
            print("##   2. That the mast3r and dust3r folders are present in the ComfyUI-mast3r directory")
            print("##   3. Check the console output above for specific import errors")

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
