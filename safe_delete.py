#!/usr/bin/env python3
"""Move files/directories to Windows Recycle Bin instead of rm -rf."""
import sys, os, subprocess

def trash(path):
    p = os.path.abspath(path)
    if not os.path.exists(p):
        print(f"Not found: {p}")
        return
    kind = 'DeleteDirectory' if os.path.isdir(p) else 'DeleteFile'
    escaped = p.replace('\\', '\\\\').replace("'", "''")
    cmd = (
        f"Add-Type -AssemblyName Microsoft.VisualBasic; "
        f"[Microsoft.VisualBasic.FileIO.FileSystem]::{kind}('{escaped}','OnlyErrorDialogs','SendToRecycleBin')"
    )
    subprocess.run(['powershell.exe', '-Command', cmd], check=True)
    print(f"Moved to Recycle Bin: {p}")

if __name__ == '__main__':
    for arg in sys.argv[1:]:
        trash(arg)
