import os
import shutil
import time

files_to_remove = [
    'app.py', 'spectral_fitting.py', 'spectral_models.py', 
    'correlation_analysis.py', 'timing_analysis.py', 
    'multi_epoch.py', 'relxill_approx.py',
    'FIX_GPU_ERROR.md',
    'read_fits.py', 'verify_syntax.py', 'main.py',
    'brute_force_results.json',
    'setup_structure.py', 'copy_files.py', 'copy_log.txt', 
    'cleanup.py', 'force_cleanup.py'
]

dirs_to_remove = [
    'attached_assets', 'sounds', 'checkpoints'
]

print("Starting cleanup...")

for f in files_to_remove:
    if os.path.exists(f):
        try:
            os.remove(f)
            print(f"Deleted file: {f}")
        except Exception as e:
            print(f"Failed to delete {f}: {e}")
    else:
        print(f"File not found (already clean): {f}")

for d in dirs_to_remove:
    if os.path.exists(d):
        try:
            shutil.rmtree(d)
            print(f"Deleted directory: {d}")
        except Exception as e:
            print(f"Failed to delete {d}: {e}")
    else:
        print(f"Directory not found (already clean): {d}")

print("\nCleanup complete!")
print("You can now delete this script (final_cleanup.py).")
