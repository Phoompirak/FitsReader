import py_compile
import sys
import os

files = [
    r'c:\Users\User\Desktop\FitsReader\spectral_models.py',
    r'c:\Users\User\Desktop\FitsReader\spectral_fitting.py',
    r'c:\Users\User\Desktop\FitsReader\app.py'
]

print("Verifying syntax...")
for f in files:
    try:
        py_compile.compile(f, doraise=True)
        print(f"✅ {os.path.basename(f)}: OK")
    except py_compile.PyCompileError as e:
        print(f"❌ {os.path.basename(f)}: ERROR")
        print(e)
