import os
import glob

OLD = "C:\\Users\\BitaMassoudi\\KasmirWorldProjects\\toucanet-option-b"
NEW = "C:\\Users\\BitaMassoudi\\KasmirWorldProjects\\toucanet-option-b"

scripts = glob.glob("*.py")
print(f"Found {len(scripts)} scripts")

for script in sorted(scripts):
    if script == "update_paths.py":
        continue
    with open(script, "r", encoding="utf-8") as f:
        content = f.read()
    if OLD in content:
        with open(script, "w", encoding="utf-8") as f:
            f.write(content.replace(OLD, NEW))
        print(f"  UPDATED: {script}")

print("Done.")