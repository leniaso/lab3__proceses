"""
Lee un archivo con la salida (repetida varias veces) de matrix_mul.go,
promedia los tiempos por K y genera, segun el tamano de matriz detectado:
  - speedup_<tamano>.png : grafica de speedup vs K (con linea ideal)
  - tabla_<tamano>.csv   : tabla con promedios y desviacion estandar
Ejemplo: tamano 500 -> speedup_500.png / tabla_500.csv
         matrices no cuadradas -> speedup_800x600x400.png

Uso:
  python3 graficar_speedup.py resultados_500.txt
  python3 graficar_speedup.py resultados_500.txt --hw "Ryzen 5 5600, WSL2" --cores 6 --threads 12
"""
import argparse
import re
import statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("archivo", help="archivo con los resultados (ej: resultados_500.txt)")
ap.add_argument("--hw", default="Ryzen 5 5600, WSL2", help="descripcion del hardware para el titulo ('' para omitirla)")
ap.add_argument("--cores", type=int, default=6, help="nucleos fisicos (linea vertical); 0 para omitir")
ap.add_argument("--threads", type=int, default=12, help="hilos logicos (linea vertical); 0 para omitir")
args = ap.parse_args()

seq_times = []
dims = None
times = {"goroutines": defaultdict(list), "procesos": defaultdict(list)}
row = re.compile(r"^(goroutines|procesos)\s+(\d+)\s+([\d.]+)\s")
dim_re = re.compile(r"A:\s*(\d+)x(\d+)\s+B:\s*(\d+)x(\d+)")

with open(args.archivo, encoding="utf-8") as f:
    for line in f:
        m = dim_re.search(line)
        if m and dims is None:
            n, m1, m2, p = map(int, m.groups())
            dims = (n, m1, p)
            continue
        m = re.search(r"Sequential time:\s*([\d.]+)", line)
        if m:
            seq_times.append(float(m.group(1)))
            continue
        m = row.match(line.strip())
        if m:
            times[m.group(1)][int(m.group(2))].append(float(m.group(3)))

if not seq_times or not times["goroutines"]:
    raise SystemExit("No se encontraron datos en el archivo. Revisa que sea la salida de matrix_mul.")

# Tamano de la matriz: "500" si es cuadrada, "NxMxP" si no.
if dims is None:
    size_label, size_title = "desconocido", "tamaño desconocido"
elif dims[0] == dims[1] == dims[2]:
    size_label = str(dims[0])
    size_title = f"{dims[0]}×{dims[0]}"
else:
    size_label = f"{dims[0]}x{dims[1]}x{dims[2]}"
    size_title = f"{dims[0]}×{dims[1]} · {dims[1]}×{dims[2]}"

t_seq = st.mean(seq_times)
ks = sorted(times["goroutines"])
runs = len(seq_times)

stats = {}
for mode in times:
    stats[mode] = {}
    for k in ks:
        v = times[mode][k]
        mean = st.mean(v)
        sd = st.stdev(v) if len(v) > 1 else 0.0
        stats[mode][k] = (mean, sd, t_seq / mean)

csv_path = f"tabla_{size_label}.csv"
png_path = f"speedup_{size_label}.png"

with open(csv_path, "w", encoding="utf-8") as f:
    f.write("K,goroutines_tiempo,goroutines_sd,goroutines_speedup,procesos_tiempo,procesos_sd,procesos_speedup\n")
    for k in ks:
        g, p = stats["goroutines"][k], stats["procesos"][k]
        f.write(f"{k},{g[0]:.4f},{g[1]:.4f},{g[2]:.2f},{p[0]:.4f},{p[1]:.4f},{p[2]:.2f}\n")

print(f"Matrices: {size_title} | Corridas: {runs} | Tiempo secuencial promedio: {t_seq:.3f} s")
print(f"{'K':>3} | {'goroutines (s)':>15} {'speedup':>8} | {'procesos (s)':>13} {'speedup':>8}")
for k in ks:
    g, p = stats["goroutines"][k], stats["procesos"][k]
    print(f"{k:>3} | {g[0]:>8.3f} ±{g[1]:.3f} {g[2]:>7.2f}x | {p[0]:>7.3f} ±{p[1]:.3f} {p[2]:>7.2f}x")

fig, ax = plt.subplots(figsize=(8, 5.5))
ax.plot(ks, ks, "--", color="gray", label="Speedup ideal")
for mode, color, marker in (("goroutines", "tab:blue", "o"), ("procesos", "tab:red", "s")):
    y = [stats[mode][k][2] for k in ks]
    ax.plot(ks, y, marker=marker, color=color, label=mode.capitalize())
if args.cores > 0:
    ax.axvline(args.cores, color="green", linestyle=":", alpha=0.7, label=f"{args.cores} núcleos físicos")
if args.threads > 0:
    ax.axvline(args.threads, color="orange", linestyle=":", alpha=0.7, label=f"{args.threads} hilos lógicos")

subtitulo = f"promedio de {runs} corridas" + (f", {args.hw}" if args.hw else "")
ax.set_xlabel("Número de procesos / goroutines (K)")
ax.set_ylabel("Speedup (T secuencial / T paralelo)")
ax.set_title(f"Speedup en multiplicación de matrices {size_title}\n({subtitulo})")
ax.set_xticks(ks)
ax.grid(alpha=0.3)
ax.legend()
ax.set_xlim(0, max(ks) + 0.5)
ax.set_ylim(0, max(ks) + 0.5)
fig.tight_layout()
fig.savefig(png_path, dpi=150)
print(f"\nGenerados: {png_path} y {csv_path}")