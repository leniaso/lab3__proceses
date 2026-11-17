#!/bin/bash
# Compila matrix_mul.c, genera matrices 500x500 y 1000x1000 y mide K = 1..16.
# Salida: resultados_C_500.csv, resultados_C_1000.csv e info_entorno.txt
set -e
cd "$(dirname "$0")"

gcc -O2 -Wall -o matrix_mul matrix_mul.c -lm

{
  echo "== CPU ==";      lscpu | grep -E "Model name|^CPU\(s\)|Thread|Core|Socket"
  echo "== RAM ==";      free -h | head -2
  echo "== SO ==";       uname -a; grep PRETTY_NAME /etc/os-release || true
  echo "== Compilador =="; gcc --version | head -1
} > info_entorno.txt

for n in 500 1000; do
  ./matrix_mul gen $n $n A_$n.txt 1
  ./matrix_mul gen $n $n B_$n.txt 2
  echo "Midiendo ${n}x${n} (5 repeticiones por K)..."
  ./matrix_mul A_$n.txt B_$n.txt bench 5 > resultados_C_$n.csv
  cat resultados_C_$n.csv
done
echo "Listo. Envíame resultados_C_500.csv, resultados_C_1000.csv e info_entorno.txt"
