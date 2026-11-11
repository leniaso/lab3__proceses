/*
 * Practica #3 - Multiplicacion de matrices con procesos (Go)
 * Sistemas Operativos - Universidad de Antioquia
 *
 * Integrantes:
 *  - Nombre Apellido Apellido - ID 0000000000
 *  - Nombre Apellido Apellido - ID 0000000000
 *
 * Modos de ejecucion:
 *  1. Secuencial   : un solo proceso, un solo hilo (linea base).
 *  2. Goroutines   : K goroutines dentro del mismo proceso.
 *  3. Procesos     : K procesos hijos REALES (os/exec) que escriben en memoria
 *                    compartida (archivo en /dev/shm mapeado con syscall.Mmap).
 *
 * Nota: Go no tiene fork() utilizable (su runtime es multihilo), por eso el
 * programa se relanza a si mismo como "worker" con os/exec. Es el equivalente
 * a fork() + shmget()/shmat() de C.
 *
 * Formato de las matrices (archivo de texto):
 *   linea 1: filas columnas
 *   siguientes lineas: una fila por linea, numeros enteros separados por espacios
 *
 * Uso:
 *   go build -o matrix_mul matrix_mul.go
 *   ./matrix_mul -gen 1000,1000,1000            # genera A.txt y B.txt
 *   ./matrix_mul -a A.txt -b B.txt -k 4         # compara los 3 modos con K=4
 *   ./matrix_mul -a A.txt -b B.txt -k 1,2,4,8   # tabla de speedup para varios K
 *   ./matrix_mul -a A.txt -b B.txt -k 4 -mode procs   # solo un modo
 */
package main

import (
	"bufio"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
	"unsafe"
)

const intSize = 8 // int64

// ---------------------------------------------------------------------------
// Lectura / escritura de matrices
// ---------------------------------------------------------------------------

func readMatrix(path string) (data []int64, rows, cols int, err error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, 0, err
	}
	defer f.Close()

	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	sc.Split(bufio.ScanWords)

	next := func() (int64, error) {
		if !sc.Scan() {
			if e := sc.Err(); e != nil {
				return 0, e
			}
			return 0, fmt.Errorf("%s: fin de archivo inesperado", path)
		}
		return strconv.ParseInt(sc.Text(), 10, 64)
	}

	r, err := next()
	if err != nil {
		return nil, 0, 0, err
	}
	c, err := next()
	if err != nil {
		return nil, 0, 0, err
	}
	rows, cols = int(r), int(c)
	if rows <= 0 || cols <= 0 {
		return nil, 0, 0, fmt.Errorf("%s: dimensiones invalidas %dx%d", path, rows, cols)
	}
	data = make([]int64, rows*cols)
	for i := range data {
		if data[i], err = next(); err != nil {
			return nil, 0, 0, fmt.Errorf("%s: %v", path, err)
		}
	}
	return data, rows, cols, nil
}

func writeMatrix(path string, data []int64, rows, cols int) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := bufio.NewWriterSize(f, 1<<20)
	fmt.Fprintf(w, "%d %d\n", rows, cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if j > 0 {
				w.WriteByte(' ')
			}
			w.WriteString(strconv.FormatInt(data[i*cols+j], 10))
		}
		w.WriteByte('\n')
	}
	return w.Flush()
}

// generateMatrices crea A.txt (n x m) y B.txt (m x p) con enteros aleatorios 0..9.
func generateMatrices(spec string) error {
	parts := strings.Split(spec, ",")
	if len(parts) != 3 {
		return fmt.Errorf("-gen espera N,M,P (ej: 1000,1000,1000)")
	}
	var d [3]int
	for i, s := range parts {
		v, err := strconv.Atoi(strings.TrimSpace(s))
		if err != nil || v <= 0 {
			return fmt.Errorf("dimension invalida: %q", s)
		}
		d[i] = v
	}
	gen := func(path string, r, c int) error {
		m := make([]int64, r*c)
		for i := range m {
			m[i] = int64(rand.Intn(10))
		}
		return writeMatrix(path, m, r, c)
	}
	if err := gen("A.txt", d[0], d[1]); err != nil {
		return err
	}
	if err := gen("B.txt", d[1], d[2]); err != nil {
		return err
	}
	fmt.Printf("Generadas A.txt (%dx%d) y B.txt (%dx%d)\n", d[0], d[1], d[1], d[2])
	return nil
}

// ---------------------------------------------------------------------------
// Nucleo del calculo: filas [start, end) de C = A * B
// A es n x m, B es m x p, C es n x p (todas row-major).
// Orden i-k-j: recorre B por filas, mucho mas amigable con la cache que i-j-k.
// ---------------------------------------------------------------------------

func mulRows(a, b, c []int64, m, p, start, end int) {
	for i := start; i < end; i++ {
		ci := c[i*p : (i+1)*p]
		for j := range ci {
			ci[j] = 0
		}
		for k := 0; k < m; k++ {
			aik := a[i*m+k]
			bk := b[k*p : (k+1)*p]
			for j, v := range bk {
				ci[j] += aik * v
			}
		}
	}
}

// splitRows reparte n filas entre k trabajadores (los primeros n%k reciben una extra).
func splitRows(n, k int) [][2]int {
	if k > n {
		k = n
	}
	ranges := make([][2]int, 0, k)
	base, rem := n/k, n%k
	start := 0
	for i := 0; i < k; i++ {
		size := base
		if i < rem {
			size++
		}
		ranges = append(ranges, [2]int{start, start + size})
		start += size
	}
	return ranges
}

// ---------------------------------------------------------------------------
// Modo 1: secuencial
// ---------------------------------------------------------------------------

func multiplySequential(a, b []int64, n, m, p int) ([]int64, time.Duration) {
	c := make([]int64, n*p)
	t0 := time.Now()
	mulRows(a, b, c, m, p, 0, n)
	return c, time.Since(t0)
}

// ---------------------------------------------------------------------------
// Modo 2: K goroutines (hilos ligeros dentro de UN solo proceso)
// ---------------------------------------------------------------------------

func multiplyGoroutines(a, b []int64, n, m, p, k int) ([]int64, time.Duration) {
	c := make([]int64, n*p)
	var wg sync.WaitGroup
	t0 := time.Now()
	for _, r := range splitRows(n, k) {
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			mulRows(a, b, c, m, p, start, end) // cada goroutine escribe filas distintas: sin carreras
		}(r[0], r[1])
	}
	wg.Wait()
	return c, time.Since(t0)
}

// ---------------------------------------------------------------------------
// Modo 3: K procesos reales + memoria compartida (mmap)
//
// Layout del segmento compartido: [ A (n*m) | B (m*p) | C (n*p) ] de int64.
// El padre copia A y B, lanza los hijos y espera. Cada hijo mapea el mismo
// archivo y escribe solo sus filas de C.
// ---------------------------------------------------------------------------

func asInt64(b []byte) []int64 {
	return unsafe.Slice((*int64)(unsafe.Pointer(&b[0])), len(b)/intSize)
}

func openShm(path string, size int) ([]byte, *os.File, error) {
	f, err := os.OpenFile(path, os.O_RDWR, 0600)
	if err != nil {
		return nil, nil, err
	}
	mem, err := syscall.Mmap(int(f.Fd()), 0, size, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_SHARED)
	if err != nil {
		f.Close()
		return nil, nil, err
	}
	return mem, f, nil
}

func multiplyProcesses(a, b []int64, n, m, p, k int) ([]int64, time.Duration, error) {
	total := n*m + m*p + n*p
	size := total * intSize

	// /dev/shm es memoria RAM en Linux; si no existe (macOS) se usa el directorio temporal.
	dir := "/dev/shm"
	if st, err := os.Stat(dir); err != nil || !st.IsDir() {
		dir = os.TempDir()
	}
	f, err := os.CreateTemp(dir, "matmul-shm-*")
	if err != nil {
		return nil, 0, err
	}
	path := f.Name()
	defer os.Remove(path)
	if err := f.Truncate(int64(size)); err != nil {
		f.Close()
		return nil, 0, err
	}
	f.Close()

	mem, mf, err := openShm(path, size)
	if err != nil {
		return nil, 0, err
	}
	defer mf.Close()
	defer syscall.Munmap(mem)

	shm := asInt64(mem)
	copy(shm[:n*m], a)
	copy(shm[n*m:n*m+m*p], b)

	self, err := os.Executable()
	if err != nil {
		return nil, 0, err
	}

	// El cronometro incluye crear los hijos y esperarlos (el costo real de usar procesos).
	t0 := time.Now()
	ranges := splitRows(n, k)
	cmds := make([]*exec.Cmd, 0, len(ranges))
	for _, r := range ranges {
		cmd := exec.Command(self, "worker", path,
			strconv.Itoa(n), strconv.Itoa(m), strconv.Itoa(p),
			strconv.Itoa(r[0]), strconv.Itoa(r[1]))
		cmd.Stderr = os.Stderr
		if err := cmd.Start(); err != nil {
			for _, c := range cmds { // si no se pudo lanzar uno, se cancelan los demas
				c.Process.Kill()
				c.Wait()
			}
			return nil, 0, fmt.Errorf("no se pudo lanzar el proceso hijo: %v", err)
		}
		cmds = append(cmds, cmd)
	}

	var firstErr error
	for i, cmd := range cmds {
		if err := cmd.Wait(); err != nil && firstErr == nil {
			firstErr = fmt.Errorf("el hijo %d (filas %d-%d) fallo: %v", i, ranges[i][0], ranges[i][1]-1, err)
		}
	}
	elapsed := time.Since(t0)
	if firstErr != nil {
		return nil, elapsed, firstErr
	}

	c := make([]int64, n*p)
	copy(c, shm[n*m+m*p:])
	return c, elapsed, nil
}

// workerMain se ejecuta en el proceso HIJO: ./matrix_mul worker <shm> n m p start end
func workerMain(args []string) error {
	if len(args) != 6 {
		return fmt.Errorf("uso interno: worker <shm> n m p start end")
	}
	var v [5]int
	for i := 0; i < 5; i++ {
		x, err := strconv.Atoi(args[i+1])
		if err != nil {
			return err
		}
		v[i] = x
	}
	n, m, p, start, end := v[0], v[1], v[2], v[3], v[4]
	size := (n*m + m*p + n*p) * intSize

	mem, f, err := openShm(args[0], size)
	if err != nil {
		return err
	}
	defer f.Close()
	defer syscall.Munmap(mem)

	shm := asInt64(mem)
	a := shm[:n*m]
	b := shm[n*m : n*m+m*p]
	c := shm[n*m+m*p:]
	mulRows(a, b, c, m, p, start, end)
	return nil
}

// ---------------------------------------------------------------------------
// Utilidades y main
// ---------------------------------------------------------------------------

func equal(x, y []int64) bool {
	if len(x) != len(y) {
		return false
	}
	for i := range x {
		if x[i] != y[i] {
			return false
		}
	}
	return true
}

func parseKs(s string) ([]int, error) {
	var ks []int
	for _, part := range strings.Split(s, ",") {
		k, err := strconv.Atoi(strings.TrimSpace(part))
		if err != nil || k < 1 {
			return nil, fmt.Errorf("valor de -k invalido: %q", part)
		}
		ks = append(ks, k)
	}
	return ks, nil
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "error:", err)
	os.Exit(1)
}

func main() {
	// Modo interno: este mismo ejecutable actua como proceso hijo.
	if len(os.Args) > 1 && os.Args[1] == "worker" {
		if err := workerMain(os.Args[2:]); err != nil {
			fmt.Fprintln(os.Stderr, "worker:", err)
			os.Exit(1)
		}
		return
	}

	aPath := flag.String("a", "A.txt", "archivo de la matriz A")
	bPath := flag.String("b", "B.txt", "archivo de la matriz B")
	outPath := flag.String("out", "C.txt", "archivo de salida para C")
	ksFlag := flag.String("k", "4", "numero de procesos/goroutines; lista separada por comas (ej: 1,2,4,8)")
	mode := flag.String("mode", "all", "seq | goroutines | procs | all")
	gen := flag.String("gen", "", "genera A.txt y B.txt aleatorias con N,M,P (ej: 1000,1000,1000) y termina")
	flag.Parse()

	if *gen != "" {
		if err := generateMatrices(*gen); err != nil {
			fatal(err)
		}
		return
	}

	ks, err := parseKs(*ksFlag)
	if err != nil {
		fatal(err)
	}
	a, n, m, err := readMatrix(*aPath)
	if err != nil {
		fatal(err)
	}
	b, m2, p, err := readMatrix(*bPath)
	if err != nil {
		fatal(err)
	}
	if m != m2 {
		fatal(fmt.Errorf("dimensiones incompatibles: A es %dx%d y B es %dx%d", n, m, m2, p))
	}
	fmt.Printf("A: %dx%d  B: %dx%d  C: %dx%d\n\n", n, m, m2, p, n, p)

	// Linea base secuencial (siempre se calcula: sirve de referencia y para verificar).
	ref, tSeq := multiplySequential(a, b, n, m, p)
	fmt.Printf("Sequential time: %.3f seconds\n\n", tSeq.Seconds())
	result := ref

	if *mode != "seq" {
		fmt.Printf("%-12s %4s %12s %10s\n", "Modo", "K", "Tiempo (s)", "Speedup")
		for _, k := range ks {
			if *mode == "goroutines" || *mode == "all" {
				c, t := multiplyGoroutines(a, b, n, m, p, k)
				status := ""
				if !equal(c, ref) {
					status = "  (RESULTADO DIFERENTE!)"
				}
				fmt.Printf("%-12s %4d %12.3f %9.2fx%s\n", "goroutines", k, t.Seconds(), tSeq.Seconds()/t.Seconds(), status)
			}
			if *mode == "procs" || *mode == "all" {
				c, t, err := multiplyProcesses(a, b, n, m, p, k)
				if err != nil {
					fmt.Printf("%-12s %4d  ERROR: %v\n", "procesos", k, err)
					continue
				}
				status := ""
				if !equal(c, ref) {
					status = "  (RESULTADO DIFERENTE!)"
				}
				fmt.Printf("%-12s %4d %12.3f %9.2fx%s\n", "procesos", k, t.Seconds(), tSeq.Seconds()/t.Seconds(), status)
				result = c
			}
		}
	}

	if err := writeMatrix(*outPath, result, n, p); err != nil {
		fatal(err)
	}
	fmt.Printf("\nResultado escrito en %s\n", *outPath)
}