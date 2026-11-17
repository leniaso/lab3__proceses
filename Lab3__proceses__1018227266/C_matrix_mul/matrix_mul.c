/*
 * Práctica 3 - Multiplicación de matrices usando procesos
 * Sistemas Operativos - Universidad de Antioquia
 *
 * Integrantes:
 *   Alejandro Naranjo Cataño - ID 1018227266
 *   Simon Correa Rios        - ID [COMPLETAR]
 *
 * Implementación en C: versión secuencial (línea base) y versión paralela con
 * K procesos creados con fork(). IPC elegido: memoria compartida System V
 * (shmget/shmat). Las matrices A, B y C viven en un único segmento [ A | B | C ];
 * cada hijo escribe solo las filas de C que le corresponden, así que no hay
 * condiciones de carrera ni hace falta sincronización (basta con waitpid).
 *
 * Compilar:
 *   gcc -O2 -Wall -o matrix_mul matrix_mul.c
 *   (opcional, bucle i-j-k de la guía en lugar de i-k-j: -DLOOP_IJK)
 *
 * Uso:
 *   ./matrix_mul A.txt B.txt K [reps]      secuencial vs. K procesos; escribe C.txt
 *   ./matrix_mul A.txt B.txt bench [reps]  barrido K = 1,2,3,4,5,6,7,8,10,12,16 (CSV)
 *   ./matrix_mul gen N M archivo [semilla] genera una matriz N x M con enteros 0..9
 *
 * Formato de los archivos: primera línea "filas columnas", luego una fila por
 * línea con enteros separados por espacios (el mismo formato de C.txt).
 * Funciona con matrices no cuadradas: A (N x M) por B (M x P).
 */
#define _GNU_SOURCE
#include <errno.h>
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

typedef long long ll; /* enteros de 64 bits, igual que la versión en Go */

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ---------- Entrada / salida ---------- */

static ll *read_matrix(const char *path, int *r, int *c) {
    FILE *f = fopen(path, "r");
    if (!f) { perror(path); return NULL; }
    if (fscanf(f, "%d %d", r, c) != 2 || *r <= 0 || *c <= 0) {
        fprintf(stderr, "%s: encabezado inválido (se esperaba \"filas columnas\")\n", path);
        fclose(f);
        return NULL;
    }
    size_t n = (size_t)*r * (size_t)*c;
    ll *m = malloc(n * sizeof(ll));
    if (!m) { fprintf(stderr, "%s: sin memoria\n", path); fclose(f); return NULL; }
    for (size_t i = 0; i < n; i++) {
        if (fscanf(f, "%lld", &m[i]) != 1) {
            fprintf(stderr, "%s: faltan datos (elemento %zu de %zu)\n", path, i, n);
            free(m);
            fclose(f);
            return NULL;
        }
    }
    fclose(f);
    return m;
}

static int write_matrix(const char *path, const ll *m, int r, int c) {
    FILE *f = fopen(path, "w");
    if (!f) { perror(path); return -1; }
    fprintf(f, "%d %d\n", r, c);
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++)
            fprintf(f, "%lld%c", m[(size_t)i * c + j], j == c - 1 ? '\n' : ' ');
    fclose(f);
    return 0;
}

/* ---------- Cálculo ---------- */

/* Calcula las filas [r0, r1) de C = A (N x M) * B (M x P). */
static void mul_rows(const ll *A, const ll *B, ll *C, int M, int P, int r0, int r1) {
    for (int i = r0; i < r1; i++) {
        ll *Ci = C + (size_t)i * P;
#ifdef LOOP_IJK
        for (int j = 0; j < P; j++) {
            ll acc = 0;
            for (int k = 0; k < M; k++)
                acc += A[(size_t)i * M + k] * B[(size_t)k * P + j];
            Ci[j] = acc;
        }
#else
        memset(Ci, 0, (size_t)P * sizeof(ll));
        for (int k = 0; k < M; k++) {
            ll a = A[(size_t)i * M + k];
            const ll *Bk = B + (size_t)k * P;
            for (int j = 0; j < P; j++)
                Ci[j] += a * Bk[j];
        }
#endif
    }
}

/*
 * Versión paralela: K procesos hijos, cada uno con un rango de filas.
 * Reparto: las N filas se dividen entre K; los primeros N mod K procesos
 * reciben una fila extra. Si K > N se usan solo N procesos.
 * Devuelve 0 si todos los hijos terminaron bien, -1 en caso contrario.
 */
static int parallel_mul(const ll *A, const ll *B, ll *C, int N, int M, int P, int K) {
    if (K > N) K = N;
    pid_t *pids = malloc((size_t)K * sizeof(pid_t));
    int *r0s = malloc((size_t)K * sizeof(int));
    int *r1s = malloc((size_t)K * sizeof(int));
    if (!pids || !r0s || !r1s) { fprintf(stderr, "sin memoria\n"); free(pids); free(r0s); free(r1s); return -1; }

    int base = N / K, rem = N % K, start = 0, launched = 0, failed = 0;

    for (int w = 0; w < K; w++) {
        int end = start + base + (w < rem ? 1 : 0);
        pid_t p = fork();
        if (p < 0) {
            perror("fork");
            failed = 1;
            break;
        }
        if (p == 0) { /* hijo: A, B y C están en el segmento compartido heredado */
            mul_rows(A, B, C, M, P, start, end);
            _exit(0);
        }
        pids[launched] = p; r0s[launched] = start; r1s[launched] = end; launched++;
        start = end;
    }

    if (failed)  /* no se pudo lanzar un hijo: cancelar los ya iniciados */
        for (int i = 0; i < launched; i++) kill(pids[i], SIGTERM);

    for (int i = 0; i < launched; i++) {
        int st;
        if (waitpid(pids[i], &st, 0) < 0) { perror("waitpid"); failed = 1; continue; }
        if (!WIFEXITED(st) || WEXITSTATUS(st) != 0) {
            fprintf(stderr, "Error: el hijo %d (filas %d a %d) terminó de forma anormal\n",
                    (int)pids[i], r0s[i], r1s[i] - 1);
            if (!failed)  /* primer fallo: cancelar a los demás */
                for (int j = i + 1; j < launched; j++) kill(pids[j], SIGTERM);
            failed = 1;
        }
    }
    free(pids); free(r0s); free(r1s);
    return failed ? -1 : 0;
}

/* ---------- Programa principal ---------- */

static void mean_std(const double *t, int n, double *mean, double *std) {
    double s = 0;
    for (int i = 0; i < n; i++) s += t[i];
    *mean = s / n;
    double v = 0;
    for (int i = 0; i < n; i++) v += (t[i] - *mean) * (t[i] - *mean);
    *std = n > 1 ? sqrt(v / (n - 1)) : 0.0;
}

static int cmd_gen(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "Uso: %s gen N M archivo [semilla]\n", argv[0]); return 1; }
    int n = atoi(argv[2]), m = atoi(argv[3]);
    unsigned seed = argc > 5 ? (unsigned)atoi(argv[5]) : 42u;
    if (n <= 0 || m <= 0) { fprintf(stderr, "Dimensiones inválidas\n"); return 1; }
    srand(seed);
    FILE *f = fopen(argv[4], "w");
    if (!f) { perror(argv[4]); return 1; }
    fprintf(f, "%d %d\n", n, m);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            fprintf(f, "%d%c", rand() % 10, j == m - 1 ? '\n' : ' ');
    fclose(f);
    return 0;
}

int main(int argc, char **argv) {
    if (argc >= 2 && strcmp(argv[1], "gen") == 0) return cmd_gen(argc, argv);
    if (argc < 4) {
        fprintf(stderr, "Uso:\n  %s A.txt B.txt K [reps]\n  %s A.txt B.txt bench [reps]\n  %s gen N M archivo [semilla]\n",
                argv[0], argv[0], argv[0]);
        return 1;
    }

    int bench = strcmp(argv[3], "bench") == 0;
    int K = 0;
    int reps = argc > 4 ? atoi(argv[4]) : (bench ? 5 : 1);
    if (!bench) {
        K = atoi(argv[3]);
        if (K < 1) { fprintf(stderr, "K debe ser un entero >= 1\n"); return 1; }
    }
    if (reps < 1) { fprintf(stderr, "reps debe ser >= 1\n"); return 1; }

    /* Lectura de las matrices (no se incluye en las mediciones) */
    int N, M, M2, P;
    ll *A0 = read_matrix(argv[1], &N, &M);
    ll *B0 = read_matrix(argv[2], &M2, &P);
    if (!A0 || !B0) return 1;
    if (M != M2) {
        fprintf(stderr, "Dimensiones incompatibles: A es %dx%d y B es %dx%d\n", N, M, M2, P);
        return 1;
    }

    /* Segmento de memoria compartida [ A | B | C ] */
    size_t nA = (size_t)N * M, nB = (size_t)M * P, nC = (size_t)N * P;
    int shmid = shmget(IPC_PRIVATE, (nA + nB + nC) * sizeof(ll), IPC_CREAT | 0600);
    if (shmid < 0) { perror("shmget"); return 1; }
    ll *shm = shmat(shmid, NULL, 0);
    if (shm == (void *)-1) { perror("shmat"); shmctl(shmid, IPC_RMID, NULL); return 1; }
    /* Se marca para destruir de inmediato: el kernel lo libera al último shmdt/exit,
       así que no queda basura en el sistema aunque el programa falle. */
    shmctl(shmid, IPC_RMID, NULL);

    ll *A = shm, *B = shm + nA, *C = shm + nA + nB;
    memcpy(A, A0, nA * sizeof(ll));   /* la copia inicial no entra en el tiempo medido */
    memcpy(B, B0, nB * sizeof(ll));
    free(A0); free(B0);

    ll *Cseq = malloc(nC * sizeof(ll));
    if (!Cseq) { fprintf(stderr, "sin memoria\n"); return 1; }

    /* Línea base secuencial */
    double *ts = malloc((size_t)reps * sizeof(double));
    for (int r = 0; r < reps; r++) {
        double t0 = now();
        mul_rows(A, B, Cseq, M, P, 0, N);
        ts[r] = now() - t0;
    }
    double seq_mean, seq_std;
    mean_std(ts, reps, &seq_mean, &seq_std);

    int Ks_bench[] = {1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16};
    int nK = bench ? (int)(sizeof(Ks_bench) / sizeof(int)) : 1;

    if (bench) {
        printf("# N=%d M=%d P=%d reps=%d\n", N, M, P, reps);
        printf("K,tiempo_s,std_s,speedup\n");
        printf("secuencial,%.6f,%.6f,1.00\n", seq_mean, seq_std);
    }

    for (int idx = 0; idx < nK; idx++) {
        int k = bench ? Ks_bench[idx] : K;
        for (int r = 0; r < reps; r++) {
            double t0 = now();
            int rc = parallel_mul(A, B, C, N, M, P, k);
            ts[r] = now() - t0;
            if (rc != 0) { fprintf(stderr, "La versión paralela falló; no se entrega resultado parcial.\n"); return 1; }
            if (memcmp(C, Cseq, nC * sizeof(ll)) != 0) {  /* verificación fuera del tiempo medido */
                fprintf(stderr, "ERROR: el resultado paralelo (K=%d) difiere del secuencial\n", k);
                return 1;
            }
        }
        double mean, std;
        mean_std(ts, reps, &mean, &std);
        if (bench) {
            printf("%d,%.6f,%.6f,%.2f\n", k, mean, std, seq_mean / mean);
        } else {
            printf("Sequential time: %.4f seconds\n", seq_mean);
            printf("Parallel time (%d processes): %.4f seconds\n", k > N ? N : k, mean);
            printf("Speedup: %.2fx\n", seq_mean / mean);
            printf("Verificación: el resultado paralelo coincide con el secuencial\n");
            if (write_matrix("C.txt", C, N, P) != 0) return 1;
            printf("Resultado escrito en C.txt (%d x %d)\n", N, P);
        }
    }

    free(ts);
    free(Cseq);
    shmdt(shm);
    return 0;
}
