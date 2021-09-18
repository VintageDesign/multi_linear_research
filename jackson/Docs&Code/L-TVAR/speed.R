library(matlib)

N = 100
StartN = 2
NumTests = 100
avgs = rep(0, N-StartN)
for (n in StartN:N) {
  A = matrix(rnorm(n*n), nrow=n)
  times = rep(0, NumTests)
  print(n)
  for (i in 1:NumTests) {
    start = Sys.time()
    tmp = inv(A)
    end = Sys.time()
    times[i] = end - start
  }
  avgs[n-StartN+1] = mean(times)
}