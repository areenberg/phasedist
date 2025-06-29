#----------------------------------------------------
# BENCHMARK TEST: CPH Erlang-3 (R Code)
# See the file 'erlang3.py'
#----------------------------------------------------

# Install dependencies (do not change order)
install.packages("benchmarks/R_lib/cli_3.6.5.tar.gz", repos = NULL, type = "source", lib="benchmarks/R_lib")
install.packages("benchmarks/R_lib/glue_1.8.0.tar.gz", repos = NULL, type = "source", lib="benchmarks/R_lib")
install.packages("benchmarks/R_lib/rlang_1.1.6.tar.gz", repos = NULL, type = "source", lib="benchmarks/R_lib")
install.packages("benchmarks/R_lib/lifecycle_1.0.4.tar.gz", repos = NULL, type = "source", lib="benchmarks/R_lib")
install.packages("benchmarks/R_lib/magrittr_2.0.3.tar.gz", repos = NULL, type = "source", lib="benchmarks/R_lib")
install.packages("benchmarks/R_lib/stringi_1.8.7.tar.gz", repos = NULL, type = "source", lib="benchmarks/R_lib")
install.packages("benchmarks/R_lib/vctrs_0.6.5.tar.gz", repos = NULL, type = "source", lib="benchmarks/R_lib")
install.packages("benchmarks/R_lib/stringr_1.5.1.tar.gz", repos = NULL, type = "source", lib="benchmarks/R_lib")
install.packages("benchmarks/R_lib/plyr_1.8.9.tar.gz", repos = NULL, type = "source", lib="benchmarks/R_lib")
install.packages("benchmarks/R_lib/reshape2_1.4.4.tar.gz", repos = NULL, type = "source", lib="benchmarks/R_lib")
install.packages("benchmarks/R_lib/Rcpp_1.0.14.tar.gz", repos = NULL, type = "source", lib="benchmarks/R_lib")
install.packages("benchmarks/R_lib/RcppArmadillo_14.4.3-1.tar.gz", repos = NULL, type = "source", lib="benchmarks/R_lib")
install.packages("benchmarks/R_lib/matrixdist_1.1.9.tar.gz", repos = NULL, type = "source", lib="benchmarks/R_lib")

# Load matrixdist
library(matrixdist,lib="benchmarks/R_lib")

# Load data
obs <- read.table(file="benchmarks/erlang3_data.csv",header=TRUE)

# Fit the model
obj <- ph(structure = "gerlang", dimension = 3)
f <- fit(obj, obs$Samples, stepsEM = 1000, every = 100)
initdist <- f@pars$alpha
phgen <- f@pars$S

# Save the parameters
save(initdist,phgen, file = "benchmarks/ph_parameters.RData")