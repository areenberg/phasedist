#----------------------------------------------------
# R-file for validation test: CPH Erlang-3
# To run the test, use the file 'erlang3.py'
#----------------------------------------------------

# Install dependencies (do not change order)
install.packages("https://cran.r-project.org/src/contrib/Archive/cli/cli_3.6.4.tar.gz",
repos = NULL, type = "source", lib = "validation/R_lib")
install.packages("https://cran.r-project.org/src/contrib/Archive/glue/glue_1.7.0.tar.gz",
repos = NULL, type = "source", lib = "validation/R_lib")
install.packages("https://cran.r-project.org/src/contrib/Archive/rlang/rlang_1.1.5.tar.gz",
repos = NULL, type = "source", lib = "validation/R_lib")
install.packages("https://cran.r-project.org/src/contrib/Archive/lifecycle/lifecycle_1.0.3.tar.gz",
repos = NULL, type = "source", lib = "validation/R_lib")
install.packages("https://cran.r-project.org/src/contrib/Archive/magrittr/magrittr_2.0.2.tar.gz",
repos = NULL, type = "source", lib = "validation/R_lib")
install.packages("https://cran.r-project.org/src/contrib/Archive/stringi/stringi_1.8.4.tar.gz",
repos = NULL, type = "source", lib = "validation/R_lib")
install.packages("https://cran.r-project.org/src/contrib/Archive/vctrs/vctrs_0.6.4.tar.gz",
repos = NULL, type = "source", lib = "validation/R_lib")
install.packages("https://cran.r-project.org/src/contrib/Archive/stringr/stringr_1.5.0.tar.gz",
repos = NULL, type = "source", lib = "validation/R_lib")
install.packages("https://cran.r-project.org/src/contrib/Archive/Rcpp/Rcpp_1.0.14.tar.gz",
repos = NULL, type = "source", lib = "validation/R_lib")
install.packages("https://cran.r-project.org/src/contrib/Archive/plyr/plyr_1.8.8.tar.gz",
repos = NULL, type = "source", lib = "validation/R_lib")
install.packages("https://cran.r-project.org/src/contrib/Archive/reshape/reshape_0.8.9.tar.gz",
repos = NULL, type = "source", lib = "validation/R_lib")
install.packages("https://cran.r-project.org/src/contrib/Archive/reshape2/reshape2_1.4.3.tar.gz",
repos = NULL, type = "source", lib = "validation/R_lib")
install.packages("https://cran.r-project.org/src/contrib/Archive/RcppArmadillo/RcppArmadillo_14.4.3-1.tar.gz",
repos = NULL, type = "source", lib = "validation/R_lib")
install.packages("https://cran.r-project.org/src/contrib/Archive/matrixdist/matrixdist_1.1.8.tar.gz",
repos = NULL, type = "source", lib = "validation/R_lib")

# Load matrixdist
library(matrixdist,lib="validation/R_lib")

# Load data
obs <- read.table(file="validation/erlang3_data.csv",header=TRUE)

# Fit the model
obj <- ph(structure = "gerlang", dimension = 3)
f <- fit(obj, obs$Samples, stepsEM = 1000, every = 100)
initdist <- f@pars$alpha
phgen <- f@pars$S

# Save the parameters
save(initdist,phgen, file = "validation/erlang3_ph_parameters.RData")