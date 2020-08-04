#' @title Install Pycox With Reticulate
#' @description Installs the python 'pycox' package via reticulate.
#' @param method,conda,pip See [reticulate::py_install]
#' @param install_torch If `TRUE` installs the dependency `torch` package as well.
#' @export
install_pycox <- function(method = "auto", conda = "auto", pip = FALSE, install_torch = FALSE) {
  pkg = "pycox"
  if (install_torch) {
    pkg = c("torch", pkg)
  }
  reticulate::py_install(pkg, method = method, conda = conda, pip = pip)
}

#' @title Install Torch With Reticulate
#' @description Installs the python 'torch' package via reticulate.
#' @param method,conda See [reticulate::py_install]
#' @export
install_torch <- function(method = "auto", conda = "auto", pip = FALSE) {
  reticulate::py_install("torch", method = method, conda = conda, pip = pip)
}