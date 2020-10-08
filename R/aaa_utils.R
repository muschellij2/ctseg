check_requirements = function() {
  have = list_requirements()
  all(have)
}

list_requirements = function() {
  reticulate::py_module_available("numpy")
  if (!reticulate::py_module_available("numpy")) {
    stop("numpy is not available - use reticulate::py_install(\"numpy\")")
  }
  if (!reticulate::py_module_available("nibabel")) {
    stop("nibabel is not available - use reticulate::py_install(\"nibabel\")")
  }
  fname = system.file("requirements.txt", package = "ctseg")
  suppressWarnings({
    deps = readLines(fname, warn = FALSE)
  })
  deps = gsub("(<|=|>)*=.*", "", deps)
  deps = trimws(deps)
  packages = deps
  packages[packages == "keras-applications"] = "keras_applications"
  packages[packages == "scikit-learn"] = "sklearn"
  have = sapply(packages, reticulate::py_module_available)
  have = sapply(packages, reticulate::py_module_available)
  have
}