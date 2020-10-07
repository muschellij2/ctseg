library(reticulate)
library(neurobase)
deps = readLines("requirements.txt")
deps = gsub("(<|=|>)*=.*", "", deps)
deps = trimws(deps)
packages = deps
packages[packages == "keras-applications"] = "keras_applications"
packages[packages == "scikit-learn"] = "sklearn"
have = sapply(packages, py_module_available)
have = sapply(packages, py_module_available)
if (any(!have)) {
  packages = deps[!have]
  py_install(packages = packages)
}



destfile = file.path(tempdir(), "01.tar.xz")
dl = download.file("https://archive.data.jhu.edu/api/access/datafile/1311?gbrecs=true",
                   destfile = destfile)
res = untar(tarfile = destfile, exdir = tempdir())
fname = file.path(tempdir(), "01", "BRAIN_1_Anonymized.nii.gz")
mask = file.path(tempdir(), "01", "BRAIN_1_Anonymized_Mask.nii.gz")
# model = reticulate::import_from_path(module = "deepModels", path = "inst/")
img = ANTsRCore::antsImageRead(fname)
data_subset = FALSE
if (dim(img)[3] > 48) {
  data_subset = TRUE
  new_img = extrantsr::resample_image(img, c(pixdim(img)[2:3], 5))
  tfile = tempfile(fileext = ".nii.gz")
  ANTsRCore::antsImageWrite(new_img, tfile)
  fname = tfile
}
dataAugmentation = TRUE


library(ctseg)
# image = c(fname, fname)
image = fname
outdir = tempdir()
type = "unet_CT_SS_20171114_170726.h5"
type = "unet_CT_SS_3D_201843_163521.h5"
