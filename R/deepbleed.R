#' CTSeg Model
#'
#' @param outdir Output directory for `CTSeg` model
#'
#' @note \url{https://github.com/jasonccai/HeadCTSegmentation/}
#'
#' @return A list of the output images and predictions.
#' @export
#' @rdname ctseg
#'
#' @examples
#' \donttest{
#' destfile = file.path(tempdir(), "01.tar.xz")
#' dl = download.file(
#'   "https://archive.data.jhu.edu/api/access/datafile/1311?gbrecs=true",
#'   destfile = destfile)
#' res = untar(tarfile = destfile, exdir = tempdir())
#' fname = file.path(tempdir(), "01", "BRAIN_1_Anonymized.nii.gz")
#' mask = file.path(tempdir(), "01", "BRAIN_1_Anonymized_Mask.nii.gz")
#' tdir = tempfile()
#' dir.create(tdir)
#' download_ctseg_model(outdir = tdir)
#' mod = load_ctseg_model(outdir = tdir)
#' predict_ctseg(fname, mask = mask, outdir = tdir)
#' }
download_ctseg_model = function(
  outdir = NULL,
  type = c("NPH", "Normal_Only")) {
  if (is.null(outdir)) {
    outdir = system.file(package = "ctseg")
  }
  type = match.arg(
    type[1],
    choices = c("NPH", "Review", "Normal_Only",
                "unet_CT_SS_3D_201843_163521.h5",
                "unet_CT_SS_20171114_170726.h5"))
  if (type == "Normal_Only") {
    type = "Review"
  }
  ids = c(
    Review = "1h-mS_JywoBotS_qT88ob0qfzYfEqu0vS",
    NPH = "1fSe7oDaE8NYh5GLUN6h-Yo-ZlrJHXeuZ",
    unet_CT_SS_3D_201843_163521.h5 = "1YJZjLDp9edyUPVwnLqgOKZoNUWYo6Lat",
    unet_CT_SS_20171114_170726.h5 = "1Iq-wIlB-MmWFXgG9tZ5aQhZs2vUBWIOG")

  id = ids[[type]]

  dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
  outfile = file.path(outdir, paste0(type, "_weights.hdf5"))

  # drive_url = "https://drive.google.com/"
  # url = paste0(drive_url, "uc?id=",
  #              id, "&export=download")
  #   sub_url = xml2::read_html(url) %>%
  #     xml2::xml_find_all(xpath = "//a[@id='uc-download-link']") %>%
  #     xml2::xml_attr(attr = "href")
  #   if (nchar(sub_url) > 0) {
  #     url = paste0(drive_url, sub_url)
  #   }
  # curl::curl_download(url, destfile = outfile, quiet = FALSE)
  if (!file.exists(outfile)) {
    id = googledrive::as_id(id)
    out = googledrive::drive_download(file = id, path = outfile)
  }
  stopifnot(all(file.exists(outfile)))

  outfile = path.expand(outfile)
  outfile = normalizePath(outfile)
  return(outfile)
}

#' @rdname ctseg
#' @export
download_ctbet_model = function(
  outdir = NULL,
  type = c("unet_CT_SS_20171114_170726.h5",
           "unet_CT_SS_3D_201843_163521.h5")) {
  download_ctseg_model(outdir = outdir, type = type[1])
}

#' @rdname ctseg
#' @export
load_ctbet_model = function(outdir = NULL,
                            type = c("unet_CT_SS_20171114_170726.h5",
                                     "unet_CT_SS_3D_201843_163521.h5")
) {
  outfile = download_ctbet_model(outdir, type = type)
  try({reticulate::import("h5py")}, silent = TRUE)
  # need to configure model
  pkg_path = system.file(package = "ctseg")
  model = reticulate::import_from_path(module = "deepModels",
                                       path = pkg_path)
  model$load_weights(outfile)
  model$compile()
  model
}

#' @rdname ctseg
#' @export
load_ctseg_model = function(outdir = NULL,
                            type = c("NPH", "Normal_Only")) {
  outfile = download_ctseg_model(outdir, type = type)
  try({reticulate::import("h5py")}, silent = TRUE)
  model = keras::load_model_hdf5(filepath = outfile)
  model
}

