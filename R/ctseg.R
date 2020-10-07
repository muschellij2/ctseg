#' @rdname ctseg
#' @param image image to segment using `HeadCTSegmentation` model
#' @param mask brain mask image
#' @param verbose print diagnostic messages
#' @param ... additional arguments to send to
#' \code{\link{CT_Skull_Stripper_mask}}
#' @export
#' @examples
#' url = paste0("https://github.com/jasonccai/HeadCTSegmentation/",
#' "raw/master/image_data_predict/1.nii.gz")
#' image = tempfile(fileext = ".nii.gz")
#' utils::download.file(url, destfile = image, quiet = FALSE)
#' out = predict_ctseg(image)
predict_ctseg = function(image,
                         register = FALSE,
                         verbose = TRUE,
                         ...,
                         outdir = NULL,
                         type = c("NPH", "Normal_Only")
) {

  type = match.arg(type)
  reticulate::py_module_available("numpy")
  if (!reticulate::py_module_available("numpy")) {
    stop("numpy is not available - use reticulate::py_install(\"numpy\")")
  }
  if (!reticulate::py_module_available("nibabel")) {
    stop("nibabel is not available - use reticulate::py_install(\"nibabel\")")
  }
  nb = reticulate::import("nibabel")
  np = reticulate::import("numpy")

  nim = neurobase::check_nifti(image)
  fname = neurobase::checkimg(image)
  nb_classes = 17L # number of classes (+1 for background)

  if (register) {
    if (verbose) {
      message("Registering to Template")
    }
    L = register_ctseg(
      image = image,
      verbose = verbose,
      ...)
    image = L$template_space
    reg = L$registration
    native_space = L$native_space
    fname = tempfile(fileext = ".nii.gz")
    neurobase::writenii(image, fname)
  } else {
    dimg = dim(nim)
    if (!all(dimg[1:2] %in% 512)) {
      stop("Dimensions of image must be 512x512xZ, use register = TRUE")
    }
    L = list(native_space = nim)
  }

  # load predict
  # testimage = nb$load(fname)$get_fdata()
  testimage = array(nim, dim = dim(nim))
  # affine = nb$load(fname)$affine
  # header = nb$load(fname)$header
  numimgs = dim(testimage)[3]
  testimage = np$moveaxis(testimage, -1L, 0L)
  testimage = np$expand_dims(testimage, -1L)

  if (verbose) {
    message("Loading HeadCTSegmentation Model")
  }

  model = load_ctseg_model(outdir = outdir, type = type[1])

  if (verbose) {
    message("Prediction")
  }
  predlabel = model$predict(testimage)

  outfile = tempfile(fileext = ".nii.gz")
  # save predict
  predlabel = np$reshape(predlabel, c(numimgs,512L,512L, nb_classes))
  predlabel = np$argmax(predlabel, axis = 3L)
  predlabel = np$moveaxis(predlabel, 0L, -1L)
  outimg = neurobase::copyNIfTIHeader(img = nim, arr = predlabel)
  outimg = neurobase::datatyper(outimg)
  # outimg = nb$Nifti1Image(predlabel, affine, header)
  # nb$save(outimg, outfile)
  # outimg = neurobase::readnii(outfile)

  if (register) {
    if (verbose) {
      message("Projecting back into Native Space")
    }
    native = extrantsr::ants_apply_transforms(
      fixed = native_space,
      moving = outimg,
      interpolator = "nearestNeighbor",
      transformlist = reg$invtransforms,
      verbose = verbose > 1,
      whichtoinvert = 1)
    L$registration_matrix = reg$fwdtransforms
    L$registration = NULL
    L$native_prediction = native
    L$template_prediction = outimg
  } else {
    L$native_prediction = outimg
  }
  return(L)

}

#' @rdname predict_deepbleed
#' @export
register_ctseg = function(
  image,
  mask = NULL,
  verbose = TRUE, ...) {

  if (!requireNamespace("ANTsRCore", quietly = TRUE)) {
    stop("ANTsRCore is required for registration, use register = FALSE")
  }
  if (!requireNamespace("extrantsr", quietly = TRUE)) {
    stop("extrantsr is required for registration, use register = FALSE")
  }
  image = check_nifti(image)
  template.file = system.file(
    'template_with_skull.nii.gz',
    package = 'ctseg')
  if (verbose) {
    message("Registration")
  }
  reg = extrantsr::registration(
    image,
    template.file = template.file,
    typeofTransform = "Rigid",
    affSampling = 64,
    verbose = verbose > 1)
  temp_space = reg$outfile

  L = list(
    template_space = temp_space,
    registration = reg,
    native_space = image
  )
}