#' @rdname ctseg
#' @param image image to segment using `HeadCTSegmentation` model
#' @param mask brain mask image
#' @param verbose print diagnostic messages
#' @param ... additional arguments to send to
#' \code{\link{CT_Skull_Stripper_mask}}
#' @export
predict_ctseg = function(image,
                         mask = NULL,
                         verbose = TRUE,
                         ...,
                         outdir = NULL,
                         type = c("NPH", "Normal_Only")
) {

  if (verbose) {
    message("Loading HeadCTSegmentation Model")
  }
  L = register_deepbleed(
    image = image,
    mask = mask,
    verbose = verbose,
    ...)
  image = L$template_space
  reg = L$registration
  ss = L$skull_stripped
  image = array(image, dim = c(1L, dim(image), 1L))


  vnet = load_deepbleed_model(outdir = outdir)
  if (verbose) {
    message("Prediction")
  }

  prediction = vnet$predict(image)

  arr = drop(prediction)
  arr = neurobase::copyNIfTIHeader(arr =  arr, img = L$template_space)
  if (verbose) {
    message("Projecting back into Native Space")
  }
  native = extrantsr::ants_apply_transforms(
    fixed = ss,
    moving = arr,
    interpolator = "nearestNeighbor",
    transformlist = reg$invtransforms,
    verbose = verbose > 1,
    whichtoinvert = 1)
  L$registration_matrix = reg$fwdtransforms
  L$registration = NULL
  L$native_prediction = native
  L$template_prediction = arr
  return(L)

}

#' @rdname predict_deepbleed
#' @export
register_ctseg = function(
  image,
  mask = NULL,
  verbose = TRUE, ...) {

  image = check_nifti(image)
  if (is.null(mask)) {
    if (verbose) {
      message("Skull Stripping")
    }
    mask = CT_Skull_Stripper_mask(image, verbose = verbose, ...)
    mask = mask$mask
  }
  mask = check_nifti(mask)
  if (verbose) {
    message("Masking Image")
  }
  ss = mask_img(image, mask)
  template.file = system.file(
    # 'template_with_skull.nii.gz',
    'template.nii.gz',
    package = 'ctseg')
  if (verbose) {
    message("Registration")
  }
  reg = extrantsr::registration(
    ss,
    template.file = template.file,
    typeofTransform = "Rigid",
    affSampling = 64,
    verbose = verbose > 1)
  temp_space = reg$outfile

  L = list(
    skull_stripped = ss,
    brain_mask = mask,
    template_space = temp_space,
    registration = reg
  )
}