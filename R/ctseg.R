predict_image = function(img, outdir, type, verbose = TRUE) {
  nb_classes = 17L # number of classes (+1 for background)

  np = reticulate::import("numpy")

  testimage = array(img, dim = dim(img))
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

  if (inherits(img, "nifti")) {
    outimg = neurobase::copyNIfTIHeader(img = img, arr = predlabel)
    outimg = neurobase::datatyper(outimg)
  }
  if (inherits(img, "niftiImage")) {
    outimg = RNifti::asNifti(reference = img, x = predlabel)
  }
  return(outimg)
}

#' Predict Head Segmentation of CT images
#' @param image image to segment using `HeadCTSegmentation` model
#' @param register should image registration be done for prediction? The
#' prediction will be inverted back into native space
#' @param outdir Output directory for `CTSeg` model,
#'  passed to \code{\link{load_ctseg_model}}
#' @param type Specific model to download,
#'  passed to \code{\link{load_ctseg_model}}
#' @param ... not used
#' @param verbose print diagnostic messages
#'
#' @rdname ctseg
#' @export
#' @examples
#' url = paste0("https://github.com/jasonccai/HeadCTSegmentation/",
#' "raw/master/image_data_predict/1.nii.gz")
#' image = tempfile(fileext = ".nii.gz")
#' utils::download.file(url, destfile = image, quiet = FALSE)
#' out = predict_ctseg(image, register = FALSE)
#' \dontrun{
#' reg_out = predict_ctseg(image, register = TRUE, verbose = 2,
#' register_type = "NiftyReg")
#' reg_out_nifty = predict_ctseg_nifty(image, register = TRUE, verbose = 2,
#' register_type = "ANTsRCore")
#' }
#'
predict_ctseg = function(image,
                         register = FALSE,
                         register_type = c("NiftyReg", "ANTsRCore"),
                         verbose = TRUE,
                         ...,
                         outdir = NULL,
                         type = c("NPH", "Normal_Only")
) {

    if (register) {
      register_type = match.arg(register_type)
      if (register_type == "NiftyReg") {
        out = predict_ctseg_nifty(
          image,
          register = register,
          verbose = verbose,
          ...,
          outdir = outdir,
          type = type
        )
      }
      if (register_type == "ANTsRCore") {
        out = predict_ctseg_ants(
          image,
          register = register,
          verbose = verbose,
          ...,
          outdir = outdir,
          type = type
        )
      }
    } else {
      out = predict_ctseg_ants(image,
                         register = register,
                         verbose = verbose,
                         ...,
                         outdir = outdir,
                         type = type
      )
    }
  return(out)
}

#' @rdname ctseg
#' @export
predict_ctseg_ants = function(image,
                              register = FALSE,
                              register_type = c("NiftyReg", "ANTsRCore"),
                              verbose = TRUE,
                              ...,
                              outdir = NULL,
                              type = c("NPH", "Normal_Only")
) {

  type = match.arg(type)
  if (!check_requirements()) {
    warning("Not all modules may not be installed for ctseg")
  }
  if (verbose) {
    message("Loading Python Modules")
  }
  nb = reticulate::import("nibabel")
  np = reticulate::import("numpy")

  nim = neurobase::check_nifti(image)
  fname = neurobase::checkimg(image)

  if (register) {
    if (verbose) {
      message("Registering to Template")
    }
    L = register_ctseg(
      image = image,
      verbose = verbose,
      ...)
    image = L$template_space
    fname = tempfile(fileext = ".nii.gz")
    neurobase::write_nifti(image, fname)
  } else {
    L = list(native_space = nim,
             template_space = nim)
  }
  dimg = dim(L$template_space)
  if (!all(dimg[1:2] %in% 512)) {
    stop("Dimensions of image must be 512x512xZ, use register = TRUE")
  }

  outimg = predict_image(
    img =  L$template_space,
    outdir = outdir,
    type = type[1],
    verbose = verbose)


  if (register) {
    if (verbose) {
      message("Projecting back into Native Space")
    }
    native = extrantsr::ants_apply_transforms(
      fixed = L$native_space,
      moving = outimg,
      interpolator = "nearestNeighbor",
      transformlist = L$registration$invtransforms,
      verbose = verbose > 1,
      whichtoinvert = 1)
    L$registration_matrix = L$registration$fwdtransforms
    L$registration = NULL
    L$native_prediction = native
    L$template_prediction = outimg
  } else {
    L$native_prediction = outimg
  }
  return(L)

}

#' @rdname ctseg
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
  image = neurobase::check_nifti(image)
  template.file = system.file(
    'short_template_with_skull.nii.gz',
    package = 'ctseg')
  if (verbose) {
    message("Registration")
  }
  reg = extrantsr::registration(
    image,
    template.file = template.file,
    typeofTransform = "Rigid",
    verbose = verbose > 1)
  temp_space = reg$outfile
  temp_space[is.na(temp_space)] = -1024
  temp_space[temp_space == 0] = -1024

  L = list(
    template_space = temp_space,
    registration = reg,
    native_space = image
  )
}






#' @rdname ctseg
#' @export
register_ctseg_nifty = function(
  image,
  verbose = TRUE, ...) {

  if (!requireNamespace("RNiftyReg", quietly = TRUE)) {
    stop("RNiftyReg is required for registration, use register = FALSE")
  }
  image = RNifti::asNifti(image)
  template.file = system.file(
    'short_template_with_skull.nii.gz',
    package = 'ctseg')
  template = RNifti::asNifti(template.file)
  if (verbose) {
    message("Registration")
  }
  reg = RNiftyReg::niftyreg(
    source = image,
    target = template,
    scope = "rigid",
    verbose = verbose)

  temp_space = reg$image
  temp_space[is.na(temp_space)] = -1024
  temp_space[temp_space == 0] = -1024

  L = list(
    template_space = temp_space,
    registration = reg,
    native_space = image
  )
}


#' @rdname ctseg
#' @export
predict_ctseg_nifty = function(image,
                               register = FALSE,
                               verbose = TRUE,
                               ...,
                               outdir = NULL,
                               type = c("NPH", "Normal_Only")
) {

  type = match.arg(type)
  if (!check_requirements()) {
    warning("Not all modules may not be installed for ctseg")
  }
  if (verbose) {
    message("Loading Python Modules")
  }
  nb = reticulate::import("nibabel")
  np = reticulate::import("numpy")

  nim = RNifti::asNifti(image)
  fname = tempfile(fileext = ".nii.gz")
  RNifti::writeNifti(nim, fname)

  if (register) {
    if (verbose) {
      message("Registering to Template")
    }
    L = register_ctseg_nifty(
      image = image,
      verbose = verbose > 1,
      ...)
    fname = tempfile(fileext = ".nii.gz")
    RNifti::writeNifti(L$template_space, fname)
  } else {
    L = list(native_space = nim,
             template_space = nim)
  }
  dimg = dim(L$template_space)
  if (!all(dimg[1:2] %in% 512)) {
    stop("Dimensions of image must be 512x512xZ, use register = TRUE")
  }

  # load predict
  outimg = predict_image(
    img =  L$template_space,
    outdir = outdir,
    type = type[1],
    verbose = verbose)

  if (register) {
    if (verbose) {
      message("Projecting back into Native Space")
    }
    native <- RNiftyReg::applyTransform(
      transform = RNiftyReg::reverse(L$registration),
      x = outimg,
      interpolation = 0L)
    # native = native$image

    L$registration_matrix = RNiftyReg::forward(L$registration)
    L$registration = NULL
    L$native_prediction = native
    L$template_prediction = outimg
  } else {
    L$native_prediction = outimg
  }
  return(L)
}
