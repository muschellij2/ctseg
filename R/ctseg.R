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


run_registration = function(image, register,
                            register_type = c("RNiftyReg", "ANTsRCore"),
                            verbose, ...) {
  if (register) {
    register_type = match.arg(register_type)
    if (verbose) {
      message("Registering to Template")
    }
    if (register_type == "RNiftyReg") {
      L = register_ctseg_nifty(
        image = image,
        verbose = verbose > 1,
        ...)
    }
    # if (register_type == "ANTsRCore") {
    #   L = register_ctseg(
    #     image = image,
    #     verbose = verbose > 1,
    #     ...)
    # }
  } else {
    L = list(native_space = image,
             template_space = resample_512(image))
  }
  L
}

resample_512 = function(image) {

  image = RNifti::asNifti(image)
  dimg = dim(image)
  if (all(dimg[1:2] == 512) &&
      dimg[3] <= 46) {
    outimg = image
    attr(outimg, "resampled") = FALSE
    return(outimg)
  }
  new_dim = c(512L, 512L, min(40, dimg[3]))
  scales = round((new_dim / dimg), digits = 5)
  outimg = RNiftyReg::rescale(image, scales = scales)
  trans = RNiftyReg::buildAffine(source = image, target = outimg)
  outimg = RNiftyReg::applyTransform(trans, image)
  attr(outimg, "resampled") = TRUE
  outimg
}

reverse_resample_512 = function(image, original_image) {
  resampled = attr(image, "resampled")
  if (is.null(resampled)) {
    resampled = TRUE
  }
  if (resampled) {
    trans = RNiftyReg::buildAffine(source = image, target = original_image)
    outimg = RNiftyReg::applyTransform(trans, image)
  } else {
    outimg = image
  }
  outimg
}

reverse_registration = function(L, outimg, register,
                                register_type = c("RNiftyReg", "ANTsRCore"),
                                verbose) {
  if (register) {
    register_type = match.arg(register_type)
    if (verbose) {
      message("Projecting back into Native Space")
    }
    # if (register_type == "ANTsRCore") {
    #
    #   native = extrantsr::ants_apply_transforms(
    #     fixed = L$native_space,
    #     moving = outimg,
    #     interpolator = "nearestNeighbor",
    #     transformlist = L$registration$invtransforms,
    #     verbose = verbose > 1,
    #     whichtoinvert = 1)
    #   L$registration_matrix = L$registration$fwdtransforms
    # }
    if (register_type == "RNiftyReg") {
      native <- RNiftyReg::applyTransform(
        transform = RNiftyReg::reverse(L$registration),
        x = outimg,
        interpolation = 0L)
      native = neurobase::check_nifti(native)
      L$registration_matrix = RNiftyReg::forward(L$registration)
    }
    L$registration = NULL
    L$native_prediction = native
    L$template_prediction = outimg
  } else {
    L$native_prediction = reverse_resample_512(
      outimg, original_image = L$native_space)
  }
  L$native_prediction = neurobase::check_nifti(L$native_prediction)
  return(L)
}


# #' @param register_type if \code{register = TRUE}, then a switch for
# #' which package to use for registration

#' Predict Head Segmentation of CT images
#' @param image image to segment using `HeadCTSegmentation` model
#' @param register should image registration be done for prediction? The
#' prediction will be inverted back into native space
#' @param outdir Output directory for `CTSeg` model,
#'  passed to \code{\link{load_ctseg_model}}
#' @param type Specific model to download,
#'  passed to \code{\link{load_ctseg_model}}
#' @param ... arguments passed to registration functions
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
#' \donttest{
#' reg_out_nifty = predict_ctseg(image, register = TRUE, verbose = 2)
#' }
predict_ctseg = function(image,
                         register = FALSE,
                         verbose = TRUE,
                         # register_type = c("RNiftyReg", "ANTsRCore"),
                         ...,
                         outdir = NULL,
                         type = c("NPH", "Normal_Only")
) {

  # #' reg_out_nifty = predict_ctseg(image, register = TRUE, verbose = 2,
  # #' register_type = "RNiftyReg")
  # #' reg_out = predict_ctseg(image, register = TRUE, verbose = 2,
  # #' register_type = "ANTsRCore")
  type = match.arg(type)
  # register_type = match.arg(register_type)
  register_type = "RNiftyReg"

  check_ct_requirements()
  if (verbose) {
    message("Loading Python Modules")
  }
  nb = reticulate::import("nibabel")
  np = reticulate::import("numpy")

  nim = neurobase::check_nifti(image)

  L = run_registration(
    image = nim,
    verbose = verbose,
    register_type = register_type,
    register = register,
    ...)
  dimg = dim(L$template_space)
  if (!all(dimg[1:2] %in% 512)) {
    stop("Dimensions of image must be 512x512xZ, use register = TRUE")
  }

  outimg = predict_image(
    img =  L$template_space,
    outdir = outdir,
    type = type[1],
    verbose = verbose)

  L = reverse_registration(L, outimg, register, register_type, verbose)
  return(L)
}

# #' @rdname ctseg
# #' @export
# register_ctseg = function(
#   image,
#   verbose = TRUE, ...) {
#
#   if (!requireNamespace("ANTsRCore", quietly = TRUE)) {
#     stop("ANTsRCore is required for registration, use register = FALSE")
#   }
#   if (!requireNamespace("extrantsr", quietly = TRUE)) {
#     stop("extrantsr is required for registration, use register = FALSE")
#   }
#   image = neurobase::check_nifti(image)
#   template.file = system.file(
#     'short_template_with_skull.nii.gz',
#     package = 'ctseg')
#   if (verbose) {
#     message("Registration")
#   }
#   reg = extrantsr::registration(
#     image,
#     template.file = template.file,
#     typeofTransform = "Rigid",
#     verbose = verbose > 1,
#     ...)
#   temp_space = reg$outfile
#   temp_space[is.na(temp_space)] = -1024
#   temp_space[temp_space == 0] = -1024
#
#   L = list(
#     template_space = temp_space,
#     registration = reg,
#     native_space = image
#   )
# }






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
    verbose = verbose,
    ...)

  temp_space = reg$image
  temp_space[is.na(temp_space)] = -1024
  temp_space[temp_space == 0] = -1024

  L = list(
    template_space = temp_space,
    registration = reg,
    native_space = image
  )
}

