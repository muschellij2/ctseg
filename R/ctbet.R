# #' @param register_type if \code{register = TRUE}, then a switch for
# #' which package to use for registration

#' Segment Brain from Scan with CTBET CNN
#'
#' @param image Image or set of images to segment
#' @param weight_file Model weight file, see \code{\link{download_ctbet_model}}
#' @param dimension what dimension model is this?  Either 2 or 3
#' @param register should image registration be done for prediction? The
#' prediction will be inverted back into native space
#' @param verbose print diagnostic messages
#' @param ... arguments passed to registration functions
#'
#' @return A list of \code{nifti}
#' @export
#'
#' @examples
#' authed = !inherits(try(googledrive::drive_auth()), "try-error")
#' if (check_ct_requirements() && authed) {
#'   url = paste0("https://github.com/jasonccai/HeadCTSegmentation/",
#'                "raw/master/image_data_predict/1.nii.gz")
#'   image = tempfile(fileext = ".nii.gz")
#'   utils::download.file(url, destfile = image, quiet = FALSE)
#'   out = predict_ctbet(image, register = FALSE)
#'   kern = array(1, dim = c(3,3,3))
#'   if (requireNamespace("mmand", quietly = TRUE)) {
#'     res = mmand::components(x = out$native_prediction, kern)
#'   }
#' }
#' \donttest{
#' if (check_ct_requirements() && authed) {
#' reg_out = predict_ctbet(image, register = TRUE, verbose = 2)
#' }
#' }
predict_ctbet = function(
  image,
  weight_file = NULL,
  register = FALSE,
  verbose = TRUE,
  dimension = 2L,
  ...) {
  # #' @param data_augmentation Should data augmentation be done before
  # #' prediction?
  # #' reg_out_nifty = predict_ctbet(image, register = TRUE, verbose = 2,
  # #' register_type = "RNiftyReg")
  # #' reg_out = predict_ctbet(image, register = TRUE, verbose = 2,
  # #' register_type = "ANTsRCore")
  check_ct_requirements()
  if (is.null(weight_file)) {
    message("Pulling Default Weight File")
    weight_file = download_ctbet_model()
    dimension = 2L
  }
  stopifnot(file.exists(weight_file))
  # register_type = c("RNiftyReg", "ANTsRCore"),
  register_type = "RNiftyReg"
  # register_type = match.arg(register_type)
  L = run_registration(
    image = image,
    verbose = verbose,
    register_type = register_type,
    register = register,
    ...)

  image = L$template_space
  image = neurobase::checkimg(image)
  stopifnot(length(image) == 1)

  sys_path = system.file(package = "ctseg")
  data_augmentation = FALSE
  if (data_augmentation) {
    auggen = reticulate::import_from_path(module = "auggen", path = sys_path)
    dataGenerator = auggen$AugmentationGenerator
    datagen = dataGenerator(
      rotation_z = 30L,
      rotation_x = 0L,
      rotation_y = 0L,
      translation_xy = 5L,
      translation_z = 0L,
      scale_xy = 0.1,
      scale_z = 0L,
      flip_h = TRUE,
      flip_v = FALSE
    )
    afold = 3L
  } else {
    datagen = ""
    afold = ""
  }


  model = reticulate::import_from_path(module = "model_CT_SS", path = sys_path)
  genUnet = model$Unet_CT_SS

  root_folder = tempfile()
  dir.create(root_folder, recursive = TRUE, showWarnings = FALSE)
  root_folder = normalizePath(root_folder)
  image_folder = "image_data"
  mask_folder = "mask_data"
  pred_folder = "prediction"
  lapply(c(image_folder, mask_folder, pred_folder), function(x) {
    dir.create(
      file.path(root_folder, x),
      recursive = TRUE,
      showWarnings = FALSE
    )
  })
  save_folder = file.path(root_folder, pred_folder)
  oLabel = ""
  file.copy(image,
            file.path(root_folder, image_folder, basename(image)),
            overwrite = TRUE)

  lr = 1e-5
  decay = 1e-6
  optimizer = "adam"

  # !!!!!!!!!!!
  # FIXME img_row, img_col to match image?
  unetSS = genUnet(
    root_folder = root_folder,
    image_folder = image_folder,
    mask_folder = mask_folder,
    save_folder = save_folder,
    pred_folder = pred_folder,
    savePredMask = TRUE,
    testLabelFlag = FALSE,
    testMetricFlag = FALSE,
    dataAugmentation = data_augmentation,
    logFileName = paste0("log_", oLabel, ".txt"),
    datagen = datagen,
    oLabel = oLabel,
    checkWeightFileName = paste0(oLabel, ".h5"),
    afold = afold,
    numEpochs = 100L,
    bs = 1L,
    nb_classes = 2L,
    sC = 2L,
    #saved class
    img_row = 512L,
    img_col = 512L,
    channel = 1L,
    classifier = "softmax",
    optimizer = optimizer,
    lr = lr,
    decay = decay,
    dtype = "float32",
    dtypeL = "uint8",
    wType = "slice",
    loss = "categorical_crossentropy",
    metric = "accuracy",
    model = "unet"
  )

  unetSS$weight_folder = dirname(weight_file)

  if (dimension %in% 3) {
    pred_fun = unetSS$Predict3D(weight_file)
  } else if (dimension %in% 2) {
    pred_fun = unetSS$Predict(weight_file)
  }

  outfile = file.path(save_folder, basename(image))
  if (!all(file.exists(outfile))) {
    warning("Something went wrong with the prediction!")
  }
  masks = neurobase::check_nifti(outfile)
  unlink(root_folder, recursive = TRUE)

  L = reverse_registration(L, outimg = masks, register, register_type, verbose)

  return(L)
}
