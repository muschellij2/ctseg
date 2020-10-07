#' Segment Brain from Scan with CTBET CNN
#'
#' @param image Image or set of images to segment
#' @param weight_file Model weight file, see \code{\link{download_ctbet_model}}
#' @param dimension what dimension model is this?  Either 2 or 3
#'
#' @return A list of \code{nifti}
#' @export
#'
#' @examples
predict_ctbet = function(
  image,
  weight_file = NULL,
  dimension = 2L) {
  # #' @param data_augmentation Should data augmentation be done before
  #' prediction?

  if (!check_requirements()) {
    warning("Not all modules may not be installed for ctseg")
  }
  if (is.null(weight_file)) {
    weight_file = download_ctbet_model()
  }
  stopifnot(file.exists(weight_file))


  image = neurobase::checkimg(image)
  if (anyDuplicated(image) > 0 ||
      anyDuplicated(basename(image)) > 0) {
    i = 0
    base_fnames = sapply(image, function(x) {
      i <<- i +1
      sub("[.]nii", paste0("_", i, ".nii"), x)
    })
    new_images = file.path(tempdir(), basename(base_fnames))
    on.exit(unlink(new_images))
    file.copy(image, new_images)
    image = new_images
  }
  if (anyDuplicated(image) > 0 ||
      anyDuplicated(basename(image)) > 0) {
    stop("Duplicated images in the prediction - use unique")
  }

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

  return(masks)
}
