prepare_train_data = function(task, frac = 0, standardize_time = FALSE, log_duration = FALSE,
                              with_mean = TRUE, with_std = TRUE) {

  x_train = task$data(cols = task$feature_names)
  y_train = task$data(cols = task$target_names)

  if (frac) {
    val = sample(seq_len(task$nrow), task$nrow * frac)
    x_val = x_train[val, ]
    y_val = y_train[val, ]
    x_train = x_train[-val, ]
    y_train = y_train[-val, ]
  }

  y_train = reticulate::r_to_py(y_train)

  x_train = reticulate::r_to_py(x_train)$values$astype('float32')
  y_train = reticulate::tuple(y_train[task$target_names[1L]]$values$astype('float32'),
                              y_train[task$target_names[2L]]$values$astype('float32'))

  if (frac) {
    x_val = reticulate::r_to_py(x_val)$values$astype('float32')
    y_val = reticulate::r_to_py(y_val)
    y_val = reticulate::tuple(y_val[task$target_names[1L]]$values$astype('float32'),
                              y_val[task$target_names[2L]]$values$astype('float32'))
  }

  ret = list(x_train = x_train, y_train = y_train)

  if (standardize_time) {
    labtrans = mlr3misc::invoke(
      pycox$models$CoxTime$label_transform,
      log_duration = log_duration,
      with_mean = with_mean,
      with_std = with_std
    )
    y_train = reticulate::r_to_py(labtrans$fit_transform(y_train[0], y_train[1]))
    y_train = reticulate::tuple(
      y_train[0]$astype('float32'),
      y_train[1]$astype('float32')
    )

    ret$y_train = y_train
    ret$labtrans = labtrans

    if (frac) {
      y_val = reticulate::r_to_py(labtrans$fit_transform(y_val[0], y_val[1]))
      y_val = reticulate::tuple(
        y_val[0]$astype('float32'),
        y_val[1]$astype('float32')
      )
    }
  }

  if (frac) {
    ret$val = torchtuples$tuplefy(x_val, y_val)
  }

  return(ret)
}

activations = c("celu", "elu", "gelu", "glu", "hardshrink", "hardsigmoid", "hardswish",
                "hardtanh", "relu6", "leakyrelu", "logsigmoid", "logsoftmax",
                "prelu", "rrelu", "relu", "selu", "sigmoid",
                "softmax", "softmax2d", "softmin", "softplus", "softshrink", "softsign",
                "tanh", "tanhshrink", "threshold")

get_activation = function(activation = "relu") {
  act = torch$nn$modules$activation
  switch(activation,
         celu = act$CELU,
         elu = act$ELU,
         gelu = act$GELU,
         glu = act$GLU,
         hardshrink = act$Hardshrink,
         hardsigmoid = act$Hardsigmoid,
         hardswish = act$Hardswish,
         hardtanh = act$Hardtanh,
         relu6 = act$ReLU6,
         leakyrelu = act$LeakyReLU,
         logsigmoid = act$LogSigmoid,
         logsoftmax = act$LogSoftmax,
         prelu = act$PReLU,
         rrelu = act$RReLU,
         relu = act$ReLU,
         selu = act$SELU,
         sigmoid = act$Sigmoid,
         softmax = act$Softmax,
         softmax2d = act$Softmax2d,
         softmin = act$Softmin,
         softplus = act$Softplus(beta, threshold),
         softshrink = act$Softshrink(lambd),
         softsign = act$Softsign(),
         tanh = act$Tanh(),
         tanhshrink = act$Tanhshrink(),
         threshold = act$Threshold(threshold, value)
  )
}

optimizers = c("adadelta", "adagrad", "adam", "adamax", "adamw", "asgd",
           "rmsprop", "rprop", "sgd", "sparse_adam")

get_optim = function(optimizer = "adam", net, rho = 0.9, eps = 1e-8, lr = 1,
                     weight_decay = 0, learning_rate = 1e-2, lr_decay = 0,
                     betas = c(0.9, 0.999), amsgrad = FALSE,
                     lambd = 1e-4, alpha = 0.75, t0 = 1e6,
                     momentum = 0, centered = TRUE, etas = c(0.5, 1.2),
                     step_sizes = c(1e-6, 50), dampening = 0,
                     nesterov = FALSE) {
  opt = torch$optim
  params = net$parameters()

  switch(optimizer,
    adadelta = opt$Adadelta(params, rho, eps, lr, weight_decay),
    adagrad = opt$Adagrad(params, learning_rate, lr_decay, weight_decay, eps),
    adam = opt$Adam(params, learning_rate, betas, eps, weight_decay, amsgrad),
    adamax = opt$Adamax(params, learning_rate, betas, eps, weight_decay),
    adamw = opt$AdamW(params, learning_rate, betas, eps, weight_decay, amsgrad),
    asgd = opt$ASGD(params, learning_rate, lambd, alpha, t0, weight_decay),
    rmsprop = opt$RMSprop(params, learning_rate, momentum, alpha, eps, centered,
                          weight_decay),
    rprop = opt$Rprop(params, learning_rate, etas, step_sizes),
    sgd = opt$SGD(params, learning_rate, momentum, weight_decay, dampening, nesterov),
    sparse_adam = opt$SparseAdam(params, learning_rate, betas, eps)
  )
}