prepare_train_data = function(task, frac = 0, standardize_time = FALSE, log_duration = FALSE,
                              with_mean = TRUE, with_std = TRUE, discretise = FALSE, cuts = 10,
                              cutpoints = NULL, scheme = "equidistant", cut_min = 0, model) {

  x_train = task$data(cols = task$feature_names)
  y_train = task$data(cols = task$target_names)

  conv = ifelse(discretise, "int64", "float32")

  if (frac) {
    val = sample(seq_len(task$nrow), task$nrow * frac)
    x_val = x_train[val, ]
    y_val = y_train[val, ]
    x_train = x_train[-val, ]
    y_train = y_train[-val, ]
  }

  y_train = reticulate::r_to_py(y_train)

  x_train = reticulate::r_to_py(x_train)$values$astype('float32')
  y_train = reticulate::tuple(y_train[task$target_names[1L]]$values$astype(conv),
                              y_train[task$target_names[2L]]$values$astype(conv))


  if (frac) {
    x_val = reticulate::r_to_py(x_val)$values$astype('float32')
    y_val = reticulate::r_to_py(y_val)
    y_val = reticulate::tuple(y_val[task$target_names[1L]]$values$astype(conv),
                              y_val[task$target_names[2L]]$values$astype(conv))
  }

  ret = list(x_train = x_train, y_train = y_train)

  if (standardize_time || discretise) {
    if (standardize_time) {
      labtrans = mlr3misc::invoke(
        pycox$models$CoxTime$label_transform,
        log_duration = log_duration,
        with_mean = with_mean,
        with_std = with_std
      )
    } else {
      if (!is.null(cutpoints)) {
        cuts = cutpoints
      }
      if (model == "DeepHit") {
        labtrans = mlr3misc::invoke(
          pycox$models$DeepHitSingle$label_transform,
          cuts = as.integer(cuts),
          scheme = scheme,
          min_ = as.integer(cut_min)
        )
      } else if (model == "LH") {
        labtrans = mlr3misc::invoke(
          pycox$models$LogisticHazard$label_transform,
          cuts = as.integer(cuts),
          scheme = scheme,
          min_ = as.integer(cut_min)
        )
      } else if (model == "PCH") {
        labtrans = mlr3misc::invoke(
          pycox$models$PCHazard$label_transform,
          cuts = as.integer(cuts),
          scheme = scheme,
          min_ = as.integer(cut_min)
        )
      }

    }

    y_train = reticulate::r_to_py(labtrans$fit_transform(y_train[0], y_train[1]))

    if (model == "DeepHit") {
      y_train = reticulate::tuple(
        y_train[0]$astype(conv),
        y_train[1]$astype(conv)
      )
    } else if (model == "LH") {
      y_train = reticulate::tuple(
        y_train[0]$astype('int64'),
        y_train[1]$astype('float32')
      )
    } else if (model == "PCH") {
      y_train = reticulate::tuple(
        y_train[0]$astype('int64'),
        y_train[1]$astype('float32'),
        y_train[2]$astype('float32')
      )
    }

    ret$y_train = y_train
    ret$labtrans = labtrans

    if (frac) {
      y_val = reticulate::r_to_py(labtrans$transform(y_val[0], y_val[1]))

      if (model == "DeepHit") {
        y_val = reticulate::tuple(
          y_val[0]$astype(conv),
          y_val[1]$astype(conv)
        )
      } else if (model == "LH") {
        y_val = reticulate::tuple(
          y_val[0]$astype('int64'),
          y_val[1]$astype('float32')
        )
      } else if (model == "PCH") {
        y_val = reticulate::tuple(
          y_val[0]$astype('int64'),
          y_val[1]$astype('float32'),
          y_val[2]$astype('float32')
        )
      }
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

get_activation = function(activation = "relu", construct = TRUE, alpha = 1, dim = NULL, lambd = 0.5,
                          min_val = -1, max_val = 1, negative_slope = 0.01,
                          num_parameters = 1L, init = 0.25, lower = 1/8, upper = 1/3,
                          beta = 1, threshold = 20, value = 20) {
  act = torch$nn$modules$activation

  if (construct) {
    activation = switch(activation,
                        celu = act$CELU(alpha),
                        elu = act$ELU(alpha),
                        gelu = act$GELU(),
                        glu = act$GLU(as.integer(dim)),
                        hardshrink = act$Hardshrink(lambd),
                        hardsigmoid = act$Hardsigmoid(),
                        hardswish = act$Hardswish(),
                        hardtanh = act$Hardtanh(as.integer(min_val), as.integer(max_val)),
                        relu6 = act$ReLU6(),
                        leakyrelu = act$LeakyReLU(negative_slope),
                        logsigmoid = act$LogSigmoid(),
                        logsoftmax = act$LogSoftmax(as.integer(dim)),
                        prelu = act$PReLU(num_parameters = as.integer(num_parameters), init = init),
                        rrelu = act$RReLU(lower, upper),
                        relu = act$ReLU(),
                        selu = act$SELU(),
                        sigmoid = act$Sigmoid(),
                        softmax = act$Softmax(as.integer(dim)),
                        softmax2d = act$Softmax2d(),
                        softmin = act$Softmin(as.integer(dim)),
                        softplus = act$Softplus(beta, threshold),
                        softshrink = act$Softshrink(lambd),
                        softsign = act$Softsign(),
                        tanh = act$Tanh(),
                        tanhshrink = act$Tanhshrink(),
                        threshold = act$Threshold(threshold, value)
    )
  } else {
    activation = switch(activation,
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

initializers = c("uniform", "normal", "constant", "xavier_uniform", "xavier_normal",
                 "kaiming_uniform", "kaiming_normal", "orthogonal")

get_init = function(init = "uniform", a = 0, b = 1, mean = 0, std = 1, gain = 1,
                    mode = c("fan_in", "fan_out"), non_linearity = c("leaky_relu", "relu")) {

  switch(init,
    uniform = paste0("torch.nn.init.uniform_(m.weight, ", a, ", ", b, ")"),
    normal = paste0("torch.nn.init.normal_(m.weight, ", mean, ", ", std, ")"),
    constant = paste0("torch.nn.init.constant_(m.weight, ", val, ")"),
    xavier_uniform = paste0("torch.nn.init.xavier_uniform_(m.weight, ", gain, ")"),
    xavier_normal = paste0("torch.nn.init.xavier_normal_(m.weight, ", gain, ")"),
    kaiming_uniform = paste0("torch.nn.init.kaiming_uniform_(m.weight, ", a, ", '",
                             match.arg(mode), "', '", match.arg(non_linearity), "')"),
    kaiming_normal = paste0("torch.nn.init.kaiming_normal_(m.weight, ", a, ", '",
                            match.arg(mode), "', '", match.arg(non_linearity), "')"),
    orthogonal = paste0("torch.nn.init.orthogonal_(m.weight, ", gain, ")")
  )
}
