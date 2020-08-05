#' @title Build a Pytorch Multilayer Perceptron
#' @description Utility function to build an MLP with a choice of activation function and weight
#' initialization with optional dropout and batch normalization.
#' @details This function is a helper for R users with less Python experience. Currently it is
#' limited to simple MLPs. More advanced networks will require manual creation with
#' \CRANpkg{reticulate.}
#'
#' @param n_in `(integer(1))`\cr Number of input features.
#' @param n_out `(integer(1))`\cr Number of targets.
#' @param nodes `(numeric())`\cr Hidden nodes in network, each element in vector represents number
#' of hidden nodes in respective layer.
#' @param activation `(character(1)|list())`\cr Activation function, can either be a single character
#' and the same function is used in all layers, or a list of length `length(nodes)`. See
#' [get_activation] for options.
#' @param act_pars `(list())`\cr Passed to [get_activation].
#' @param dropout `(numeric())`\cr Optional dropout layer, if `NULL` then no dropout layer added
#' otherwise either a single numeric which will be added to all layers or a vector of differing
#' drop-out amounts.
#' @param bias `(logical(1))`\cr If `TRUE` (default) then a bias parameter is added to all linear
#' layers.
#' @param batch_norm `(logical(1))`\cr If `TRUE` (default) then batch normalisation is applied
#' to all layers.
#' @param batch_pars `(list())`\cr Parameters for batch normalisation, see
#' `reticulate::py_help(torch$nn$BatchNorm1d)`.
#' @param init `(character(1))`\cr Weight initialization method. See
#' [get_init] for options.
#' @param init_pars `(list())`\cr Passed to [get_init].
#'
#' @examples
#' build_pytorch_net(10, 1)
#'
#' build_pytorch_net(n_in = 10, n_out = 1, nodes = c(4, 4, 4), activation = "elu",
#' act_pars = list(alpha = 0.5), dropout = c(0.2, 0.1, 0.6),
#' batch_norm = TRUE, init = "kaiming_normal", init_pars = list(non_linearity = "relu"))
#'
#' @export
build_pytorch_net = function(n_in, n_out,
                             nodes = c(32, 32), activation = "relu",
                             act_pars = list(),  dropout = 0.1,
                             bias = TRUE, batch_norm = TRUE, batch_pars = list(eps = 1e-5,
                             momentum = 0.1, affine = TRUE), init = "uniform", init_pars = list()) {
  nodes = as.integer(nodes)
  n_in = as.integer(n_in)
  n_out = as.integer(n_out)
  nn = torch$nn
  lng = length(nodes)

  if (length(activation) == 1) {
    checkmate::assert_character(activation)
    activation = rep(list(mlr3misc::invoke(get_activation,
                                           activation = activation,
                                           construct = TRUE,
                                           .args = act_pars)), lng)
  } else {
    checkmate::assert_character(activation, len = lng)
    activation = lapply(activation, function(x) {
      mlr3misc::invoke(get_activation,
                       activation = x,
                       construct = TRUE,
                       .args = act_pars)
    })
  }


  if (is.null(dropout) || length(dropout) == 1) {
    dropout = rep(list(dropout), lng)
  } else {
    checkmate::assert_numeric(dropout, len = lng)
  }

  add_module = function(net, id, n_int, n_out, act, dropout) {
    # linear trafo
    net$add_module(paste0("L", id), nn$Linear(n_int, n_out, bias))
    # activation
    net$add_module(paste0("A", id), act)
    # batch normalisation
    if (batch_norm) {
      net$add_module(paste0("BN", id), mlr3misc::invoke(nn$BatchNorm1d,
                                                      num_features = n_out,
                                                      .args = batch_pars))
    }
    # dropout layer
    if (!is.null(dropout)) {
      net$add_module(paste0("D", id), nn$Dropout(dropout))
    }

    return(net)
  }

  # input layer
  net = nn$Sequential()
  add_module(net, 0, n_in, nodes[1], activation[[1]], dropout[[1]])

  # hidden layers
  for (i in seq_along(nodes)) {
    if (i < length(nodes)) {
      add_module(net, i, nodes[i], nodes[i + 1], activation[[i]], dropout[[i]])
    } else {
      # output layer
      net$add_module(as.character(length(nodes)), nn$Linear(nodes[i], n_out, bias))
    }
  }

  init = mlr3misc::invoke(get_init, init = init, .args = init_pars)
  reticulate::py_run_string(
    paste0("import torch
def init_weights(m):
      if type(m) == torch.nn.Linear:",
        init))

  net$apply(reticulate::py$init_weights)

  return(net)
}
