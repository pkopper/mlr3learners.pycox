library(mlr3learners.pycox)

test_that("surv.coxtime", {
  learner = lrn("surv.coxtime")
  fun = akritas
  exclude = c(
    "formula", # handled via mlr3
    "data", # handled via mlr3
    "time_variable", # handled via mlr3
    "status_variable", # handled via mlr3
    "x", # handled via mlr3
    "y", # handled via mlr3
    "..." # not used
  )

  ParamTest = run_paramtest(learner, fun, exclude)
  expect_true(ParamTest, info = paste0("\nMissing parameters:\n",
    paste0("- '", ParamTest$missing, "'", collapse = "\n")))
})

# example for checking a predict function of a learner
test_that("surv.coxtime_predict", {
  learner = lrn("surv.coxtime")
  fun = predict
  exclude = c(
    "object", # handled via mlr3
    "newdata", # handled via mlr3
    "type", # handled via mlr3
    "distr6", # handled via mlr3
    "times", # all times returned
    "..." # not used
  )

  ParamTest = run_paramtest(learner, fun, exclude)
  expect_true(ParamTest, info = paste0("\nMissing parameters:\n",
    paste0("- '", ParamTest$missing, "'", collapse = "\n")))
})
