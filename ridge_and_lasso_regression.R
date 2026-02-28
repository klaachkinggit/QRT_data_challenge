#Ridge and Lasso logistc regression
library(glmnet)
set.seed(1)

data_dir <- "~/Desktop/QRT/data"
output_dir <- "~/Desktop/QRT/submission"

# ---------- 1) Read files (all together) ----------
X_train_raw <- read.csv(file.path(data_dir, "X_train.csv"), row.names="ROW_ID", check.names=FALSE)
X_test_raw  <- read.csv(file.path(data_dir, "X_test.csv"),  row.names="ROW_ID", check.names=FALSE)
y_train_raw <- read.csv(file.path(data_dir, "y_train.csv"), row.names="ROW_ID", check.names=FALSE)

# Align train rows by ROW_ID (robust)
common_ids <- intersect(rownames(X_train_raw), rownames(y_train_raw))
X_train_raw <- X_train_raw[common_ids, , drop=FALSE]
y_train_raw <- y_train_raw[common_ids, , drop=FALSE]
stopifnot(identical(rownames(X_train_raw), rownames(y_train_raw)))

Y <- y_train_raw[[1]]
Y_sign <- as.integer(Y > 0)  # classification target

# ---------- 2) TS-grouped folds (built from raw train BEFORE dropping TS) ----------
K <- 5
ts <- X_train_raw$TS
ts_unique <- unique(ts)

ts_fold <- sample(rep(1:K, length.out=length(ts_unique)))
names(ts_fold) <- ts_unique
foldid <- ts_fold[as.character(ts)]  # one fold id per row

# ---------- 3) Preprocessing (single place, train-fitted) ----------
cols_to_drop <- c("SIGNED_VOLUME_1","SIGNED_VOLUME_20","MEDIAN_DAILY_TURNOVER","TS")

# Drop columns that exist (train/test separately)
drop_train <- intersect(cols_to_drop, colnames(X_train_raw))
drop_test  <- intersect(cols_to_drop, colnames(X_test_raw))

X_train_feat <- X_train_raw[, !(colnames(X_train_raw) %in% drop_train), drop=FALSE]
X_test_feat  <- X_test_raw[,  !(colnames(X_test_raw)  %in% drop_test),  drop=FALSE]

# GROUP as factor with train levels
if ("GROUP" %in% colnames(X_train_feat)) {
  X_train_feat$GROUP <- factor(X_train_feat$GROUP)
}
if ("GROUP" %in% colnames(X_test_feat) && "GROUP" %in% colnames(X_train_feat)) {
  X_test_feat$GROUP <- factor(X_test_feat$GROUP, levels = levels(X_train_feat$GROUP))
}

# Mean-impute numeric columns using TRAIN means only
num_cols_train <- sapply(X_train_feat, is.numeric)
means <- colMeans(X_train_feat[, num_cols_train, drop=FALSE], na.rm=TRUE)

for (nm in names(means)) {
  X_train_feat[[nm]][is.na(X_train_feat[[nm]])] <- means[[nm]]
  if (nm %in% colnames(X_test_feat)) {
    X_test_feat[[nm]][is.na(X_test_feat[[nm]])] <- means[[nm]]
  }
}

# ---------- 4) Build model matrices and align columns ----------
X <- model.matrix(~ . - 1, data = X_train_feat)
X_test <- model.matrix(~ . - 1, data = X_test_feat)

# Add any missing columns to test, then order to match train
missing_cols <- setdiff(colnames(X), colnames(X_test))
if (length(missing_cols) > 0) {
  X_test <- cbind(
    X_test,
    matrix(0, nrow(X_test), length(missing_cols),
           dimnames = list(NULL, missing_cols))
  )
}
X_test <- X_test[, colnames(X), drop=FALSE]

# ---------- 5) Train Ridge (alpha=0) ----------
cv_ridge <- cv.glmnet(
  X, Y_sign,
  alpha = 0,
  family = "binomial",
  type.measure = "class",
  foldid = foldid
)

ridge_lambda_min <- cv_ridge$lambda.min
ridge_lambda_1se <- cv_ridge$lambda.1se
ridge_cv_acc <- 1 - min(cv_ridge$cvm)

cat("Ridge CV accuracy:", ridge_cv_acc, "\n")
cat("Ridge lambda.min:", ridge_lambda_min, "\n")
cat("Ridge lambda.1se:", ridge_lambda_1se, "\n")

# ---------- 6) Train Lasso (alpha=1) ----------
cv_lasso <- cv.glmnet(
  X, Y_sign,
  alpha = 1,
  family = "binomial",
  type.measure = "class",
  foldid = foldid
)

lasso_lambda_min <- cv_lasso$lambda.min
lasso_lambda_1se <- cv_lasso$lambda.1se
lasso_cv_acc <- 1 - min(cv_lasso$cvm)

cat("Lasso CV accuracy:", lasso_cv_acc, "\n")
cat("Lasso lambda.min:", lasso_lambda_min, "\n")
cat("Lasso lambda.1se:", lasso_lambda_1se, "\n")

# ---------- Predict on test + save (0/1 only) ----------

# Predicted probabilities that return > 0
p_ridge <- as.numeric(predict(cv_ridge, newx = X_test, s = "lambda.min", type = "response"))
p_lasso <- as.numeric(predict(cv_lasso, newx = X_test, s = "lambda.min", type = "response"))

# Convert to 0/1 predictions (1 = positive, 0 = negative)
pred_ridge_01 <- as.integer(p_ridge > 0.5)
pred_lasso_01 <- as.integer(p_lasso > 0.5)

# Save ONLY 0/1 predictions
ridge_out <- data.frame(
  ROW_ID = rownames(X_test_raw),
  PREDICTION = pred_ridge_01,
  check.names = FALSE
)

lasso_out <- data.frame(
  ROW_ID = rownames(X_test_raw),
  PREDICTION = pred_lasso_01,
  check.names = FALSE
)

write.csv(ridge_out, file.path(output_dir, "ridge_pred_01.csv"), row.names = FALSE)
write.csv(lasso_out, file.path(output_dir, "lasso_pred_01.csv"), row.names = FALSE)

cat("Saved:\n",
    file.path(output_dir, "ridge_pred_01.csv"), "\n",
    file.path(output_dir, "lasso_pred_01.csv"), "\n")

